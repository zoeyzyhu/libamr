# pylint: disable = import-error, too-many-locals, too-many-nested-blocks, no-member, too-few-public-methods, unused-argument, unused-variable, unused-import, redefined-outer-name, undefined-variable
"""Mesh class and related functions."""

from typing import Tuple, Dict
import ray
from ray import ObjectRef
import mesh as me
import actor as ac


def launch_actors(root: me.BlockTree) -> Dict[Tuple[int, int, int], ObjectRef]:
    """Launch actors based on the tree."""
    actors = {}
    if not root.children:
        actor = ac.MeshBlockActor.remote()
        seed = root.lx3 * 2 ^ 2 + root.lx2 * 2 + root.lx1
        actor.new.remote(root, seed)
        actors[(root.lx3, root.lx2, root.lx1)] = actor
    else:
        for child in root.children:
            if child:
                actors.update(launch_actors(child))
    return actors


def orchestrate_actor(actors: Dict[Tuple[int, int, int], ObjectRef],
                      root: me.BlockTree) -> None:
    """Orchestrate the actors for a round (one time cycle)."""
    actor_dict = actors.copy()
    for logicloc, actor in actor_dict.items():
        action, point = ray.get(actor.work.remote())
        print("Center:", point)
        if action == 1:
            print("Refine actor: ********************************************", logicloc)
            refine_actor(point, root, actors)
            # print_actors(actors)

        elif action == -1:
            print("Merge actor: *********************************************", logicloc)
            merge_actor(point, root, actors)
            # print_actors(actors)

        else:
            print("No action.")
            continue
    update_ghosts_all(actors)
    for _, actor in actors.items():
        actor.reset_status.remote()


def refine_actor(point: Tuple[int, int, int], root: me.BlockTree,
                 actors: Dict[Tuple[int, int, int], ObjectRef], node=None) -> None:
    """Refine the block where the specified point locates."""
    node = root.find_node(point)
    refine_actor_chain(node, root, actors)
    update_neighbors_all(actors, root)
    new_node = root.find_node(point)


def refine_actor_chain(node: me.BlockTree, root: me.BlockTree,
                       actors: Dict[Tuple[int, int, int], ObjectRef]) -> None:
    """Refine the block where the specified point locates."""
    # Launch chidlren actors
    node.split()
    new_actors = launch_actors(node)

    logicloc = node.lx3, node.lx2, node.lx1
    parent = actors[logicloc]
    old_avg = ray.get(parent.get_avg.remote())

    # For each child actor, fill in the interior data
    tasks = []
    for new_actor in new_actors.values():
        new_actor.fill_interior_data.remote(parent)
        tasks.append(new_actor.get_avg.remote())

    new_avg = sum(ray.get(tasks)) / len(new_actors)

    for new_actor in new_actors.values():
        new_actor.fix_interior_data.remote(old_avg - new_avg)

    tasks = [new_actor.get_avg.remote() for new_actor in new_actors.values()]
    new_avg = sum(ray.get(tasks)) / len(new_actors)

    # Check whether neighbors need to be refined
    coord = ray.get(actors[logicloc].get_coord.remote())
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            for k in [-1, 0, 1]:
                if i == 0 and j == 0 and k == 0:
                    continue
                offset = (i, j, k)
                neighbors = node.find_neighbors(offset, coord)
                for nb in neighbors:
                    if nb is not None and nb.level - node.level < 0:
                        refine_actor_chain(nb, root, actors)

    # Kill the parent actor, update the actors dict
    ray.kill(parent)
    actors.pop(logicloc)
    actors.update(new_actors)


def merge_actor(point: Tuple[int, int, int], root: me.BlockTree,
                actors: Dict[Tuple[int, int, int], ObjectRef]) -> None:
    """Merge the block where the specified point locates."""
    node = root.find_node(point)
    parent = node.parent
    mergeable = True

    for child in parent.children:
        if len(child.children) > 0:
            mergeable = False
            break
        mergeable = mergeable and check_mergeability(child, root, actors)

    if mergeable:
        merge_actor_chain(node, root, actors)
        update_neighbors_all(actors, root)
    else:
        raise ValueError("The block cannot be merged.")

    new_node = root.find_node(point)


def check_mergeability(node: me.BlockTree, root: me.BlockTree,
                       actors: Dict[Tuple[int, int, int], ObjectRef]) -> bool:
    """Check whether the block can be merged."""
    mergeable = True
    logicloc = node.lx3, node.lx2, node.lx1
    coord = ray.get(actors[logicloc].get_coord.remote())
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            for k in [-1, 0, 1]:
                if i == 0 and j == 0 and k == 0:
                    continue
                offset = (i, j, k)
                neighbors = node.find_neighbors(offset, coord)
                for nb in neighbors:
                    if nb is not None and nb.level - node.level >= 1:
                        mergeable = mergeable and check_mergeability(
                            nb, root, actors)
    return mergeable


def merge_actor_chain(node: me.BlockTree, root: me.BlockTree,
                      actors: Dict[Tuple[int, int, int], ObjectRef]) -> None:
    """Merge the block where the specified point locates."""
    parent = node.parent
    logicloc = node.lx3, node.lx2, node.lx1
    if logicloc not in actors:
        return

    for child in parent.children:
        child_logicloc = child.lx3, child.lx2, child.lx1
        coord = ray.get(actors[child_logicloc].get_coord.remote())

        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                for k in [-1, 0, 1]:
                    if i == 0 and j == 0 and k == 0:
                        continue
                    offset = (i, j, k)
                    neighbors = node.find_neighbors(offset, coord)
                    for nb in neighbors:
                        if nb is not None and nb.level - node.level >= 1:
                            merge_actor_chain(nb, root, actors)

    p_lx3, p_lx2, p_lx1 = parent.lx3, parent.lx2, parent.lx1
    if (p_lx3, p_lx2, p_lx1) not in actors:
        children = parent.children.copy()
        parent.merge()
        new_actors = launch_actors(parent)

        tasks = []
        for child in children:
            logicloc = child.lx3, child.lx2, child.lx1

            tasks.append(list(new_actors.values())[
                         0].fill_interior_data.remote(actors[logicloc], logicloc))
        ray.get(tasks)

        for child in children:
            logicloc = child.lx3, child.lx2, child.lx1
            ray.kill(actors[logicloc])
            actors.pop(logicloc)

        actors.update(new_actors)


def update_neighbors_all(actors: Dict[Tuple[int, int, int], ObjectRef],
                         root: me.BlockTree) -> None:
    """Update neighbors for all actors."""
    for _, actor in actors.items():
        for o3 in [-1, 0, 1]:
            for o2 in [-1, 0, 1]:
                for o1 in [-1, 0, 1]:
                    if o3 == o2 == o1 == 0:
                        continue
                    offsets = (o3, o2, o1)
                    actor.update_neighbor.remote(offsets, root, actors)


def update_ghosts_all(actors: Dict[Tuple[int, int, int], ObjectRef]) -> None:
    """Update ghost cells for all actors."""
    tasks = {}

    for actor in actors.values():
        tasks[actor] = []
        for o3 in [-1, 0, 1]:
            for o2 in [-1, 0, 1]:
                for o1 in [-1, 0, 1]:
                    if o3 == o2 == o1 == 0:
                        continue
                    offsets = (o3, o2, o1)
                    tasks[actor].extend(
                        ray.get(actor.update_ghost.remote(offsets)))

    tasks = {actor: task for actor, task in tasks.items() if task}
    while tasks:
        print("remaining tasks:", len(tasks))
        for actor, task in tasks.items():
            tasks[actor] = ray.get(actor.wait_ghost.remote(task))
        tasks = {actor: task for actor, task in tasks.items() if task}


def print_actors(actors: Dict[Tuple[int, int, int], ObjectRef]) -> None:
    """Print the mesh block."""
    for ll in actors:
        mblock, node_id, worker_id = ray.get(actors[ll].get_data.remote())
        print(f"\nNode:{node_id}\nWorker:{worker_id}\nlogicloc:{ll}")
        print(f"size = {mblock.size}")
        mblock.print_data()


def print_actor_status(actors: Dict[Tuple[int, int, int], ObjectRef]) -> None:
    """Print the mesh block."""
    for ll in actors:
        print(ray.get(actors[ll].get_status.remote()))


def print_actor(actor: ObjectRef) -> None:
    """Print the mesh block."""
    mblock, node_id, worker_id = ray.get(actor.get_data.remote())
    print(f"\nNode:{node_id}\nWorker:{worker_id}")
    print(f"size = {mblock.size}")
    mblock.print_data()


def print_actor_coord(coord: Tuple[float, float, float], root: me.BlockTree,
                      actors: Dict[Tuple[int, int, int], ObjectRef]) -> None:
    """Print the mesh block."""
    node = root.find_node(coord)
    logicloc = node.lx3, node.lx2, node.lx1
    print_actor(actors[logicloc])


def print_actor_children(node: me.BlockTree,
                         actors: Dict[Tuple[int, int, int], ObjectRef]) -> None:
    """Print the mesh block."""
    for child in node.children:
        logicloc = child.lx3, child.lx2, child.lx1
        print_actor(actors[logicloc])
