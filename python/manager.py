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
    for _, actor in actors.items():
        action = ray.get(actor.work.remote())
        if action == "refine":
            refine_actor(actor.logicloc, root, actors)
        elif action == "merge":
            merge_actor(actor.logicloc, root, actors)
        else:
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


def refine_actor_chain(node: me.BlockTree, root: me.BlockTree,
                       actors: Dict[Tuple[int, int, int], ObjectRef]) -> None:
    """Refine the block where the specified point locates."""
    # Launch chidlren actors
    node.split()
    new_actors = launch_actors(node)

    logicloc = node.lx3, node.lx2, node.lx1
    parent = actors[logicloc]
    old_avg = ray.get(parent.get_avg.remote())
    print("old_avg=", old_avg)

    # For each child actor, fill in the interior data
    tasks = []
    for new_actor in new_actors.values():
        new_actor.fill_interior_data.remote(parent)
        tasks.append(new_actor.get_avg.remote())

    new_avg = sum(ray.get(tasks)) / len(new_actors)
    print("new_avg(before)=", new_avg)

    for new_actor in new_actors.values():
        new_actor.fix_interior_data.remote(old_avg - new_avg)

    tasks = [new_actor.get_avg.remote() for new_actor in new_actors.values()]
    new_avg = sum(ray.get(tasks)) / len(new_actors)
    print("new_avg(after)=", new_avg)

    # Get the coordinate of the parent actor to find neighbors
    coord = ray.get(parent.get_coord.remote())
    offset_neighbor_dict = ray.get(parent.get_neighbors.remote())
    neighbor_locs = []
    for _, neighbor in offset_neighbor_dict.items():
        if len(neighbor) == 0:
            continue
        if len(neighbor) == 1:
            neighbor_locs.append(neighbor[0][1])
        else:
            for _, ll in neighbor:
                neighbor_locs.append(ll)
    print("neighbor_actors=", offset_neighbor_dict)
    print("neighbor_locs=", neighbor_locs)

    neighbor_nodes = []
    for loc in neighbor_locs:
        neighbor_nodes.append(root.find_node(loc))

    for nb in neighbor_nodes:
        if nb is not None and nb.level - node.level < 0:
            refine_actor_chain(nb, root, actors)

    # Kill the parent actor, update the actors dict
    ray.kill(parent)
    actors.pop(logicloc)
    actors.update(new_actors)


def merge_actor(point: Tuple[int, int, int], root: me.BlockTree,
                actors: Dict[Tuple[int, int, int], ObjectRef]) -> None:
    """Merge the block where the specified point locates."""
    origin_node = root.find_node(point)
    print(point, origin_node)
    node_parent = origin_node.parent
    able_to_merge = True

    for node in node_parent.children:
        if len(node.children) > 0:
            able_to_merge = False
            break
        able_to_merge = able_to_merge and check_mergeability(
            node, root, actors)

    if able_to_merge:
        merge_actor_chain(origin_node, root, actors)
        update_neighbors_all(actors, root)
    else:
        raise ValueError("The block cannot merge")


def check_mergeability(node: me.BlockTree, root: me.BlockTree,
                       actors: Dict[Tuple[int, int, int], ObjectRef]) -> bool:
    """Check whether the block can be merged."""
    able_to_merge = True
    logicloc = node.lx3, node.lx2, node.lx1
    coord = ray.get(actors[logicloc].get_coord.remote())
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            for k in [-1, 0, 1]:
                if i == 0 and j == 0 and k == 0:
                    continue
                offset = (i, j, k)
                neighbors = node.find_neighbors(offset, coord)
                for neighbor_node in neighbors:
                    if neighbor_node is not None and neighbor_node.level - node.level >= 1:
                        able_to_merge = able_to_merge and check_mergeability(
                            neighbor_node, root, actors)
    return able_to_merge


def merge_actor_chain(node: me.BlockTree, root: me.BlockTree,
                      actors: Dict[Tuple[int, int, int], ObjectRef]) -> None:
    """Merge the block where the specified point locates."""
    node_parent = node.parent
    logicloc = node.lx3, node.lx2, node.lx1
    if logicloc not in actors:
        return

    for child in node_parent.children:
        child_logicloc = child.lx3, child.lx2, child.lx1
        coord = ray.get(actors[child_logicloc].get_coord.remote())

        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                for k in [-1, 0, 1]:
                    if i == 0 and j == 0 and k == 0:
                        continue
                    offset = (i, j, k)
                    neighbors = node.find_neighbors(offset, coord)
                    for neighbor_node in neighbors:
                        if neighbor_node is not None and neighbor_node.level - node.level >= 1:
                            merge_actor_chain(neighbor_node, root, actors)

    p_lx3, p_lx2, p_lx1 = node_parent.lx3, node_parent.lx2, node_parent.lx1
    if (p_lx3, p_lx2, p_lx1) not in actors:
        children = node_parent.children.copy()
        node_parent.merge()
        new_actors = launch_actors(node_parent)

        tasks = []
        for child in children:
            logicloc = child.lx3, child.lx2, child.lx1

            tasks.append(list(new_actors.values())[
                         0].fill_interior_data.remote(actors[logicloc], logicloc))
        print("task len", len(tasks))
        while tasks:
            _, tasks = ray.wait(tasks)

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
