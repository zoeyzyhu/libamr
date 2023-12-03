# pylint: disable = import-error, no-member, too-few-public-methods, unused-argument, unused-variable, unused-import, redefined-outer-name, undefined-variable
"""Mesh class and related functions."""

from typing import Tuple
import ray
from ray import ObjectRef
import mesh as me
import actor as ac


def launch_actors(root: me.BlockTree) -> dict[(int, int, int), ObjectRef]:
    """Launch actors based on the tree."""
    actors = {}
    if not root.children:
        actor = ac.MeshBlockActor.remote()
        actor.new.remote(root)
        actors[(root.lx3, root.lx2, root.lx1)] = actor
    else:
        for child in root.children:
            if child:
                actors.update(launch_actors(child))
    return actors


def refine_actor(point: Tuple[int, int, int], root: me.BlockTree,
                 actors: dict[(int, int, int), ObjectRef]) -> None:
    """Refine the block where the specified point locates."""
    node = root.find_node(point)
    node.split()

    logicloc = node.lx3, node.lx2, node.lx1
    ray.kill(actors[logicloc])
    actors.pop(logicloc)

    new_actors = launch_actors(node)
    actors.update(new_actors)

    update_neighbors_all(actors, root)


def update_neighbors_all(actors: dict[(int, int, int), ObjectRef],
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


def update_ghosts_all(actors: dict[(int, int, int), ObjectRef]) -> None:
    """Update ghost cells for all actors."""
    tasks = {}

    waiting_actors = set(actors.keys())

    while waiting_actors:
        ll = waiting_actors.pop()
        actor = actors[ll]

        ready = ray.get(actor.get_status.remote())
        if not ready:
            waiting_actors.add(ll)
            continue

        tasks[ll] = []
        for o3 in [-1, 0, 1]:
            for o2 in [-1, 0, 1]:
                for o1 in [-1, 0, 1]:
                    if o3 == o2 == o1 == 0:
                        continue
                    offsets = (o3, o2, o1)
                    tasks[ll].extend(
                        ray.get(actor.update_ghost.remote(offsets)))

    tasks = {ll: task for ll, task in tasks.items() if task}
    while tasks:
        for ll, task in tasks.items():
            tasks[ll] = ray.get(actors[ll].wait_ghost.remote(task))
        tasks = {ll: task for ll, task in tasks.items() if task}


def print_actors(actors: dict[(int, int, int), ObjectRef]) -> None:
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
