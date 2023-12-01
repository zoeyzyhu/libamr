# pylint: disable = import-error, too-few-public-methods, unused-argument, unused-variable, unused-import, redefined-outer-name, undefined-variable
"""Mesh class and related functions."""

import ray
import mesh as me
import actor as ac
from typing import Tuple
from ray import ObjectRef


def launch_actors(root: me.Tree) -> dict[(int,int,int), ac.MeshBlockActor]:
    """Launch actors based on the tree."""
    actors = {}
    if not root.children:
        actor = ac.MeshBlockActor.remote()
        actor.new.remote(root)
        actors[(root.lx3,root.lx2,root.lx1)]= actor
    else:
        for child in root.children:
            if child:
                actors.update(launch_actors(child))
    return actors


def refine_actor(point: Tuple[int, int, int], root: me.Tree,
                 actors: dict[(int,int,int), ac.MeshBlockActor]) -> None:
    """Refine the block where the specified point locates."""
    node = root.find_node(point)
    node.split_block()

    logicloc = node.lx3, node.lx2, node.lx1
    ray.kill(actors[logicloc])
    actors.pop(logicloc)

    new_actors = launch_actors(node)
    actors.update(new_actors)

    update_neighbors_all(actors, root)
    return root, actors


def update_neighbors_all(actors: dict[(int,int,int), ObjectRef],
                     root: me.Tree) -> None:
    for _, actor in actors.items():
        for o3 in [-1, 0, 1]:
            for o2 in [-1, 0, 1]:
                for o1 in [-1, 0, 1]:
                    offsets = (o3, o2, o1)
                    actor.update_neighbors.remote(offsets, root, actors)


def update_ghost_all(actors: dict[(int,int,int), ac.MeshBlockActor]) -> None:
    """Update ghost cells for all actors."""
    for _, actor in actors.items():
        for o3 in [-1, 0, 1]:
            for o2 in [-1, 0, 1]:
                for o1 in [-1, 0, 1]:
                    offsets = (o3, o2, o1)
                    actor.update_ghost.remote(offsets)


def print_actors(actors: dict[(int,int,int), ac.MeshBlockActor]) -> None:
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
