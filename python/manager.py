# pylint: disable = import-error, too-few-public-methods, unused-argument, unused-variable, unused-import, redefined-outer-name, undefined-variable
"""Mesh class and related functions."""

import ray
import mesh as me
import actor as ac
from typing import List

def launch_actors(node: me.Tree) -> List[ac.MeshBlockActor]:
    """Launch actors based on the tree."""
    actors = []
    if not node.children:
        actor = ac.MeshBlockActor.remote()
        actor.new.remote(node.size)
        actors.append(actor)
    else:
        for child in node.children:
            if child:
                actors.extend(launch_actors(child))
    return actors

def print_actors(actors: List[ac.MeshBlockActor]) -> None:
    """Print the mesh block."""
    for actor in actors:
        size, mblock, node_id, worker_id = ray.get(actor.get_data.remote())
        print(f"\nNode:{node_id}\nWorker:{worker_id}\n{size}")
        mblock.print_data()

if __name__ == "__main__":
    size = me.RegionSize(x1dim=(0, 120., 8), x2dim=(0, 120., 4))
    me.Tree.set_block_size(nx1=2, nx2=2, nx3=1)
    tree = me.Tree(size)
    tree.create_tree()

    # Launch actors based on the tree
    ray.init(runtime_env={"py_modules": [me]})
    print("\n===== Launch actors =====")
    actors = launch_actors(tree)
    print_actors(actors)
