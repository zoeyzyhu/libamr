# pylint: disable = import-error, too-many-arguments, undefined-variable, unused-argument, redefined-outer-name, too-few-public-methods, no-member, pointless-string-statement
"""Module containing MeshBlockActor class and related functions."""

from typing import List
import ray
import meshblock as mb

ray.init(runtime_env={"py_modules": [mb]})


@ray.remote
class MeshBlockActor:
    """Remotely launch actors as mesh blocks."""

    def __init__(self, size: mb.RegionSize, coordinate_type: str,
                 tree: mb.MeshBlockTree, nghost: int = 0):
        """Initialize MeshBlockActor with a mesh block and its corresponding tree."""
        self.size = size
        self.tree = tree
        self.nghost = nghost
        self.mblock = mb.MeshBlock(size, coordinate_type, nghost)
        self.mblock.allocate().fill_random()

    def get_data(self):
        """Print the mesh block."""
        return self.mblock, self.size


def launch_actors(tree: mb.MeshBlockTree, actors: List[MeshBlockActor], nghost: int) -> None:
    """Launch actors based on the tree."""
    if not tree.leaf:
        actors.append(MeshBlockActor.remote(
            tree.size, "cartesian", tree, nghost))
    else:
        for leaf in tree.leaf:
            if leaf:
                launch_actors(leaf, nghost, actors)


def print_actors(actors: List[MeshBlockActor]) -> None:
    """Print the mesh block."""
    for actor in actors:
        mblock, size = ray.get(actor.get_data.remote())
        print("\n", size)
        mblock.print_data()


if __name__ == '__main__':
    # Initial split without refinement
    mb.MeshBlockTree.set_block_size(nx1=2, nx2=2, nx3=1)
    rs = mb.RegionSize(x1dim=(0, 120., 8), x2dim=(0, 120., 4))
    tree = mb.MeshBlockTree(rs)
    tree.create_tree()
    # tree.print_tree()

    # Launch actors based on the tree
    actors = []
    launch_actors(tree, actors, nghost=1)
    print_actors(actors)
