# pylint: disable = import-error, too-many-arguments, undefined-variable, unused-argument, redefined-outer-name, too-few-public-methods, no-member, pointless-string-statement
"""Module containing MeshBlockActor class and related functions."""

from typing import List
from typing import Tuple
import ray
import meshblock as mb


@ray.remote
class MeshBlockActor:
    """Remotely launch actors as mesh blocks."""

    def __init__(self, node: mb.MeshBlockTree, coordinate_type: str,
                 nghost: int = 0) -> None:
        """Initialize MeshBlockActor with a mesh block and its corresponding tree."""
        self.tree_node = node
        self.nghost = nghost
        self.mblock = mb.MeshBlock(node.size, coordinate_type, nghost)
        self.mblock.allocate().fill_random()

    def get_data(self) -> Tuple[mb.MeshBlock, mb.RegionSize]:
        """Print the mesh block."""
        return self.mblock, self.tree_node.size


def launch_actors(node: mb.MeshBlockTree, actors: List[MeshBlockActor], nghost: int) -> None:
    """Launch actors based on the tree."""
    if not node.leaf:
        actors.append(MeshBlockActor.remote(
            node, "cartesian", nghost))
    else:
        for leaf in node.leaf:
            if leaf:
                launch_actors(leaf, actors, nghost)


def print_actors(actors: List[MeshBlockActor]) -> None:
    """Print the mesh block."""
    for actor in actors:
        mblock, size = ray.get(actor.get_data.remote())
        print("\n", size)
        mblock.print_data()


if __name__ == '__main__':
    # Initial split without refinement
    mb.MeshBlockTree.set_block_size(nx1=2, nx2=2, nx3=1)
    rs = mb.RegionSize(x1dim=(0, 120., 4), x2dim=(0, 120., 4))
    tree = mb.MeshBlockTree(rs)
    tree.create_tree()
    tree.print_tree()

    # Launch actors based on the tree
    ray.init(runtime_env={"py_modules": [mb]})
    actors = []
    launch_actors(tree, actors, nghost=1)
    print_actors(actors)

    # Refine the tree
    tree.leaf[3].split_block()
    tree.print_tree()
    ray.kill(actors[3])
    new_actors = []
    launch_actors(tree.leaf[3], new_actors, nghost=1)
    actors = actors[:3] + new_actors + actors[4:]
    print_actors(actors)

    # Shutdown Ray
    ray.shutdown()
