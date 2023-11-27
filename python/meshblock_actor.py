# pylint: disable = import-error, too-many-arguments, undefined-variable, unused-argument, redefined-outer-name, too-few-public-methods, no-member, pointless-string-statement
"""Module containing MeshBlockActor class and related functions."""

from typing import List
from typing import Tuple
import ray
from ray import ObjectRef
import meshblock as mb


@ray.remote
class MeshBlockActor:
    """Remotely launch actors as mesh blocks."""

    def __init__(self, node: mb.MeshBlockTree,
                 coordinate_type: str, nghost: int = 1,
                 refs: List[ObjectRef] = None) -> None:
        """Initialize MeshBlockActor its tree node."""
        self.nghost = nghost
        self.id = ray.worker.global_worker.worker_id.hex()
        self.node_id = ray.get_runtime_context().get_node_id()

        if refs is not None:
            self.tree_node = ray.get(refs[0])
            self.mblock = ray.get(refs[1])
        else:
            self.tree_node = node
            self.mblock = mb.MeshBlock(node.size, coordinate_type, nghost)
            self.mblock.allocate().fill_random()

    def get_data(self) -> Tuple[mb.MeshBlock, mb.RegionSize, int]:
        """Print the mesh block."""
        return self.mblock, self.tree_node.size, self.id, self.node_id

    def put_data(self):
        """Put the mesh block in Plasma store."""
        node_ref = ray.put(self.tree_node)
        mblock_ref = ray.put(self.mblock)
        return node_ref, mblock_ref


def launch_actors(node: mb.MeshBlockTree, actors: List[MeshBlockActor],
                  nghost: int = 1) -> None:
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
        mblock, size, worker_id, node = ray.get(actor.get_data.remote())
        print(f"\nNode:{node}\nWorker:{worker_id}\nSize: {size}\n")
        mblock.print_data()


if __name__ == '__main__':
    # Initial split without refinement
    mb.MeshBlockTree.set_block_size(nx1=2, nx2=2, nx3=1)
    rs = mb.RegionSize(x1dim=(0, 120., 8), x2dim=(0, 120., 4))
    tree = mb.MeshBlockTree(rs)
    tree.create_tree()
    # tree.print_tree()

    # Launch actors based on the tree
    ray.init(runtime_env={"py_modules": [mb]})
    # print(ray.nodes())
    # print(ray.available_resources())
    actors = []
    launch_actors(tree, actors, nghost=1)
    print_actors(actors)

    # Put data and relaunch actors
    data_ref = []
    for actor in actors:
        node, mblock = ray.get(actor.put_data.remote())
        data_ref.append((node, mblock))
        # ray.kill(actor)

    new_actors = []
    for node, mblock in data_ref:
        new_actors.append(MeshBlockActor.remote(
            node=None,
            coordinate_type="cartesian",
            nghost=1,
            refs=[node, mblock]))
    print(new_actors)
    print_actors(new_actors)

    """
    # Refine the tree
    node_to_refine = tree.leaf[3]
    node_to_refine.split_block()
    tree.print_tree()

    ray.kill(actors[3])
    new_actors = []
    launch_actors(node_to_refine, new_actors, nghost=1)
    actors = actors[:3] + new_actors + actors[4:]
    print_actors(actors)
    """
    # Shutdown Ray
    ray.shutdown()
