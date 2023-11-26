from mesh_block_tree import MeshBlockTree
from coordinate_factory import CoordinateFactory
from typing_extensions import Self
import ray

@ray.remote
class MeshActor:
    def __init__(self, tree: MeshBlockTree, coordinate_type: str, nghost: int):
        mblocks = []
        pass

if __name__ == "__main__":
    mesh = Mesh(tree, "cartesian", 1)
    pass
