# pylint: disable = wrong-import-position

"""Test module packing as environment on Ray."""

import sys
import ray
sys.path.append('../')
import meshblock as mb

ray.init(runtime_env={"py_modules": [mb]})


@ray.remote
def test_my_module():
    """Accessing module functions on remote nodes."""
    mb.MeshBlockTree.set_block_size(nx1=2, nx2=2, nx3=1)
    rs = mb.RegionSize(x1dim=(0, 120., 4), x2dim=(0, 120., 4))
    root = mb.MeshBlockTree(rs)
    root.create_tree()
    root.print_tree()

    block = mb.MeshBlock(rs, "cartesian", nghost=1)
    block.allocate().fill_random()
    block.print_data()


ray.get(test_my_module.remote())
