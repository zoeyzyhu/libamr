# pylint: disable = wrong-import-position

"""Test module packing as environment on Ray."""


import sys
import ray
sys.path.append('../')
import mesh as me

ray.init(runtime_env={"py_modules": [me]})


@ray.remote
def test_my_module():
    """Accessing module functions on remote nodes."""
    me.Tree.set_block_size(nx1=2, nx2=2, nx3=1)
    size = me.RegionSize(x1dim=(0, 120., 4), x2dim=(0, 120., 4))
    root = me.Tree(size)
    root.create_tree()
    root.print_tree()

    block = me.MeshBlock(size)
    block.allocate().fill_random()
    block.print_data()


ray.get(test_my_module.remote())
