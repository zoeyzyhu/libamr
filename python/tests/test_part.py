# pylint: disable = cyclic-import, wrong-import-position
"""Test part function in meshblock."""

import sys
sys.path.append('../')
import mesh as me


def test_part_function():
    """Test part function in meshblock."""
    me.Tree.set_block_size(nx1=4, nx2=4, nx3=1)
    rs = me.RegionSize(x1dim=(0, 120., 8), x2dim=(0, 120., 8), nghost=2)
    root = me.Tree(rs)
    root.create_tree()
    root.print_tree()

    coarse_node = root.children[0]
    print(f"\ncoarse_node = {coarse_node}")
    coarse_block = me.MeshBlock(coarse_node.size)
    coarse_block.allocate().fill_random()
    coarse_block.print_data()
    print("\n===== split block =====")
    root.children[1].split_block()
    fine_node1 = root.children[1].children[0]
    fine_node2 = root.children[1].children[2]
    print(f"\nfine_node1 = {fine_node1}")
    fine_block1 = me.MeshBlock(fine_node1.size)
    fine_block1.allocate().fill_random()
    fine_block1.print_data()
    print(f"\nfine_node2 = {fine_node2}")
    fine_block2 = me.MeshBlock(fine_node2.size)
    fine_block2.allocate().fill_random()
    fine_block2.print_data()

    # Test 1
    print("\n===== Test Part (0, 0, 1) =====")
    part_view1 = coarse_block.part((0, 0, 1), (0, 0, 0))
    print(part_view1[:, :, :, 0])
    part_view2 = coarse_block.part((0, 0, 1), (0, 1, 0))
    print(part_view2[:, :, :, 0])


# Run the test
test_part_function()
