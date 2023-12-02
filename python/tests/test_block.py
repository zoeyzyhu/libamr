
import sys
sys.path.append('../')
import mesh as me

def test_restriction(offsets, coarser, finer):
    print("\n===== Test restriction =====")
    print(f"offsets = {offsets}")
    print(f"\ncoarser block:")
    coarser.print_data()
    print(f"\nfiner block:")
    finer.print_data()
    view = finer.restricted_view(offsets, coarser.coord)
    print(f"\nview = {view}")
    return

def test_prolongation_restriction():
    me.BlockTree.set_block_size(nx1=2, nx2=2, nx3=1)
    size = me.RegionSize(x1dim=(0, 120., 8), x2dim=(0, 120., 8))
    root = me.BlockTree(size)
    root.create_tree()
    root.print_tree()
    coarse_node = root.children[3].children[0]
    print(f"coarse_node = {coarse_node}")
    coarse_block = me.MeshBlock(coarse_node.size)
    coarse_block.allocate().fill_random()
    coarse_block.print_data()
    print("\n===== split block =====")
    root.children[3].children[1].split()
    fine_node = root.children[3].children[1].children[0]
    print(f"fine_node = {fine_node}")
    fine_block = me.MeshBlock(fine_node.size)
    fine_block.allocate().fill_random()
    fine_block.print_data()

    print("\n===== Test Prolongation =====")
    view = coarse_block.prolongated_view((0, 0, 1), fine_block.coord)
    print(view.shape)
    print(view)

    print("\n===== Test Restriction =====")
    view = fine_block.restricted_view((0, 0, -1), coarse_block.coord)
    print(view.shape)
    print(view)

def test_part_function():
    """Test part function in meshblock."""
    me.BlockTree.set_block_size(nx1=4, nx2=4, nx3=1)
    rs = me.RegionSize(x1dim=(0, 120., 8), x2dim=(0, 120., 8), nghost=2)
    root = me.BlockTree(rs)
    root.create_tree()
    root.print_tree()

    coarse_node = root.node(0)
    print(f"\ncoarse_node = {coarse_node}")
    coarse_block = me.MeshBlock(coarse_node.size)
    coarse_block.allocate().fill_random()
    coarse_block.print_data()
    print("\n===== split block =====")
    root.node(1).split()
    fine_node1 = root.node(1,0)
    fine_node2 = root.node(1,2)
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

def test_cartesian():
    print("\n===== Test cartesian =====")
    size = me.RegionSize(x1dim=(0, 1, 2), x2dim=(0, 1, 4), nghost=2)
    mb = me.MeshBlock(size)
    mb.allocate()
    print(mb)

def test_cylindrical():
    print("\n===== Test cylindrical =====")
    size = me.RegionSize(x1dim=(0, 1, 2), x2dim=(0, 1, 4), nghost=2)
    mb = me.MeshBlock(size, "cylindrical")
    mb.allocate(2)
    print(mb)

def test_view():
    print("\n===== Test view =====")
    size1 = me.RegionSize(x1dim=(0, 40., 4), x2dim=(0, 40., 2))
    mb1 = me.MeshBlock(size1)
    mb1.allocate(2).fill_random()
    mb1.print_data()
    print(mb1.view[(0, 0, 1)][0, :, :, 0])
    print(mb1.view[(0, 0, 1)][0, :, :, 1])
    print(mb1.view[(0, -1, -1)][0, :, :, 0])
    print(mb1.view[(0, -1, -1)][0, :, :, 1])

def test_view2():
    print("\n===== Test view2 =====")
    size2 = me.RegionSize(x1dim=(40., 120., 4), x2dim=(0, 40., 2))
    mb2 = me.MeshBlock(size2)
    mb2.allocate(2).fill_random()
    print(mb2.view[(0, 1, 0)][0, :, :, 0])
    print(mb2.view[(0, 1, 0)][0, :, :, 1])
    print(mb2.view[(0, -1, 1)][0, :, :, 0])
    print(mb2.view[(0, -1, 1)][0, :, :, 1])

def test_view3():
    print("\n===== Test view3 =====")
    size3 = me.RegionSize(x1dim=(0., 40., 4), x2dim=(40, 120., 2))
    mb3 = me.MeshBlock(size3)
    mb3.allocate(2).fill_random()
    mb3.print_data()
    print(mb3.view[(0, 1, -1)][0, :, :, 0])
    print(mb3.view[(0, 1, -1)][0, :, :, 1])

def test_view4():
    print("\n===== Test view4 =====")
    size4 = me.RegionSize(x1dim=(40., 120., 4), x2dim=(40, 120., 2))
    mb4 = me.MeshBlock(size4)
    mb4.allocate(2).fill_random()
    mb4.print_data()
    print(mb4.view[(0, 1, 1)][0, :, :, 0])
    print(mb4.view[(0, 1, 1)][0, :, :, 1])

if __name__ == '__main__':

    test_prolongation_restriction()