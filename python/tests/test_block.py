
import sys
sys.path.append('../')
import mesh as me

def test_restriction(offsets, coarser, finer):
    print("\n===== Test restriction =====")
    print(f"offsets = {offsets}")
    print(f"\ncoarser block:")
    mb_coarser.print_data()
    print(f"\nfiner block:")
    mb_finer.print_data()
    view = finer.restricted_view(offsets, mb_coarser.coord)
    print(f"\nview = {view}")
    return

if __name__ == '__main__':
    size = me.RegionSize(x1dim=(0, 120., 16), x2dim=(0, 120., 8))
    me.Tree.set_block_size(nx1=4, nx2=4, nx3=1)
    root = me.Tree(size)
    root.create_tree()

    root.children[0].children[1].split_block()
    node_finer = root.children[0].children[1].children[0]
    node_coarser = root.children[0].children[0]
    mb_finer = me.MeshBlock(node_finer.size)
    mb_finer.allocate().fill_random(0)
    mb_coarser = me.MeshBlock(node_coarser.size)
    mb_coarser.allocate().fill_random(1)
    offsets = (0, 0, -1)
    #test_restriction(offsets, mb_coarser, mb_finer)


    # replicate zige test
    me.Tree.set_block_size(nx1=4, nx2=4, nx3=1)
    size = me.RegionSize(x1dim=(0, 120., 8), x2dim=(0, 120., 8))
    root = me.Tree(size)
    root.create_tree()
    root.children[3].split_block()
    #print("----------- print tree -----------")
    #root.print_tree()
    print("----------- print node info -----------")
    coarse_node = root.children[2]
    print(f"coarse_node = {coarse_node}")
    coarse_block = me.MeshBlock(coarse_node.size)
    coarse_block.allocate().fill_random(0)
    coarse_block.print_data()
    fine_node = root.children[3].children[0]
    print(f"fine_node = {fine_node}")
    fine_block = me.MeshBlock(fine_node.size)
    fine_block.allocate().fill_random(1)
    fine_block.print_data()

    """ view = coarse_block.prolongated_view((0, 0, 1), fine_block.coord)
    print(view.shape)
    print(view) """

    # test 5 (restriction)
    print("==== Test Restriction ====")
    view = fine_block.restricted_view((0, 0, -1), coarse_block.coord)
    print(view)