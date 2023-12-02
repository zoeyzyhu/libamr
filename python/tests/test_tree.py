
import sys
sys.path.append('../')
import mesh as me

def test_find_neighbors(node, offsets, coord):
    print("\n===== Test find neighbors =====")
    print(f"node = {node}")
    for nb in node.find_neighbors(offsets, coord):
        print(nb)

if __name__ == '__main__':
    me.BlockTree.set_block_size(nx1=2, nx2=2, nx3=1)
    rs = me.RegionSize(x1dim=(0, 120., 8), x2dim=(0, 120., 4))
    root = me.BlockTree(rs)
    root.create_tree()

    print("\n\n===== split block =====")
    root.node(3,1).split()
    print(root.node(1,0))

    n1 = root.node(3,1,0)

    print("\n\n===== split block chain =====")

    n1.split()

    n1.children[0].split()

    n1.node(0,0).split()

    root.print_tree()
    node = root.node(3,1).root()
    print(node)

    size = me.RegionSize(x1dim=(0, 200., 10), x2dim=(0, 120., 6))
    me.BlockTree.set_block_size(nx1=2, nx2=2, nx3=1)
    root = me.BlockTree(size)
    root.create_tree()

    root.print_tree()

    root.node(0,1,0).split()
    node_finer = root.node(0,1,0,0)
    mb = me.MeshBlock(node_finer.size)
    offsets = (0, 0, 1)
    test_find_neighbors(node_finer, offsets, mb.coord)

    node_coarser = root.node(0,0,0)
    mb = me.MeshBlock(node_coarser.size)
    test_find_neighbors(node_coarser, offsets, mb.coord)
