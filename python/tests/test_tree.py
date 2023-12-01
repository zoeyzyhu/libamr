
import sys
sys.path.append('../')
import mesh as me

def test_find_neighbors(node, offsets, coord):
    print("\n===== Test find neighbors =====")
    print(f"node = {node}")
    for nb in node.neighbors(offsets, coord):
        print(nb)

if __name__ == '__main__':
    size = me.RegionSize(x1dim=(0, 120., 8), x2dim=(0, 120., 4))
    me.Tree.set_block_size(nx1=2, nx2=2, nx3=1)
    root = me.Tree(size)
    root.create_tree()

    root.children[0].children[1].split_block()
    node_finer = root.children[0].children[1].children[0]
    mb = me.MeshBlock(node_finer.size)
    offsets = (0, 0, 1)
    test_find_neighbors(node_finer, offsets, mb.coord)

    node_coarser = root.children[0].children[0]
    mb = me.MeshBlock(node_coarser.size)
    test_find_neighbors(node_coarser, offsets, mb.coord)
