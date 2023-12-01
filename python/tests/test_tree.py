
import sys
sys.path.append('../')
import mesh as me

def test_locate_neighbors_down(root, node, offsets):
    print("\n===== Test locate neighbors down =====")
    print(f"node = {node}")
    intervals = node.calculate_intervals(offsets)
    print(f"\nintervals = {intervals}")
    node_found = root.locate_neighbors_down(*intervals)
    print(f"\nnode_found = {node_found}")

def test_find_neighbors(node, offsets):
    print("\n===== Test find neighbors =====")
    print(f"node = {node}")
    neighbors = node.find_neighbors(offsets)
    for neighbor in neighbors:
        print(neighbor)
    return neighbors

if __name__ == '__main__':
    size = me.RegionSize(x1dim=(0, 120., 8), x2dim=(0, 120., 4))
    me.Tree.set_block_size(nx1=2, nx2=2, nx3=1)
    root = me.Tree(size)
    root.create_tree()

    root.children[0].children[1].split_block()
    node_finer = root.children[0].children[1].children[0]
    offsets = (0, 0, 1)
    test_locate_neighbors_down(root, node_finer, offsets)
    neighbors = test_find_neighbors(node_finer, offsets)

    node_coarser = root.children[0]
    neighbors = test_find_neighbors(node_coarser, offsets)
    



