
import sys
import ray
sys.path.append('../')
import mesh as me
import actor as ac

def test_launch_actors(tree):
    print("\n===== Test launch actors =====")
    actors = ac.launch_actors(tree)
    ac.print_actors(actors)
    print(actors)
    return actors

def test_refine_actors(tree, actors):
    print("\n===== Test refine actors =====")
    node_to_refine = tree.children[0].children[1]
    node_to_refine.split_block()
    #tree.print_tree()

    ray.kill(actors[1])
    new_actors = ac.launch_actors(node_to_refine)
    actors = actors[:1] + new_actors + actors[2:]
    ac.print_actors(actors)
    return actors

def test_put_data(actors):
    print("\n===== Test put data =====")
    data_refs = []
    for actor in actors:
        data_refs.append(ray.get(actor.put_data.remote()))

    new_actors = []
    for data_ref in data_refs:
        actor = ac.MeshBlockActor.remote()
        actor.relaunch.remote([data_ref])
        new_actors.append(actor)

    ac.print_actors(new_actors)
    return

def test_update_neighbors(actors, root):
    print("\n===== Test find neighbors =====")
    ac.update_neighbors(actors, root)
    for ll, actor in actors.items():
        print("----- Actor {} -----".format(ll))
        print(ray.get(actor.get_neighbors.remote()))

def test_update_ghost(actor, offsets):
    print("\n===== Test update ghost =====")
    print("----- Before update ghost -----")
    ac.print_actors([actor])
    actor.update_ghost.remote(offsets)
    print("----- Updated ghost -----")
    ac.print_actors([actor])
    return


if __name__ == '__main__':
    # Initial split without refinement
    size = me.RegionSize(x1dim=(0, 120., 8), x2dim=(0, 120., 4))
    me.Tree.set_block_size(nx1=2, nx2=2, nx3=1)
    tree = me.Tree(size)
    tree.create_tree()
    #tree.print_tree()

    # Launch actors based on the tree
    ray.init(runtime_env={"py_modules": [me]})
    actors = test_launch_actors(tree)
    test_update_neighbors(actors, tree)
    #actors = test_refine_actors(tree, actors)
    #tree.print_tree()
    #test_put_data(actors)
    #test_locate_neighbors(actors[1], tree, (0, 1, 1))
    #test_locate_neighbors(actors[0], tree, (0, 0, 1))
    #test_update_ghost(actors[1], (0, 1, 1))


    ray.shutdown()
