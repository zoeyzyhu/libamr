
import os
os.environ['RAY_SCHEDULER_SPREAD_THRESHOLD'] = '0.0001'
import sys
import ray
sys.path.append('../')
import mesh as me
import actor as ac
import manager as mg


def test_launch_actors(tree):
    print("\n===== Test launch actors =====")
    actors = mg.launch_actors(tree)
    mg.print_actors(actors)
    print(actors)
    return actors

def test_refine_actors(point, root, actors):
    print("\n===== Test refine actors =====")
    #root, actors = mg.refine_actor(point, root, actors)
    mg.refine_actor(point, root, actors)

    print("\n===== After refine actors: Tree =====")
    root.print_tree()
    print("\n===== After refine actors: Actors =====")
    mg.print_actors(actors)
    #return root, actors

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

    mg.print_actors(new_actors)
    return

def test_update_neighbors(actors, root):
    print("\n===== Test find neighbors =====")
    mg.update_neighbors_all(actors, root)

def test_update_ghost(actor, offsets):
    print("\n===== Test update ghost =====")
    print("----- Before update ghost -----")
    mg.print_actor(actor)
    print(f"offsets = {offsets}")
    tasks = actor.update_ghost.remote(offsets)
    actor.wait_ghost.remote(tasks)
    print("----- Updated ghost -----")
    mg.print_actor(actor)
    return

def test_update_ghosts_all(actors):
    print("\n===== Test update ghost all =====")
    mg.update_ghosts_all(actors)
    mg.print_actors(actors)
    return

if __name__ == '__main__':
    # Initial split without refinement
    size = me.RegionSize(x1dim=(0, 240., 8), x2dim=(0, 120., 4))
    me.BlockTree.set_block_size(nx1=2, nx2=2, nx3=1)
    root = me.BlockTree(size)
    root.create_tree()

    # Launch actors based on the tree
    ray.init(runtime_env={"py_modules": [me]})
    actors = mg.launch_actors(root)
    mg.update_neighbors_all(actors, root)
    mg.update_ghosts_all(actors)

   
    # Refine an actor based on a point
    point_to_refine1 = (0, 29, 44)
    node = root.find_node(point_to_refine1)
    mg.print_actor_coord(point_to_refine1, root, actors)
    mg.refine_actor(point_to_refine1, root, actors)
    mg.print_actor_children(node, actors)

    # Refine a refined block
    point_to_refine2 = (0, 40, 44)
    node = root.find_node(point_to_refine2)
    mg.print_actor_coord(point_to_refine2, root, actors)
    mg.refine_actor(point_to_refine2, root, actors)
    mg.print_actor_children(node, actors)
    #root.print_tree()

    # merge the finest blocks
    #point_to_merge = (0, 40, 44)
    #mg.merge_actor(point_to_merge, root, actors)
    #mg.print_actor_coord(point_to_merge, root, actors)

    
    mg.orchestrate_actor(actors, root)
    #print("\n===== After orchestrate actors: Tree =====")
    #root.print_tree()
    #mg.print_actors(actors)
