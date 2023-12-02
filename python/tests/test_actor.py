
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
    root, actors = mg.refine_actor(point, root, actors)
    print("\n===== After refine actors: Tree =====")
    root.print_tree()
    print("\n===== After refine actors: Actors =====")
    mg.print_actors(actors)
    return root, actors

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
    size = me.RegionSize(x1dim=(0, 120., 8), x2dim=(0, 120., 4))
    me.BlockTree.set_block_size(nx1=2, nx2=2, nx3=1)
    root = me.BlockTree(size)
    root.create_tree()
    #root.print_tree()

    # Launch actors based on the tree
    ray.init(runtime_env={"py_modules": [me]})
    actors = mg.launch_actors(root)

    # Refine an actor based on a point
    point_to_refine = (0, 29, 44)
    mg.refine_actor(point_to_refine, root, actors)

    # Update ghost cells of a designated zone
    offsets = (0, -1, -1)
    actor = actors[(0b100, 0b110, 0b110)]
    test_update_ghost(actor, offsets)
    #test_update_ghosts_all(actors)
