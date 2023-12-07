
import os
os.environ['RAY_SCHEDULER_SPREAD_THRESHOLD'] = '0.0001'
import sys
import ray
sys.path.append('../')
import mesh as me
import actor as ac
import manager as mg

if __name__ == '__main__':
    ray.init(runtime_env={"py_modules": [me]})

    # Initial split without refinement
    size = me.RegionSize(x1dim=(0, 8., 8), x2dim=(0, 4., 4))
    me.BlockTree.set_block_size(nx1=2, nx2=2, nx3=1)
    root = me.BlockTree(size)
    root.create_tree()

    mesh = mg.MeshManager(root)
    #mg.print_actors(mesh.actors)

    mesh.run_one_step()

    #mg.print_actors(mesh.actors)
    print("after run one step...")
