
import os
os.environ['RAY_SCHEDULER_SPREAD_THRESHOLD'] = '0.0001'
import sys
import ray
sys.path.append('../')
import mesh as me
import actor as ac
import manager as mg
import time

if __name__ == '__main__':
    ray.init(runtime_env={"py_modules": [me]})

    # Initial split without refinement
    size = me.RegionSize(x1dim=(0, 3200., 3200), x2dim=(0, 3200., 3200))
    me.BlockTree.set_block_size(nx1=800, nx2=800, nx3=1)
    root = me.BlockTree(size)
    root.create_tree()

    mesh = mg.MeshManager(root)
    #mg.print_actors(mesh.actors)

    start_time = time.time()
    for i in range(1):
        print(f"===== Step {i} =====")
        mesh.run_one_step(amr = True)

    end_time = time.time()

    #mg.print_actors(mesh.actors)
    print("after run one step...")
    print(f"Total time: {end_time - start_time}")
