# pylint: disable = import-error, too-many-arguments, undefined-variable, unused-argument, redefined-outer-name, too-few-public-methods, no-member, pointless-string-statement
"""Module containing MeshBlockActor class and related functions."""

import ray
from ray import ObjectRef
import numpy as np
import mesh as me

@ray.remote
class MeshBlockActor:
    """Remotely launch actors as mesh blocks."""

    def __init__(self) -> None:
        """Initialize MeshBlockActor."""
        self.mblock = None
        self.level = 0
        self.logicloc = (0,0,0)
        self.neighbors = {}

    def new(self, node: me.Tree, coordinate_type: str = "cartesian") -> None:
        """Initialize MeshBlockActor from its tree node."""
        self.mblock = me.MeshBlock(node.size, coordinate_type)
        self.mblock.allocate().fill_random()
        self.level = node.level
        self.logicloc = node.lx3, node.lx2, node.lx1

    def relaunch(self, data_ref: [ObjectRef]) -> None:
        """Restart MeshBlockActor from its tree node."""
        self.mblock = ray.get(data_ref[0])

    def put_data(self):
        """Put the mesh block in Plasma store."""
        ref = ray.put(self.mblock)
        return ref

    def get_view(self, offsets: (int, int, int)) -> np.ndarray:
        """Get the view of the interior."""
        return self.mblock.view[offsets], None, None

    def get_view_prolong(self, my_offsets: (int, int, int),
                         finer: me.Coordinates) -> np.ndarray:
        """Get the view of the mesh block with prolongation."""
        return self.mblock.prolongated_view(my_offsets, finer), self.level, self.logicloc

    def get_view_restrict(self, my_offsets: (int, int, int),
                          coarser: me.Coordinates) -> np.ndarray:
        """Get the view of the mesh block with restriction."""
        return self.mblock.restricted_view(my_offsets, coarser), self.level, self.logicloc

    def update_ghost(self, offsets: (int, int, int)) -> [ObjectRef]:
        """Update ghost cells."""
        neighbors = self.neighbors[offsets]
        nb_offsets = tuple(-x for x in offsets)

        if len(neighbors) > 1:  # neighbors at finer level
            tasks = [nb.get_view_restrict.remote(nb_offsets, self.mblock.coord)
                for nb in neighbors]
        elif ray.get(neighbors[0].get_level.remote()) < self.level:  # neighbors at coarser level
            tasks = [neighbors[0].get_view_prolong.remote(nb_offsets, self.mblock.coord)]
        else:  # neighbors at same level
            tasks = [neighbors[0].get_view.remote(nb_offsets)]

        #ready_tasks, remain_tasks = ray.wait(tasks)

        while len(tasks) > 0:
            ready_tasks, remain_tasks = ray.wait(tasks)
            tasks = remain_tasks
            for task in ready_tasks:
                view, level, logicloc = ray.get(task)
                if level is None:
                    self.mblock.ghost[offsets][:] = view
                elif level and level <= self.level:
                    self.mblock.ghost[offsets][:] = view
                elif level and level > self.level:
                    self.mblock.part(offsets, logicloc)[:] = view

        return

    def get_data(self) -> (me.MeshBlock, me.RegionSize, int):
        """Print the mesh block."""
        node_id = ray.get_runtime_context().get_node_id()
        worker_id = ray.get_runtime_context().get_worker_id()
        return self.mblock, node_id, worker_id

    def get_neighbors(self) -> dict[(int,int,int), ObjectRef]:
        """Get the neighbors of the mesh block."""
        return self.neighbors

    def get_center(self) -> (float, float, float):
        """Get the center of the mesh block."""
        return self.mblock.size.center(), self.mblock.coord

    def set_neighbor(self, offsets, neighbors: [ObjectRef]) -> None:
        """Set the neighbors of the mesh block."""
        self.neighbors[offsets] = neighbors
    
    def get_level(self) -> int:
        """Get the level of the mesh block."""
        return self.level

def update_neighbors(actors: dict[(int,int,int), ObjectRef],
                     root: me.Tree) -> None:
    for _, actor in actors.items():
        for o3 in [-1, 0, 1]:
            for o2 in [-1, 0, 1]:
                for o1 in [-1, 0, 1]:
                    offsets = (o3, o2, o1)

                    center, coord = ray.get(actor.get_center.remote())
                    node = root.find_node(center)
                    neighbors = node.get_neighbors(offsets, coord)

                    neighbor_actors = [
                        actors[(nb.lx3, nb.lx2, nb.lx1)] for nb in neighbors
                    ]
                    
                    actor.set_neighbor.remote(offsets, neighbor_actors)

def launch_actors(root: me.Tree) -> dict[(int,int,int), MeshBlockActor]:
    """Launch actors based on the tree."""
    actors = {}
    if not root.children:
        actor = MeshBlockActor.remote()
        actor.new.remote(root)
        actors[(root.lx3,root.lx2,root.lx1)]= actor
    else:
        for child in root.children:
            if child:
                actors.update(launch_actors(child))
    return actors

def print_actors(actors: dict[(int,int,int), MeshBlockActor]) -> None:
    """Print the mesh block."""
    for ll in actors:
        mblock, node_id, worker_id = ray.get(actors[ll].get_data.remote())
        print(f"\nNode:{node_id}\nWorker:{worker_id}\nlogicloc:{ll}")
        print(f"size = {mblock.size}")
        mblock.print_data()

def print_actor(actor: ObjectRef) -> None:
    """Print the mesh block."""
    mblock, node_id, worker_id = ray.get(actor.get_data.remote())
    print(f"\nNode:{node_id}\nWorker:{worker_id}")
    print(f"size = {mblock.size}")
    mblock.print_data()


if __name__ == '__main__':
    # Initial split without refinement
    size = me.RegionSize(x1dim=(0, 120., 8), x2dim=(0, 120., 4))
    me.Tree.set_block_size(nx1=4, nx2=2, nx3=1)
    tree = me.Tree(size)
    tree.create_tree()
    tree.print_tree()

    # Launch actors based on the tree
    ray.init(runtime_env={"py_modules": [me]})
    actors = launch_actors(tree)
    print_actors(actors)

    # Refine the tree
    node_to_refine = tree.children[3]
    node_to_refine.split_block()
    tree.print_tree()

    ray.kill(actors[3])
    new_actors = launch_actors(node_to_refine)
    actors = actors[:3] + new_actors + actors[4:]
    print_actors(actors)
    print(actors)

    # Shutdown Ray
    ray.shutdown()

    """
    # Put data and relaunch actors
    data_refs = []
    for actor in actors:
        data_refs.append(ray.get(actor.put_data.remote()))
        # ray.kill(actor)

    new_actors = []
    for data_ref in data_refs:
        actor = MeshBlockActor.remote()
        actor.relaunch.remote(data_ref)
        new_actors.append(actor)

    print(new_actors)
    print_actors(new_actors)


    # Refine the tree
    node_to_refine = tree.children[3]
    node_to_refine.split_block()
    tree.print_tree()

    ray.kill(actors[3])
    new_actors = []
    launch_actors(node_to_refine, new_actors, nghost=1)
    actors = actors[:3] + new_actors + actors[4:]
    print_actors(actors)
    """
