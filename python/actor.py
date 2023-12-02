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
        self.logicloc = (1,1,1)
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

    def get_level(self) -> int:
        """Get the level of the mesh block."""
        return self.level

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

    def update_neighbors(self, offsets: (int, int, int), root:me.Tree, 
                         actors: dict[(int,int,int), ObjectRef]) -> None:
        """Update the neighbors of the mesh block."""
        node = root.find_node(self.mblock.size.center())
        neighbors = node.find_neighbors(offsets, self.mblock.coord)
        neighbor_actors = [
            actors[(nb.lx3, nb.lx2, nb.lx1)] for nb in neighbors
        ]
        self.neighbors[offsets] = neighbors

@ray.remote
class MeshBlockActorTestOnly(MeshBlockActor):
    def get_data(self) -> (me.MeshBlock, me.RegionSize, int):
        """Print the mesh block."""
        node_id = ray.get_runtime_context().get_node_id()
        worker_id = ray.get_runtime_context().get_worker_id()
        return self.mblock, node_id, worker_id
