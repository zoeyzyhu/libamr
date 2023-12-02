# pylint: disable = import-error, too-many-arguments, undefined-variable, unused-argument, redefined-outer-name, too-few-public-methods, no-member, pointless-string-statement
"""Module containing MeshBlockActor class and related functions."""

from math import floor, log2
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
        self.logicloc = (1, 1, 1)
        self.neighbors = {}

    def new(self, node: me.BlockTree, coordinate_type: str = "cartesian") -> None:
        """Initialize MeshBlockActor from its tree node."""
        self.mblock = me.MeshBlock(node.size, coordinate_type)
        self.mblock.allocate().fill_random()
        self.logicloc = node.lx3, node.lx2, node.lx1

    def relaunch(self, refs: [ObjectRef], root: me.BlockTree) -> None:
        """Restart MeshBlockActor from its tree node."""
        self.logicloc, data = ray.get(refs)
        node = root.find_node_by_logicloc(self.logicloc)
        self.mblock = me.MeshBlock(node.size)
        self.mblock.allocate(data.shape[-1])
        self.mblock.data[:] = data[:]
        self.neighbors = {}

    def put_data(self):
        """Put the mesh block in Plasma store."""
        refs = ray.put([self.logicloc, self.mblock.data])
        return refs

    def get_view(self, offsets: (int, int, int)) -> np.ndarray:
        """Get the view of the interior."""
        return self.mblock.view[offsets], self.logicloc

    def get_prolong(self, my_offsets: (int, int, int),
                    finer: me.Coordinates) -> np.ndarray:
        """Get the view of the mesh block with prolongation."""
        nb_offsets = tuple(-x for x in my_offsets)
        data = self.mblock.prolongated_view(my_offsets, finer)
        return data, self.logicloc, nb_offsets

    def get_restrict(self, my_offsets: (int, int, int),
                     coarser: me.Coordinates) -> np.ndarray:
        """Get the view of the mesh block with restriction."""
        nb_offsets = tuple(-x for x in my_offsets)
        data = self.mblock.restricted_view(my_offsets, coarser)
        return data, self.logicloc, nb_offsets

    def get_logicloc(self) -> [int]:
        """Get the logic location of the mesh block."""
        return self.logicloc

    def level_diff(self, logicloc: (int, int, int)) -> int:
        """Get the level difference between two logic locations."""
        my_level = floor(log2(self.logicloc[2]))
        other_level = floor(log2(logicloc[2]))
        return my_level - other_level

    def wait_ghost(self, tasks: [ObjectRef]) -> [ObjectRef]:
        """Wait for ghost-cell tasks to finish."""
        ready_tasks, remain_tasks = ray.wait(tasks)

        for task in ready_tasks:
            view, logicloc, offsets = ray.get(task)
            if self.level_diff(logicloc) < 0:  # neighbor at finer level
                self.mblock.part(offsets, logicloc)[:] = view
            else:
                self.mblock.ghost[offsets][:] = view

        return remain_tasks

    def update_ghost(self, offsets: (int, int, int)) -> [ObjectRef]:
        """Launch ghost-cell tasks."""
        nbs = self.neighbors[offsets]
        nb_offsets = tuple(-x for x in offsets)

        if len(nbs) > 1:  # neighbors at finer level
            tasks = [nb.get_restrict.remote(
                nb_offsets, self.mblock.coord) for nb in nbs]
        # neighbor at coarser level
        elif self.level_diff(ray.get(nbs[0].get_logicloc.remote())) > 0:
            tasks = [nbs[0].get_prolong.remote(nb_offsets, self.mblock.coord)]
        else:  # neighbor at same level
            tasks = [nbs[0].get_view.remote(nb_offsets)]

        return tasks

    def update_neighbor(self, offsets: (int, int, int), root: me.BlockTree,
                        actors: dict[(int, int, int), ObjectRef]) -> None:
        """Update the neighbors of the mesh block."""
        node = root.find_node(self.mblock.size.center())
        neighbors = node.find_neighbors(offsets, self.mblock.coord)

        self.neighbors[offsets] = [
            actors[(nb.lx3, nb.lx2, nb.lx1)] for nb in neighbors
        ]

    def get_data(self) -> (me.MeshBlock, me.RegionSize, int):
        """Print the mesh block."""
        node_id = ray.get_runtime_context().get_node_id()
        worker_id = ray.get_runtime_context().get_worker_id()
        return self.mblock, node_id, worker_id
