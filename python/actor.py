# pylint: disable = import-error, too-many-arguments, undefined-variable, unused-argument, redefined-outer-name, too-few-public-methods, no-member, pointless-string-statement
"""Module containing MeshBlockActor class and related functions."""

from math import floor, log2
from typing import Tuple, Dict, List
import time
import ray
from ray import ObjectRef
import numpy as np
import mesh as me


def check_neighbor_ready(func):
    """Check if the neighbor is ready."""

    def wrapper(self, offsets):
        waiting_actors = set(self.neighbors[offsets])

        while waiting_actors:
            actor = waiting_actors.pop()
            ready = ray.get(actor.get_status.remote())
            if not ready:
                waiting_actors.add(actor)
                continue

        return func(self, offsets)

    return wrapper


@ray.remote
class MeshBlockActor:
    """Remotely launch actors as mesh blocks."""

    def __init__(self) -> None:
        """Initialize MeshBlockActor."""
        self.mblock = None
        self.logicloc = (1, 1, 1)
        self.neighbors = {}

    def new(self, node: me.BlockTree, seed: int = 0,
            coordinate_type: str = "cartesian") -> None:
        """Initialize MeshBlockActor from its tree node."""
        self.mblock = me.MeshBlock(node.size, coordinate_type)
        self.mblock.allocate().fill_random(seed)
        self.logicloc = node.lx3, node.lx2, node.lx1

    def relaunch(self, refs: List[ObjectRef], root: me.BlockTree) -> None:
        """Restart MeshBlockActor from its tree node."""
        self.logicloc, data = ray.get(refs)
        node = root.find_node_by_logicloc(self.logicloc)
        self.mblock = me.MeshBlock(node.size)
        self.mblock.allocate(data.shape[-1])
        self.mblock.data[:] = data[:]
        self.neighbors = {}

    def work(self) -> None:
        """Update the interior of the mesh block."""
        time.sleep(10)
        thresholds = (0.1, 0.9)  # coarsen, refine
        x = np.random.rand(1)
        if x < thresholds[0]:
            return -1  # coarsen
        if x > thresholds[1]:
            return 1  # refine
        return 0

    def reset_status(self) -> None:
        """Reset the status of the mesh block."""
        self.mblock.is_ready = False

    def put_data(self):
        """Put the mesh block in Plasma store."""
        refs = ray.put([self.logicloc, self.mblock.data])
        return refs

    def get_view(self, my_offsets: Tuple[int, int, int]) -> np.ndarray:
        """Get the view of the interior."""
        nb_offsets = tuple(-x for x in my_offsets)
        return self.mblock.view[my_offsets], self.logicloc, nb_offsets

    def get_prolong(self, my_offsets: Tuple[int, int, int],
                    finer: me.Coordinates) -> np.ndarray:
        """Get the view of the mesh block with prolongation."""
        nb_offsets = tuple(-x for x in my_offsets)
        data = self.mblock.prolongated_view(my_offsets, finer)
        return data, self.logicloc, nb_offsets

    def get_restrict(self, my_offsets: Tuple[int, int, int],
                     coarser: me.Coordinates) -> np.ndarray:
        """Get the view of the mesh block with restriction."""
        nb_offsets = tuple(-x for x in my_offsets)
        data = self.mblock.restricted_view(my_offsets, coarser)
        return data, self.logicloc, nb_offsets

    def fill_internal_data(self, parent_actor: ObjectRef, logloc=None) -> None:
        """Fill the internal data of the mesh block."""
        if logloc is None:
            internal_view = ray.get(
                parent_actor.get_prolong.remote((0, 0, 0), self.mblock.coord))
            self.mblock.fill_data(internal_view[0])
        else:
            internal_view = ray.get(
                parent_actor.get_restrict.remote((0, 0, 0), self.mblock.coord))
            logloc = (logloc[0], 1 - logloc[1], logloc[2])
            self.mblock.part((0, 0, 0), logloc)[:] = internal_view[0]
        self.mblock.is_ready = True

    def update_neighbor(self, offsets: Tuple[int, int, int], root: me.BlockTree,
                        actors: Dict[Tuple[int, int, int], ObjectRef]) -> None:
        """Update the neighbors of the mesh block."""
        node = root.find_node(self.mblock.size.center())
        neighbors = node.find_neighbors(offsets, self.mblock.coord)

        self.neighbors[offsets] = [
            actors[(nb.lx3, nb.lx2, nb.lx1)] for nb in neighbors
        ]

    def level_diff(self, logicloc: Tuple[int, int, int]) -> int:
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

    @check_neighbor_ready
    def update_ghost(self, offsets: Tuple[int, int, int]) -> [ObjectRef]:
        """Launch ghost-cell tasks."""
        if offsets not in self.neighbors:
            return []

        nbs = self.neighbors[offsets]
        if len(nbs) == 0:
            return []

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

    def get_data(self) -> (me.MeshBlock, me.RegionSize, int):
        """Print the mesh block."""
        node_id = ray.get_runtime_context().get_node_id()
        worker_id = ray.get_runtime_context().get_worker_id()
        return self.mblock, node_id, worker_id

    def get_logicloc(self) -> [int]:
        """Get the logic location of the mesh block."""
        return self.logicloc

    def get_coord(self) -> me.CoordinateFactory:
        """Return coord of mblock."""
        return self.mblock.coord

    def get_status(self) -> bool:
        """Get the status of the interior updates."""
        if self.mblock is None:
            return False
        return self.mblock.is_ready
