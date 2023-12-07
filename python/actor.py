# pylint: disable = import-error, unspecified-encoding, too-many-arguments, undefined-variable, too-many-public-methods, unused-argument, redefined-outer-name, too-few-public-methods, no-member, pointless-string-statement
"""Module containing MeshBlockActor class and related functions."""

from math import floor, log2
from typing import Tuple, Dict, List
import time
from datetime import datetime
import ray
from ray import ObjectRef
import numpy as np
import mesh as me
import random 

def check_neighbor_ready(func):
    """Check if the neighbor is ready."""

    def wrapper(self, offset):
        waiting_actors = set(self.neighbors[offset])

        while waiting_actors:
            actor = waiting_actors.pop()
            ready = ray.get(actor.get_status.remote())
            if not ready:
                waiting_actors.add((actor))
                continue

        return func(self, offset)

    return wrapper

@ray.remote(num_cpus=1)
class MeshBlockActor:
    """Remotely launch actors as mesh blocks."""

    def __init__(self) -> None:
        """Initialize MeshBlockActor."""
        self.mblock = None
        self.logicloc = (1, 1, 1)
        self.neighbors = {}
        self.neighbor_locs = {}

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
        self.neighbor_locs = {}

    def reset_status(self) -> None:
        """Reset the status of the mesh block."""
        self.mblock.is_ready = False

    def work(self) -> int:
        """Update the interior of the mesh block."""
        start_time = time.time()

        self.run_stencil()

        stime = datetime.fromtimestamp(
            start_time).strftime('%Y-%m-%d %H:%M:%S')
        etime = datetime.fromtimestamp(
            time.time()).strftime('%Y-%m-%d %H:%M:%S')
        duration = time.time() - start_time

        with open(f"log/work_{self.logicloc}.txt", "w") as f:
            f.write(f"\n\n\n{self.logicloc} {self.mblock.is_ready}\n")
            f.write(f"Start time: {stime}\n")
            f.write(f"End time: {etime}\n")
            f.write(f"Duration: {duration} seconds\n")

        thresholds = (0.2, 0.8)  # coarsen, refine
        return self.check_refine(*thresholds), self.mblock.size.center()

    def run_stencil(self) -> None:
        """Calculate stencil for interior block."""
        if not self.mblock.is_ready:
            raise ValueError("interior matrix is not ready")

        def diffusion_2d(arr):
            return arr[0,1:-1,:-2,:] + arr[0,1:-1,2:,:] \
                    + arr[0,2:,1:-1,:] + arr[0,:-2,1:-1,:] \
                    - 4. * arr[0,1:-1,1:-1,:]

        level = floor(log2(self.logicloc[2]))

        diffusivity = min(0.00001 * (2 ** level), 1.)
        iter_times = 100000 * level
        key = (0,0,0)

        for n in range(iter_times):
            self.mblock.ghost[key] += diffusivity * diffusion_2d(self.mblock.data)

        #for n in range(10):
        #    i = random.randint(0, self.mblock.size.nx1)
        #    j = random.randint(0, self.mblock.size.nx2)
        #    self.mblock.ghost[key][i,j] += np.random.normal(0, 10.) / level

    def check_refine(self, low: float, high: float) -> int:
        key = (0,0,0)

        ddx = abs(self.mblock.data[0,1:-1,1:,:] - self.mblock.data[0,1:-1,:-1,:])
        dx = self.mblock.coord.x1v[1] - self.mblock.coord.x1v[0]
        ddx_min, ddx_max = ddx.min() / dx, ddx.max() / dx

        if ddx_max > high:
            return 1

        ddy = abs(self.mblock.data[0,1:,1:-1,:] - self.mblock.data[0,:-1,1:-1,:])
        dy = self.mblock.coord.x2v[1] - self.mblock.coord.x2v[0]
        ddy_min, ddy_max = ddy.min() / dy, ddy.max() / dy

        if ddy_max > high:
            return 1

        if ddx_min < low and ddy_min < low:
            return -1

        return 0

    def put_data(self):
        """Put the mesh block in Plasma store."""
        refs = ray.put([self.logicloc, self.mblock.data])
        return refs

    def get_view(self, my_offset: Tuple[int, int, int]) -> np.ndarray:
        """Get the view of the interior."""
        nb_offset = tuple(-x for x in my_offset)
        return self.mblock.view[my_offset], self.logicloc, nb_offset

    def get_prolong(self, my_offset: Tuple[int, int, int],
                    finer: me.Coordinates) -> np.ndarray:
        """Get the view of the mesh block with prolongation."""
        nb_offset = tuple(-x for x in my_offset)
        data = self.mblock.prolongated_view(my_offset, finer)
        return data, self.logicloc, nb_offset

    def get_restrict(self, my_offset: Tuple[int, int, int],
                     coarser: me.Coordinates) -> np.ndarray:
        """Get the view of the mesh block with restriction."""
        nb_offset = tuple(-x for x in my_offset)
        data = self.mblock.restricted_view(my_offset, coarser)
        return data, self.logicloc, nb_offset

    def fill_interior_data(self, parent_actor: ObjectRef, logloc=None) -> None:
        """Fill the interior data of the mesh block."""
        if logloc is None:  # prolongation
            interior_view = ray.get(
                parent_actor.get_prolong.remote((0, 0, 0), self.mblock.coord))
            self.mblock.fill_data(interior_view[0])
        else:  # restriction
            interior_view = ray.get(
                parent_actor.get_restrict.remote((0, 0, 0), self.mblock.coord))
            logloc = (logloc[0], 1 - logloc[1], logloc[2])
            self.mblock.part((0, 0, 0), logloc)[:] = interior_view[0]

    def get_avg(self) -> np.ndarray:
        """Get the sum of the mesh block."""
        nvar = self.mblock.data.shape[-1]
        ngrids = self.mblock.view[(0, 0, 0)].size / nvar
        return np.array([self.mblock.view[(0, 0, 0)][:, :, :, i].sum() / ngrids
                         for i in range(nvar)])

    def fix_interior_data(self, diff: np.ndarray, logloc=None) -> None:
        """Fix the internal data of the mesh block."""
        self.mblock.ghost[(0, 0, 0)] += diff.reshape((1, 1, 1, -1))
        self.mblock.is_ready = True

    def update_neighbor(self, offset: Tuple[int, int, int], root: me.BlockTree,
                        actors: Dict[Tuple[int, int, int], ObjectRef]) -> None:
        """Update the neighbors of the mesh block."""
        node = root.find_node(self.mblock.size.center())
        neighbors = node.find_neighbors(offset, self.mblock.coord)

        self.neighbors[offset] = [
            actors[(nb.lx3, nb.lx2, nb.lx1)] for nb in neighbors
        ]

        self.neighbor_locs[offset] = [
            (nb.lx3, nb.lx2, nb.lx1) for nb in neighbors
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
            view, logicloc, offset = ray.get(task)
            if self.level_diff(logicloc) < 0:  # neighbor at finer level
                self.mblock.part(offset, logicloc)[:] = view
            else:
                self.mblock.ghost[offset][:] = view

        return remain_tasks

    @check_neighbor_ready
    def update_ghost(self, offset: Tuple[int, int, int]) -> [ObjectRef]:
        """Launch ghost-cell tasks."""
        if offset not in self.neighbors:
            return []

        nbs = self.neighbors[offset]
        if len(nbs) == 0:
            return []

        nb_offset = tuple(-x for x in offset)

        if len(nbs) > 1:  # neighbors at finer level
            tasks = [nb.get_restrict.remote(
                nb_offset, self.mblock.coord) for nb in nbs]
        # neighbor at coarser level
        elif self.level_diff(ray.get(nbs[0].get_logicloc.remote())) > 0:
            tasks = [nbs[0].get_prolong.remote(
                nb_offset, self.mblock.coord)]
        else:  # neighbor at same level
            tasks = [nbs[0].get_view.remote(nb_offset)]

        return tasks

    def get_data(self) -> (me.MeshBlock, me.RegionSize, int):
        """Print the mesh block."""
        node_id = ray.get_runtime_context().get_node_id()
        worker_id = ray.get_runtime_context().get_worker_id()
        return self.mblock, node_id, worker_id

    def get_logicloc(self) -> [int]:
        """Get the logic location of the mesh block."""
        return self.logicloc

    def get_neighbor_locs(self) -> Dict[Tuple[int, int, int],
                                        Tuple[ObjectRef, Tuple[int, int, int]]]:
        """Get the list of my neighbors' logiclocs."""
        return self.neighbor_locs

    def get_coord(self) -> me.CoordinateFactory:
        """Return coord of mblock."""
        return self.mblock.coord

    def get_status(self) -> bool:
        """Get the status of the interior updates."""
        if self.mblock is None:
            return False
        return self.mblock.is_ready
