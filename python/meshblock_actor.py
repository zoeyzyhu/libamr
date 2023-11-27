# pylint: disable = import-error, too-many-arguments, undefined-variable, unused-argument, redefined-outer-name, too-few-public-methods, no-member, pointless-string-statement
"""Module containing MeshBlockActor class and related functions."""

import ray
import numpy as np
import meshblock as mb

ray.init(runtime_env={"py_modules": [mb]})

# @ray.remote


def put_neighbors_info(node: mb.MeshBlockTree) -> None:
    """Put the neighbor information of a node into the object store."""
    neighbor_info = {}
    for ox3 in [-1, 0, 1]:
        for ox2 in [-1, 0, 1]:
            for ox1 in [-1, 0, 1]:
                cubic_offset = (ox3, ox2, ox1)
                if ox1 == 0 and ox2 == 0 and ox3 == 0:
                    continue
                neighbors = node.find_neighbors(cubic_offset)
                neighbor_info[cubic_offset] = neighbors

    neighbor_object = ray.put(neighbor_info)
    return neighbor_object


# @ray.remote
class MeshBlockActor:
    """Remotely launch actors as mesh blocks."""

    def __init__(self, size: mb.RegionSize, coordinate_type: str,
                 tree: mb.MeshBlockTree, nghost: int = 0):
        """Initialize MeshBlockActor with a mesh block and its corresponding tree."""
        #self.mblock = mblock
        #self.tree_node = tree.find_node(mblock)
        #self.neighbor_object = ray.put(self.mblock)

    def get_view(self, cubic_offset: (int, int, int), ox1: int = 0,
                 ox2: int = 0, ox3: int = 0, method: str = "") -> np.ndarray:
        """Get a view of the mesh block with optional prolongation or restriction."""
        # prolongation and restriction go here
        if method == 'prolongation':
            return self.mblock.prolongated_view(cubic_offset, ox1, ox2, ox3)
        if method == 'restriction':
            return self.mblock.restricted_view(cubic_offset, ox1, ox2, ox3)
        return self.mblock.view[cubic_offset]

    def update_ghost(self, cubic_offset: (int, int, int)) -> None:
        """Update ghost cells based on neighboring blocks."""
        neighbors = ray.get(self.neighbor_object)[cubic_offset]

        if len(neighbors) > 1:  # neighbors at finer level
            tasks = [neighbor.get_view.remote(-cubic_offset, neighbor.lx1, neighbor.lx2,
                                              neighbor.lx3, "prolongation")
                     for neighbor in neighbors]
        elif neighbors[0].level < self.mblock.level:  # neighbors at coarser level
            tasks = [neighbor.get_view.remote(-cubic_offset, neighbor.lx1, neighbor.lx2,
                                              neighbor.lx3, "restriction")
                     for neighbor in neighbors]
        else:  # neighbors at the same level
            tasks = [neighbor.get_view.remote(-cubic_offset)
                     for neighbor in neighbors]

        while len(tasks) > 0:
            ready_tasks, remain_tasks = ray.wait(tasks)
            for task in ready_tasks:
                view, level, logicloc = ray.get(task)
                if level <= self.mblock.level:
                    self.mblock.ghost[cubic_offset][:] = view
                else:
                    self.mblock.part(cubic_offset, logicloc)[:] = view
            tasks = remain_tasks


if __name__ == '__main__':
    pass
