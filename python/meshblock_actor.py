# pylint: disable = import-error, too-many-arguments, undefined-variable, unused-argument, redefined-outer-name, too-few-public-methods, no-member, pointless-string-statement
"""Module containing MeshBlockActor class and related functions."""

import numpy as np
import ray
from region_size import RegionSize
from meshblock import MeshBlock
from meshblock_tree import MeshBlockTree


# @ray.remote
def put_neighbors_info(node: MeshBlockTree) -> None:
    """Put the neighbor information of a node into the object store."""
    neighbor_info = {}
    for ox3 in [-1, 0, 1]:
        for ox2 in [-1, 0, 1]:
            for ox1 in [-1, 0, 1]:
                offset = (ox3, ox2, ox1)
                if ox1 == 0 and ox2 == 0 and ox3 == 0:
                    continue
                neighbors = node.find_neighbors(offset)
                neighbor_info[offset] = neighbors

    neighbor_object = ray.put(neighbor_info)
    return neighbor_object


# @ray.remote
class MeshBlockActor:
    """Remotely launch actors as mesh blocks."""

    def __init__(self, mblock: MeshBlock, tree: MeshBlockTree):
        """Initialize MeshBlockActor with a mesh block and its corresponding tree."""
        self.mblock = mblock
        self.tree_node = tree.find_node(mblock)
        self.neighbor_object = ray.put(self.mblock)

    def get_view(self, offset: (int, int, int), ox1: int = 0,
                 ox2: int = 0, ox3: int = 0, method: str = "") -> np.ndarray:
        """Get a view of the mesh block with optional prolongation or restriction."""
        # prolongation and restriction go here
        if method == 'prolongation':
            return self.mblock.prolongated_view(offset, ox1, ox2, ox3)
        if method == 'restriction':
            return self.mblock.restricted_view(offset, ox1, ox2, ox3)
        return self.mblock.view[offset]

    def update_ghost(self, offset: (int, int, int)) -> None:
        """Update ghost cells based on neighboring blocks."""
        neighbors = ray.get(self.neighbor_object)[offset]

        if len(neighbors) > 1:  # neighbors at finer level
            tasks = [neighbor.get_view.remote(-offset, neighbor.lx1, neighbor.lx2,
                                              neighbor.lx3, "prolongation")
                     for neighbor in neighbors]
        elif neighbors[0].level < self.mblock.level:  # neighbors at coarser level
            tasks = [neighbor.get_view.remote(-offset, neighbor.lx1, neighbor.lx2,
                                              neighbor.lx3, "restriction")
                     for neighbor in neighbors]
        else:  # neighbors at the same level
            tasks = [neighbor.get_view.remote(-offset)
                     for neighbor in neighbors]

        while len(tasks) > 0:
            ready_tasks, remain_tasks = ray.wait(tasks)
            for task in ready_tasks:
                view, level, logicloc = ray.get(task)
                if level <= self.mblock.level:
                    self.mblock.ghost[offset][:] = view
                else:
                    self.mblock.part(offset, logicloc)[:] = view
            tasks = remain_tasks


@ray.remote
class BlockActor:
    """Remotely launch actors as mesh blocks."""

    def __init__(self, rs: RegionSize, tree: MeshBlockTree):
        """Initialize MeshBlockActor with a mesh block and its corresponding tree."""
        self.mb = MeshBlock(rs, "cartesian", nghost=1)

    def print_block(self):
        """Print the mesh block."""
        self.mb.print_data()


if __name__ == '__main__':
    # ray.init()
    """
    # Test local mesh block
    MeshBlockTree.set_block_size(nx1=2, nx2=2, nx3=1)
    rs = RegionSize(x1dim=(0, 120., 4), x2dim=(0, 120., 4))
    root = MeshBlockTree(rs)
    root.create_tree()
    root.print_tree()

    mb = MeshBlock(rs, "cartesian", nghost=1)
    mb.allocate().fill_random()
    mb.print_data()
    """
    # Test remote mesh block
    MeshBlockTree.set_block_size(nx1=2, nx2=2, nx3=1)
    rs = RegionSize(x1dim=(0, 120., 4), x2dim=(0, 120., 4))
    root = MeshBlockTree(rs)
    root.create_tree()
    actors = [BlockActor.remote(rs, root) for _ in range(2)]
    for actor in actors:
        actor.print_block.remote()

    #mesh_block = MeshBlock()
    #mesh_block_actor = MeshBlockActor.remote(mesh_block)

    #mesh_block_actor.get_view.remote((0, 0, 0))
    #mesh_block_actor.update_ghost.remote((0, 0, 0))

    # ray.shutdown()
