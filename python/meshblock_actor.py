# pylint: disable = import-error, too-many-arguments, undefined-variable, unused-argument, redefined-outer-name, too-few-public-methods, no-member, pointless-string-statement
"""Module containing MeshBlockActor class and related functions."""

from typing import List
from typing import Tuple
import ray
from ray import ObjectRef
import mesh as me
import numpy as np


@ray.remote
class MeshBlockActor:
    """Remotely launch actors as mesh blocks."""

    def __init__(self, tree: me.Tree,
                 coordinate_type: str, nghost: int) -> None:
        """Initialize MeshBlockActor from its tree node."""
        self.worker_id = ray.worker.global_worker.worker_id.hex()
        self.device_id = ray.get_runtime_context().get_node_id()
        self.node = tree
        self.mblock = me.MeshBlock(tree.size, coordinate_type, nghost)
        self.mblock.allocate().fill_random()

    def restart_from(self, data_ref: List[ObjectRef]) -> None:
        """Restart MeshBlockActor from its tree node."""
        self.worker_id = ray.worker.global_worker.worker_id.hex()
        self.device_id = ray.get_runtime_context().get_node_id()
        self.node, self.mblock = ray.get(data_ref)

    def get_data(self) -> Tuple[me.MeshBlock, me.RegionSize, int]:
        """Print the mesh block."""
        return self.mblock, self.node.size, self.worker_id, self.device_id

    def put_data(self):
        """Put the mesh block in Plasma store."""
        return ray.put([self.node, self.mblock])

    def get_view(self, offset: (int, int, int)) -> np.ndarray:
        """Get the view of the mesh block."""
        return self.mblock.view[offset]

    def get_view_prolong(self, my_offset: (int, int, int),
                         finer: me.Coordinates) -> np.ndarray:
        """Get the view of the mesh block with prolongation."""
        of1 = self.node.lx1 % 2
        of2 = self.node.lx2 % 2
        of3 = self.node.lx3 % 2
        logicloc = (of3, of2, of1)
        level = self.node.level  # check
        return self.mblock.prolongated_view(my_offset, finer), level, logicloc

    def get_view_restrict(self, my_offset: (int, int, int),
                          coarser: me.Coordinates) -> np.ndarray:
        """Get the view of the mesh block with restriction."""
        of1 = self.node.lx1 % 2
        of2 = self.node.lx2 % 2
        of3 = self.node.lx3 % 2
        logicloc = (of1, of2, of3)
        level = self.node.level
        return self.mblock.restricted_view(my_offset, coarser), level, logicloc

    def update_ghost(self, offset: (int, int, int), neighbor) -> None:
        """Update ghost cells."""
        node = self.tree.find_node(self.mblock)
        neighbors = node.find_neighbors(offset)

        if len(neighbors) > 1:  # neighbors at finer level
            tasks = [
                neighbor.get_view_restrict.remote(-offset, self.mblock.coord)
                for neighbor in neighbors]
        elif neighbors[0].level < self.mblock.level:  # neighbors at coarser level
            tasks = [
                neighbor.get_view_prolong.remote(-offset, self.mblock.coord)]
        else:  # neighbors at same level
            tasks = [neighbor.get_view.remote(-offset)]

        ready_tasks, remain_tasks = ray.wait(tasks)
        for task in ready_tasks:
            view, level, logicloc = ray.get(task)
            if level <= self.mblock.level:
                self.mblock.ghost[offset][:] = view
            else:
                self.mblock.part(offset, logicloc)[:] = view

        return remain_tasks


def launch_actors(node: me.Tree) -> List[MeshBlockActor]:
    """Launch actors based on the tree."""
    actors = []
    nghost = 1

    if not node.leaf:
        actor = MeshBlockActor.remote()
        actor.new.remote(node, "cartesian", nghost)
        actors.append(actor)
    else:
        for leaf in node.leaf:
            if leaf:
                actors.extend(launch_actors(leaf))

    return actors


def print_actors(actors: List[MeshBlockActor]) -> None:
    """Print the mesh block."""
    for actor in actors:
        mblock, size, worker_id, node = ray.get(actor.get_data.remote())
        print(f"\nNode:{node}\nWorker:{worker_id}\nSize: {size}\n")
        mblock.print_data()


if __name__ == '__main__':
    # Initial split without refinement
    me.Tree.set_block_size(nx1=2, nx2=2, nx3=1)
    rs = me.RegionSize(x1dim=(0, 120., 8), x2dim=(0, 120., 4))
    tree = me.Tree(rs)
    tree.create_tree()
    # tree.print_tree()

    # Launch actors based on the tree
    ray.init(runtime_env={"py_modules": [mb]})
    # print(ray.nodes())
    # print(ray.available_resources())
    actors = launch_actors(tree)

    print_actors(actors)

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
        actor.restart_from.remote(data_ref)
        new_actors.append(actor)

    print(new_actors)
    print_actors(new_actors)


    # Refine the tree
    node_to_refine = tree.leaf[3]
    node_to_refine.split_block()
    tree.print_tree()

    ray.kill(actors[3])
    new_actors = []
    launch_actors(node_to_refine, new_actors, nghost=1)
    actors = actors[:3] + new_actors + actors[4:]
    print_actors(actors)
    """
