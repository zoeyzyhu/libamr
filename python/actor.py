# pylint: disable = import-error, too-many-arguments, undefined-variable, unused-argument, redefined-outer-name, too-few-public-methods, no-member, pointless-string-statement
"""Module containing MeshBlockActor class and related functions."""

from typing import List
from typing import Tuple
import ray
from ray import ObjectRef
import numpy as np
import mesh as me


@ray.remote
class MeshBlockActor:
    """Remotely launch actors as mesh blocks."""

    def __init__(self) -> None:
        """Initialize MeshBlockActor."""
        self.node_id = ray.get_runtime_context().get_node_id()
        self.worker_id = ray.get_runtime_context().get_worker_id()
        self.size = None
        self.mblock = None

    def new(self, size: me.RegionSize, coordinate_type: str = "cartesian") -> None:
        """Initialize MeshBlockActor from its tree node."""
        self.size = size
        self.mblock = me.MeshBlock(size, coordinate_type)
        self.mblock.allocate().fill_random()

    def relaunch(self, data_ref: ObjectRef) -> None:
        """Restart MeshBlockActor from its tree node."""
        self.mblock = ray.get(data_ref)

    def get_data(self) -> Tuple[me.MeshBlock, me.RegionSize, int]:
        """Print the mesh block."""
        return self.size, self.mblock, self.node_id, self.worker_id

    def put_data(self):
        """Put the mesh block in Plasma store."""
        return ray.put(self.mblock)

    def get_view(self, offset: (int, int, int)) -> np.ndarray:
        """Get the view of the interior."""
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
    if not node.children:
        actor = MeshBlockActor.remote()
        actor.new.remote(node.size)
        actors.append(actor)
    else:
        for leaf in node.children:
            if leaf:
                actors.extend(launch_actors(leaf))
    return actors


def print_actors(actors: List[MeshBlockActor]) -> None:
    """Print the mesh block."""
    for actor in actors:
        size, mblock, node_id, worker_id = ray.get(actor.get_data.remote())
        print(f"\nNode:{node_id}\nWorker:{worker_id}\n{size}")
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
