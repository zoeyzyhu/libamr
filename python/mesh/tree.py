# pylint: disable = import-error, too-many-nested-blocks, too-many-instance-attributes, cyclic-import, too-many-arguments, too-many-locals, redefined-outer-name, too-many-boolean-expressions,too-many-branches,undefined-variable, fixme
"""Tree class and related functions."""

from math import floor, log2
from typing import Optional
from typing import Tuple
from typing import List
from typing_extensions import Self
from .region_size import RegionSize


class Tree:
    """A class representing a mesh block tree."""

    max_num_leaves = 8  # default # leaves in 3d after split
    block_size = (1, 1, 1)  # default min block size

    @staticmethod
    def set_block_size(nx1: int, nx2: int = 1, nx3: int = 1) -> None:
        """Set the block size."""
        Tree.block_size = (nx1, nx2, nx3)

    def __init__(self, size: RegionSize,
                 logicloc: Tuple[int, int, int] = (0, 0, 0), parent=None):
        """Initialize Tree with size, level, and optional parent."""
        self.size = size
        self.lx3, self.lx2, self.lx1 = logicloc
        self.parent = parent
        self.children = []
        self.level = parent.level + 1 if parent else 0

    def generate_child(self, ox1: int = 0, ox2: int = 0, ox3: int = 0) -> Optional[Self]:
        """Generate a child block without refinement."""
        nb1 = self.size.nx1 // Tree.block_size[0]
        nb2 = self.size.nx2 // Tree.block_size[1]
        nb3 = self.size.nx3 // Tree.block_size[2]

        if (ox1 == 1 and nb1 == 1) or \
           (ox2 == 1 and nb2 == 1) or \
           (ox3 == 1 and nb3 == 1):
            return None

        if nb1 == 1 and nb2 == 1 and nb3 == 1:
            return None

        x1min = self.size.x1min
        x1max = self.size.x1max
        dx1 = (self.size.x1max - self.size.x1min) / self.size.nx1
        nx1 = 2 ** floor(log2(nb1)) * Tree.block_size[0]
        # even split
        if nx1 == self.size.nx1 and nb1 > 1:
            nx1 = self.size.nx1 // 2
        if ox1 == 0:
            x1max = self.size.x1min + dx1 * nx1
        else:
            nx1 = self.size.nx1 - nx1
            x1min = self.size.x1max - dx1 * nx1

        x2min = self.size.x2min
        x2max = self.size.x2max
        dx2 = (self.size.x2max - self.size.x2min) / self.size.nx2
        nx2 = 2 ** floor(log2(nb2)) * Tree.block_size[1]
        # even split
        if nx2 == self.size.nx2 and nb2 > 1:
            nx2 = self.size.nx2 // 2
        if ox2 == 0:
            x2max = self.size.x2min + dx2 * nx2
        else:
            nx2 = self.size.nx2 - nx2
            x2min = self.size.x2max - dx2 * nx2

        x3min = self.size.x3min
        x3max = self.size.x3max
        dx3 = (self.size.x3max - self.size.x3min) / self.size.nx3
        nx3 = 2 ** floor(log2(nb3)) * Tree.block_size[2]
        # even split
        if nx3 == self.size.nx3 and nb3 > 1:
            nx3 = self.size.nx3 // 2
        if ox3 == 0:
            x3max = self.size.x3min + dx3 * nx3
        else:
            nx3 = self.size.nx3 - nx3
            x3min = self.size.x3max - dx3 * nx3

        rs = RegionSize(x1dim=(x1min, x1max, nx1),
                        x2dim=(x2min, x2max, nx2),
                        x3dim=(x3min, x3max, nx3))

        lx1 = self.lx1 * 2 + ox1
        lx2 = self.lx2 * 2 + ox2
        lx3 = self.lx3 * 2 + ox3

        return Tree(rs, (lx3, lx2, lx1), self)

    def generate_child_refine(self, ox1: int = 0, ox2: int = 0, ox3: int = 0) -> Self:
        """Generate a child block with refinement."""
        nx1 = self.size.nx1
        dx1 = (self.size.x1max - self.size.x1min) / (2. * nx1)
        x1min = self.size.x1min + ox1 * dx1 * nx1
        x1max = self.size.x1max - (1 - ox1) * dx1 * nx1

        nx2 = self.size.nx2
        if nx2 > 1:
            dx2 = (self.size.x2max - self.size.x2min) / (2. * nx2)
            x2min = self.size.x2min + ox2 * dx2 * nx2
            x2max = self.size.x2max - (1 - ox2) * dx2 * nx2
        else:
            x2min = self.size.x2min
            x2max = self.size.x2max

        nx3 = self.size.nx3
        if nx3 > 1:
            dx3 = (self.size.x3max - self.size.x3min) / (2. * nx3)
            x3min = self.size.x3min + ox3 * dx3 * nx3
            x3max = self.size.x3max - (1 - ox3) * dx3 * nx3
        else:
            x3min = self.size.x3min
            x3max = self.size.x3max

        rs = RegionSize(x1dim=(x1min, x1max, nx1),
                        x2dim=(x2min, x2max, nx2),
                        x3dim=(x3min, x3max, nx3))

        lx1 = self.lx1 * 2 + ox1
        lx2 = self.lx2 * 2 + ox2
        lx3 = self.lx3 * 2 + ox3

        return Tree(rs, (lx3, lx2, lx1), self)

    def split_block(self, refine=True) -> None:
        """Split the block into 8 sub-blocks."""
        if len(self.children) > 0:
            raise ValueError("This block is not a leaf, can not split it")

        self.children = [None] * self.max_num_leaves

        if self.size.nx1 > 1:
            ox1_range = [0, 1]
        else:
            ox1_range = [0]

        if self.size.nx2 > 1:
            ox2_range = [0, 1]
        else:
            ox2_range = [0]

        if self.size.nx3 > 1:
            ox3_range = [0, 1]
        else:
            ox3_range = [0]

        for ox3 in ox3_range:
            for ox2 in ox2_range:
                for ox1 in ox1_range:
                    child_index = ox1 + ox2 * 2 + ox3 * 4
                    if refine:
                        self.children[child_index] = self.generate_child_refine(
                            ox1, ox2, ox3)
                    else:
                        self.children[child_index] = self.generate_child(
                            ox1, ox2, ox3)

        all_none = True
        for child in self.children:
            if child is not None:
                all_none = False
                break
        if all_none:
            self.children = []
        else:
            if refine:
                # print("original from", self)
                for ox3 in [-1, 0, 1]:
                    for ox2 in [-1, 0, 1]:
                        for ox1 in [-1, 0, 1]:
                            if ox1 == 0 and ox2 == 0:  # and ox3 == 0
                                continue
                            cubic_offset = (ox3, ox2, ox1)
                            neighbors = self.find_neighbors(cubic_offset)
                            for neighbor in neighbors:
                                if neighbor is not None:
                                    # print(cubic_offset, neighbor)
                                    neighbor.split_block_chain(self.level)

    def split_block_chain(self, neighbor_level):
        """Split the block chain."""
        if self.level - neighbor_level >= 0:
            # print("does not need to split")
            return
        self.split_block()

    def merge_blocks(self):
        """Merge children blocks into a parent block."""

    def calculate_intervals1(self, cubic_offset: Tuple[int, int, int]) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
        """Calculate x intervals based on cubic offset."""
        x1_ghost = (self.size.x1max - self.size.x1min) / \
            self.size.nx1 * self.size.nghost
        x2_ghost = (self.size.x2max - self.size.x2min) / \
            self.size.nx2 * self.size.nghost
        x3_ghost = (self.size.x3max - self.size.x3min) / \
            self.size.nx3 * self.size.nghost

        x1_interval = self.calculate_interval(
            self.size.x1min, self.size.x1max, cubic_offset[2], x1_ghost)
        x2_interval = self.calculate_interval(
            self.size.x2min, self.size.x2max, cubic_offset[1], x2_ghost)
        x3_interval = self.calculate_interval(
            self.size.x3min, self.size.x3max, cubic_offset[0], x3_ghost)

        return x1_interval, x2_interval, x3_interval

    def calculate_interval1(self, min_val: int, max_val: int, offset: int, ghost: float) -> Tuple[int, int]:
        """Calculate interval based on offset and ghost."""
        if offset == 0:
            return min_val, max_val
        elif offset == 1:
            return max_val, max_val + ghost
        else:
            return min_val - ghost, min_val

    def calculate_intervals(self, cubic_offset: Tuple[int, int, int]) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
        """Calculate x intervals based on cubic offset."""
        self_x1s, self_x1e = self.size.x1min, self.size.x1max
        self_x2s, self_x2e = self.size.x2min, self.size.x2max
        self_x3s, self_x3e = self.size.x3min, self.size.x3max

        x1_interval = x2_interval = x3_interval = None

        x1_ghost = (self_x1e - self_x1s) / self.size.nx1 * self.size.nghost
        x2_ghost = (self_x2e - self_x2s) / self.size.nx2 * self.size.nghost
        x3_ghost = (self_x3e - self_x3s) / self.size.nx3 * self.size.nghost

        if cubic_offset[0] == 0:
            x3_interval = (self_x3s, self_x3e)
        elif cubic_offset[0] == 1:
            x3_interval = (self_x3e, self_x3e + x3_ghost)
        else:
            x3_interval = (self_x3s - x3_ghost, self_x3s)

        if cubic_offset[1] == 0:
            x2_interval = (self_x2s, self_x2e)
        elif cubic_offset[1] == 1:
            x2_interval = (self_x2e, self_x2e + x2_ghost)
        else:
            x2_interval = (self_x2s - x2_ghost, self_x2s)

        if cubic_offset[2] == 0:
            x1_interval = (self_x1s, self_x1e)
        elif cubic_offset[2] == 1:
            x1_interval = (self_x1e, self_x1e + x1_ghost)
        else:
            x1_interval = (self_x1s - x1_ghost, self_x1s)

        return x1_interval, x2_interval, x3_interval

    def find_neighbors(self, cubic_offset: (int, int, int)) -> [Self]:
        """Find neighbors of the block."""
        # -1, 0, 1 represent left, mid, right on viewpoint
        self_x1s, self_x1e = self.size.x1min, self.size.x1max
        self_x2s, self_x2e = self.size.x2min, self.size.x2max
        self_x3s, self_x3e = self.size.x3min, self.size.x3max

        x1_interval = x2_interval = x3_interval = None

        x1_ghost = (self_x1e - self_x1s) / self.size.nx1 * self.size.nghost
        x2_ghost = (self_x2e - self_x2s) / self.size.nx2 * self.size.nghost
        x3_ghost = (self_x3e - self_x3s) / self.size.nx3 * self.size.nghost

        if cubic_offset[0] == 0:
            x3_interval = (self_x3s, self_x3e)
        elif cubic_offset[0] == 1:
            x3_interval = (self_x3e, self_x3e + x3_ghost)
        else:
            x3_interval = (self_x3s - x3_ghost, self_x3s)

        if cubic_offset[1] == 0:
            x2_interval = (self_x2s, self_x2e)
        elif cubic_offset[1] == 1:
            x2_interval = (self_x2e, self_x2e + x2_ghost)
        else:
            x2_interval = (self_x2s - x2_ghost, self_x2s)

        if cubic_offset[2] == 0:
            x1_interval = (self_x1s, self_x1e)
        elif cubic_offset[2] == 1:
            x1_interval = (self_x1e, self_x1e + x1_ghost)
        else:
            x1_interval = (self_x1s - x1_ghost, self_x1s)

        neighbor = self.locate_neighbors_up(
            x1_interval, x2_interval, x3_interval)

        if neighbor is None:
            return []
        if len(neighbor.children) == 0:
            return [neighbor]
        return neighbor.children

    # only support 2d now
    def locate_neighbors_up(self, x1_interval: (int, int),
                            x2_interval: (int, int), x3_interval: (int, int)):
        """Locate neighbors up."""
        self_x1s, self_x1e = self.size.x1min, self.size.x1max
        self_x2s, self_x2e = self.size.x2min, self.size.x2max
        self_x3s, self_x3e = self.size.x3min, self.size.x3max

        if self_x1s <= x1_interval[0] < x1_interval[1] <= self_x1e and \
                self_x2s <= x2_interval[0] < x2_interval[1] <= self_x2e and \
                self_x3s <= x3_interval[0] < x3_interval[1] <= self_x3e:
            # already reach most recent common ancestor
            neighbor = self.locate_neighbors_down(
                x1_interval, x2_interval, x3_interval)

            return neighbor
        if self.level == 0:
            return None  # it is on board, ghost zone does not exist
        return self.parent.locate_neighbors_up(x1_interval, x2_interval, x3_interval)

    # only support 2d now
    def locate_neighbors_down(self, x1_interval: (int, int),
                              x2_interval: (int, int), x3_interval: (int, int)):
        """Locate neighbors down."""
        neighbor = None
        for child in self.children:
            if child is not None:
                child_x1s, child_x1e = child.size.x1min, child.size.x1max
                child_x2s, child_x2e = child.size.x2min, child.size.x2max
                child_x3s, child_x3e = child.size.x3min, child.size.x3max

                if child_x1s <= x1_interval[0] < x1_interval[1] <= child_x1e \
                        and child_x2s <= x2_interval[0] < x2_interval[1] <= child_x2e \
                        and child_x3s <= x3_interval[0] < x3_interval[1] <= child_x3e:

                    neighbor = child.locate_neighbors_down(
                        x1_interval, x2_interval, x3_interval)
                    break

        if neighbor is None:
            return self

        return neighbor

    def find_root(self) -> Self:
        """Find the root of the tree."""
        node = self
        while node.parent:
            node = node.parent
        return node

    def relocate_neighbors(self, root: Self, offsets: (int, int, int)) -> List[Self]:
        """Relocate the node in case tree outdated."""
        intervals = self.calculate_intervals(offsets)
        neighbor = root.locate_neighbors_down(*intervals)

        if neighbor is None:
            return []
        if len(neighbor.children) == 0:
            return [neighbor]
        return neighbor.children

    def get_child(self, ox1: int, ox2: int = 0, ox3: int = 0) -> Optional[Self]:
        """Get the child block."""
        child_index = ox1 + ox2 * 2 + ox3 * 4
        return self.children[child_index]

    def create_tree(self) -> None:
        """Create the tree."""
        nx1, nx2, nx3 = Tree.block_size
        if self.size.nx3 % nx3 != 0 or \
           self.size.nx2 % nx2 != 0 or \
           self.size.nx1 % nx1 != 0:
            raise ValueError("region size is not divisible by block size")

        self.split_block(refine=False)

        for child in self.children:
            if child is not None:
                child.create_tree()

    def print_tree(self) -> None:
        """Print the tree."""
        print(self)
        for child in self.children:
            if child is not None:
                child.print_tree()

    def __str__(self):
        """Return a string representation of the node."""
        return f"\nlevel={self.level}: " + \
               f"lx3={str(bin(self.lx3))[2:]}, " + \
               f"lx2={str(bin(self.lx2))[2:]}, " + \
               f"lx1={str(bin(self.lx1))[2:]}\n" + \
               f"size={self.size}\nchildren={self.children}"


if __name__ == "__main__":
    Tree.set_block_size(nx1=2, nx2=2, nx3=1)
    rs = RegionSize(x1dim=(0, 120., 8), x2dim=(0, 120., 4))
    root = Tree(rs)
    root.create_tree()

    print("\n\n===== split block =====")
    root.children[3].children[1].split_block()
    print(root.children[1].children[0])

    n1 = root.children[3].children[1].children[0]

    print("\n\n===== split block chain =====")

    n1.split_block()

    n1.children[0].split_block()

    root.children[0].children[0].split_block()

    root.print_tree()
    # print(root.leaf[3].leaf[1].leaf[0])
    # print(root.leaf[3].leaf[1].leaf[1])
    # print(root.leaf[3].leaf[1].leaf[2])
    # print(root.leaf[3].leaf[1].leaf[3])
    node = root.children[3].children[1].find_root()
    print(node)
