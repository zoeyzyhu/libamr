# pylint: disable = import-error, too-many-nested-blocks, too-many-instance-attributes, cyclic-import, too-many-arguments, too-many-locals, redefined-outer-name, too-many-boolean-expressions,too-many-branches,undefined-variable, fixme
"""Tree class and related functions."""

from math import floor, log2, ceil
from typing import Optional
from typing_extensions import Self
from .region_size import RegionSize
from .coordinates import Coordinates


class BlockTree:
    """A class representing a mesh block tree."""

    block_size = (1, 2, 2)  # default min block size

    @staticmethod
    def set_block_size(nx1: int, nx2: int = 1, nx3: int = 1) -> None:
        """Set the block size."""
        BlockTree.block_size = (nx3, nx2, nx1)

    def __init__(self, size: RegionSize,
                 logicloc: (int, int, int) = (1, 1, 1), parent=None):
        """Initialize BlockTree with size, level, and optional parent."""
        if size.nx3 % BlockTree.block_size[0] != 0 or \
           size.nx2 % BlockTree.block_size[1] != 0 or \
           size.nx1 % BlockTree.block_size[2] != 0:
            raise ValueError("region size is not divisible by block size")

        self.size = size
        self.lx3, self.lx2, self.lx1 = logicloc
        self.parent = parent
        self.children = []
        self.level = parent.level + 1 if parent else 0

    def root(self) -> Self:
        """Find the root of the tree."""
        node = self
        while node.parent:
            node = node.parent
        return node

    def node(self, *children_indices: [int]) -> Self:
        """Get the tree node at the given offset."""
        node = self
        for child_index in children_indices:
            node = node.children[child_index]
        return node

    def refine(self, locbit1: int, locbit2: int, locbit3: int) -> Self:
        """Refine a block at offset."""
        nx1 = self.size.nx1
        dx1 = (self.size.x1max - self.size.x1min) / (2. * nx1)
        x1min = self.size.x1min + locbit1 * dx1 * nx1
        x1max = self.size.x1max - (1 - locbit1) * dx1 * nx1

        nx2 = self.size.nx2
        if nx2 > 1:
            dx2 = (self.size.x2max - self.size.x2min) / (2. * nx2)
            x2min = self.size.x2min + locbit2 * dx2 * nx2
            x2max = self.size.x2max - (1 - locbit2) * dx2 * nx2
        else:
            x2min = self.size.x2min
            x2max = self.size.x2max

        nx3 = self.size.nx3
        if nx3 > 1:
            dx3 = (self.size.x3max - self.size.x3min) / (2. * nx3)
            x3min = self.size.x3min + locbit3 * dx3 * nx3
            x3max = self.size.x3max - (1 - locbit3) * dx3 * nx3
        else:
            x3min = self.size.x3min
            x3max = self.size.x3max

        size = RegionSize(x1dim=(x1min, x1max, nx1),
                          x2dim=(x2min, x2max, nx2),
                          x3dim=(x3min, x3max, nx3))

        lx1 = self.lx1 * 2 + locbit1
        lx2 = self.lx2 * 2 + locbit2
        lx3 = self.lx3 * 2 + locbit3

        return BlockTree(size, (lx3, lx2, lx1), self)

    def split(self) -> None:
        """Split the block into 8 sub-blocks."""
        if len(self.children) > 0:
            raise ValueError(
                f"This block is not a leaf, can not split it: {self.lx3, self.lx2, self.lx1}")

        locbit1_range = [0, 1]
        locbit2_range = [0, 1] if self.size.nx2 > 1 else [0]
        locbit3_range = [0, 1] if self.size.nx3 > 1 else [0]

        for locbit3 in locbit3_range:
            for locbit2 in locbit2_range:
                for locbit1 in locbit1_range:
                    self.children.append(
                        self.refine(locbit1, locbit2, locbit3))

    def merge(self):
        """Merge children blocks into a parent block."""
        self.children.clear()

    def divide_mesh(self) -> None:
        """Divide the block into 8 sub-blocks."""
        def split_dimension(dim, nb, xmin, xmax, nx):
            """Calculate the split dimension."""
            if nb > 1:
                nbl = 2 ** floor(log2(nb - 0.1))
                nbr = nb - nbl
                xmid = (nbr * xmin + nbl * xmax) / nb
                return [(xmin, xmid, nbl * BlockTree.block_size[3 - dim]),
                        (xmid, xmax, nbr * BlockTree.block_size[3 - dim])]
            return [(xmin, xmax, nx)]

        nb1 = self.size.nx1 // BlockTree.block_size[2]
        nb2 = self.size.nx2 // BlockTree.block_size[1]
        nb3 = self.size.nx3 // BlockTree.block_size[0]

        x1dims = split_dimension(
            1, nb1, self.size.x1min, self.size.x1max, self.size.nx1)
        x2dims = split_dimension(
            2, nb2, self.size.x2min, self.size.x2max, self.size.nx2)
        x3dims = split_dimension(
            3, nb3, self.size.x3min, self.size.x3max, self.size.nx3)

        for k, x3dim in enumerate(x3dims):
            for j, x2dim in enumerate(x2dims):
                for i, x1dim in enumerate(x1dims):
                    size = RegionSize(x1dim=x1dim, x2dim=x2dim, x3dim=x3dim)
                    lx1 = self.lx1 * 2 + i
                    lx2 = self.lx2 * 2 + j
                    lx3 = self.lx3 * 2 + k
                    self.children.append(
                        BlockTree(size, (lx3, lx2, lx1), self))

    def create_tree(self, max_level=None) -> None:
        """Create the tree."""
        nb1 = self.size.nx1 // BlockTree.block_size[2]
        nb2 = self.size.nx2 // BlockTree.block_size[1]
        nb3 = self.size.nx3 // BlockTree.block_size[0]

        if max_level is None:
            max_level = ceil(log2(max(nb1, nb2, nb3)))

        if self.level >= max_level:
            return

        self.divide_mesh()
        for child in self.children:
            child.create_tree(max_level)

    def find_node(self, point: (float, float, float)) -> Optional[Self]:
        """Find the block that contains the point."""
        x3, x2, x1 = point
        if x3 < self.size.x3min or x3 > self.size.x3max:
            return None
        if x2 < self.size.x2min or x2 > self.size.x2max:
            return None
        if x1 < self.size.x1min or x1 > self.size.x1max:
            return None

        if len(self.children) == 0:
            return self

        for child in self.children:
            node = child.find_node(point)
            if node is not None:
                return node

        return None

    def find_node_by_logicloc(self, logicloc: (int, int, int), level = None) -> Optional[Self]:
        """Find the block that has the logic location."""
        if logicloc == (self.lx3, self.lx2, self.lx1):
            return self

        if level is None:
            level = floor(log2(logicloc[2]))
        if level == 0:
            return None

        lx3 = logicloc[0] >> (level - 1)
        lx2 = logicloc[1] >> (level - 1)
        lx1 = logicloc[2] >> (level - 1)

        for child in self.children:
            if child.lx1 == lx1 and child.lx2 == lx2 and child.lx3 == lx3:
                return child.find_node_by_logicloc(logicloc, level - 1)

        return None

    def find_neighbors(self, offset: (int, int, int), coord: Coordinates) -> Self:
        """Neighbor generator of the block."""
        si, ei, sj, ej, sk, ek = self.size.ghost_range(offset)
        neighbors = []
        root = self.root()

        for k in range(sk, ek):
            for j in range(sj, ej):
                for i in range(si, ei):
                    point = coord.x3v[k], coord.x2v[j], coord.x1v[i]
                    nb = root.find_node(point)
                    if nb is not None and nb not in neighbors:
                        neighbors.append(nb)

        return neighbors

    def print_tree(self) -> None:
        """Print the tree."""
        print(self)
        for child in self.children:
            child.print_tree()

    def __str__(self):
        """Return a string representation of the node."""
        return f"\nlevel={self.level}: " + \
               f"lx3={str(bin(self.lx3))[2:]}, " + \
               f"lx2={str(bin(self.lx2))[2:]}, " + \
               f"lx1={str(bin(self.lx1))[2:]}\n" + \
               f"size={self.size}\nchildren={self.children}"

    def __eq__(self, other) -> bool:
        """Check if two BlockTree instances are equal."""
        if isinstance(other, BlockTree):
            return self.size == other.size and \
                self.lx1 == other.lx1 and self.lx2 == other.lx2 and \
                self.lx3 == other.lx3 and self.level == other.level and \
                self.children == other.children
        return False
