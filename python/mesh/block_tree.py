# pylint: disable = import-error, too-many-nested-blocks, too-many-instance-attributes, cyclic-import, too-many-arguments, too-many-locals, redefined-outer-name, too-many-boolean-expressions,too-many-branches,undefined-variable, fixme
"""Tree class and related functions."""

from math import floor, log2, ceil
from typing import Optional
from typing_extensions import Self
from .region_size import RegionSize
from .meshblock import MeshBlock
from .coordinates import Coordinates

class BlockTree:
    """A class representing a mesh block tree."""

    block_size = (1, 1, 1)  # default min block size

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

    def node(self, *ofs : [int]) -> Self:
        """Get the tree node at the given offset."""
        node = self
        for of in ofs:
            node = node.children[of]
        return node

    def refine(self, of1: int = 0, of2: int = 0, of3: int = 0) -> Self:
        """Refine a block at offset"""
        nx1 = self.size.nx1
        dx1 = (self.size.x1max - self.size.x1min) / (2. * nx1)
        x1min = self.size.x1min + of1 * dx1 * nx1
        x1max = self.size.x1max - (1 - of1) * dx1 * nx1

        nx2 = self.size.nx2
        if nx2 > 1:
            dx2 = (self.size.x2max - self.size.x2min) / (2. * nx2)
            x2min = self.size.x2min + of2 * dx2 * nx2
            x2max = self.size.x2max - (1 - of2) * dx2 * nx2
        else:
            x2min = self.size.x2min
            x2max = self.size.x2max

        nx3 = self.size.nx3
        if nx3 > 1:
            dx3 = (self.size.x3max - self.size.x3min) / (2. * nx3)
            x3min = self.size.x3min + of3 * dx3 * nx3
            x3max = self.size.x3max - (1 - of3) * dx3 * nx3
        else:
            x3min = self.size.x3min
            x3max = self.size.x3max

        rs = RegionSize(x1dim=(x1min, x1max, nx1),
                        x2dim=(x2min, x2max, nx2),
                        x3dim=(x3min, x3max, nx3))

        lx1 = self.lx1 * 2 + of1
        lx2 = self.lx2 * 2 + of2
        lx3 = self.lx3 * 2 + of3

        return BlockTree(rs, (lx3, lx2, lx1), self)

    def split(self) -> None:
        """Split the block into 8 sub-blocks."""
        if len(self.children) > 0:
            raise ValueError("This block is not a leaf, can not split it")

        of1_range = [0, 1]

        if self.size.nx2 > 1:
            of2_range = [0, 1]
        else:
            of2_range = [0]

        if self.size.nx3 > 1:
            of3_range = [0, 1]
        else:
            of3_range = [0]

        for of3 in of3_range:
            for of2 in of2_range:
                for of1 in of1_range:
                    self.children.append(self.refine(of1, of2, of3))

    def merge(self):
        """Merge children blocks into a parent block."""

    def divide(self) -> None:
        """Divide the block into 8 sub-blocks."""
        nb1 = self.size.nx1 // BlockTree.block_size[2]
        nb2 = self.size.nx2 // BlockTree.block_size[1]
        nb3 = self.size.nx3 // BlockTree.block_size[0]

        x1min, x1max = self.size.x1min, self.size.x1max
        x2min, x2max = self.size.x2min, self.size.x2max
        x3min, x3max = self.size.x3min, self.size.x3max

        if nb1 > 1:
            nbl = 2 ** floor(log2(nb1 - 0.1))
            nbr = nb1 - nbl
            x1mid = (nbr * x1min + nbl * x1max) / nb1
            x1dims =[(x1min, x1mid, nbl * BlockTree.block_size[2]),
                     (x1mid, x1max, nbr * BlockTree.block_size[2])]
        else:
            x1dims = [(x1min, x1max, self.size.nx1)]

        if nb2 > 1:
            nbl = 2 ** floor(log2(nb2 - 0.1))
            nbr = nb2 - nbl
            x2mid = (nbr * x2min + nbl * x2max) / nb2
            x2dims = [(x2min, x2mid, nbl * BlockTree.block_size[1]),
                      (x2mid, x2max, nbr * BlockTree.block_size[1])]
        else:
            x2dims = [(x2min, x2max, self.size.nx2)]

        if nb3 > 1:
            nbl = 2 ** floor(log2(nb3 - 0.1))
            nbr = nb3 - nbl
            x3mid = (nbr * x3min + nbl * x3max) / nb3
            x3dims = [(x3min, x3mid, nbl * BlockTree.block_size[0]),
                      (x3mid, x3max, nbr * BlockTree.block_size[0])]
        else:
            x3dims = [(x3min, x3max, self.size.nx3)]

        for k, x3dim in enumerate(x3dims):
            for j, x2dim in enumerate(x2dims):
                for i, x1dim in enumerate(x1dims):
                    rs = RegionSize(x1dim=x1dim, x2dim=x2dim, x3dim=x3dim)
                    lx1 = self.lx1 * 2 + i
                    lx2 = self.lx2 * 2 + j
                    lx3 = self.lx3 * 2 + k
                    self.children.append(BlockTree(rs, (lx3, lx2, lx1), self))

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

    def find_node_by_logicloc(self, logicloc: (int, int, int)) -> Optional[Self]:
        """Find the block that has the logic location."""
        if logicloc == (self.lx3, self.lx2, self.lx1):
            return self

        level = floor(log2(logicloc[2]))
        if level == 0:
            return None

        lx3 = logicloc[0] >> (level - 1)
        lx2 = logicloc[1] >> (level - 1)
        lx1 = logicloc[2] >> (level - 1)

        for child in self.children:
            if child.lx1 == lx1 and child.lx2 == lx2 and child.lx3 == lx3:
                lx3 = logicloc[0] >> 1
                lx2 = logicloc[1] >> 1
                lx1 = logicloc[2] >> 1
                return child.find_node_by_logicloc((lx3, lx2, lx1))

        return None

    def find_neighbors(self, offsets: (int, int, int), coord: Coordinates) -> Self:
        """neighbor generator of the block."""
        si, ei, sj, ej, sk, ek = self.size.ghost_range(offsets)
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

    def create_tree(self, max_level = None) -> None:
        """Create the tree."""

        nb1 = self.size.nx1 // BlockTree.block_size[2]
        nb2 = self.size.nx2 // BlockTree.block_size[1]
        nb3 = self.size.nx3 // BlockTree.block_size[0]

        if max_level is None:
            max_level = ceil(log2(max(nb1, nb2, nb3)))

        if self.level >= max_level:
            return
    
        self.divide()
        for child in self.children:
            child.create_tree(max_level)

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

