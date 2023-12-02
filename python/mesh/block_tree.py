# pylint: disable = import-error, too-many-nested-blocks, too-many-instance-attributes, cyclic-import, too-many-arguments, too-many-locals, redefined-outer-name, too-many-boolean-expressions,too-many-branches,undefined-variable, fixme
"""Tree class and related functions."""

from math import floor, log2
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
        BlockTree.block_size = (nx1, nx2, nx3)

    def __init__(self, size: RegionSize,
                 logicloc: (int, int, int) = (1, 1, 1), parent=None):
        """Initialize BlockTree with size, level, and optional parent."""
        if size.nx3 % block_size[0] != 0 or \
           size.nx2 % block_size[1] != 0 or \
           size.nx1 % block_size[2] != 0:
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

    def spawn(self, ox1: int = 0, ox2: int = 0, ox3: int = 0) -> Self:
        """Generate a child block."""
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

        return BlockTreeTree(rs, (lx3, lx2, lx1), self)

    def split_block(self) -> None:
        """Split the block into 8 sub-blocks."""
        if len(self.children) > 0:
            raise ValueError("This block is not a leaf, can not split it")

        ox1_range = [0, 1]

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
                    self.children.append(self.spawn(ox1, ox2, ox3))

    def merge_blocks(self, blocks: List[Self]) -> Self:
        """Merge blocks into a parent block."""

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
            if child is None:
                continue
            node = child.find_node(point)
            if node is not None:
                return node

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

    def create_tree(self) -> None:
        """Create the tree."""

        if self.size.nx3 > nx3 or \
           self.size.nx2 > nx2 or \
           self.size.nx1 > nx1:
        self.split_block()

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
