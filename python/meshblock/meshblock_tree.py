# pylint: disable = import-error, too-many-arguments, too-many-locals, redefined-outer-name, too-many-boolean-expressions,too-many-branches,undefined-variable, fixme
"""MeshBlockTree class and related functions."""

from math import floor, log2
from typing import Optional
from typing_extensions import Self
from .region_size import RegionSize
from .meshblock import MeshBlock


class MeshBlockTree:
    """A class representing a mesh block tree."""

    max_num_leaves = 8
    block_size = (1, 1, 1)  # default min block size

    @staticmethod
    def set_block_size(nx1: int, nx2: int = 1, nx3: int = 1) -> None:
        """Set the block size."""
        MeshBlockTree.block_size = (nx1, nx2, nx3)

    def __init__(self, size: RegionSize, lx1: int = 0, lx2: int = 0, lx3: int = 0, parent=None):
        """Initialize MeshBlockTree with size, level, and optional parent."""
        self.size = size
        self.lx1 = lx1
        self.lx2 = lx2
        self.lx3 = lx3

        if parent is None:
            self.level = 0
        else:
            self.level = parent.level + 1
        self.leaf = []

    def generate_leaf(self, ox1: int = 0, ox2: int = 0, ox3: int = 0) -> Optional[Self]:
        """Generate a leaf block without refinement."""
        nb1 = self.size.nx1 // MeshBlockTree.block_size[0]
        nb2 = self.size.nx2 // MeshBlockTree.block_size[1]
        nb3 = self.size.nx3 // MeshBlockTree.block_size[2]

        if (ox1 == 1 and nb1 == 1) or \
           (ox2 == 1 and nb2 == 1) or \
           (ox3 == 1 and nb3 == 1):
            return None

        if nb1 == 1 and nb2 == 1 and nb3 == 1:
            return None

        x1min = self.size.x1min
        x1max = self.size.x1max
        dx1 = (self.size.x1max - self.size.x1min) / self.size.nx1
        nx1 = 2 ** floor(log2(nb1)) * MeshBlockTree.block_size[0]
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
        nx2 = 2 ** floor(log2(nb2)) * MeshBlockTree.block_size[1]
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
        nx3 = 2 ** floor(log2(nb3)) * MeshBlockTree.block_size[2]
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

        return MeshBlockTree(rs, lx1, lx2, lx3, self)

    def generate_leaf_refine(self, ox1: int = 0, ox2: int = 0, ox3: int = 0) -> Self:
        """Generate a leaf block with refinement."""
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

        return MeshBlockTree(rs, lx1, lx2, lx3, self)

    def split_block(self, refine=True) -> None:
        """Split the block into 8 sub-blocks."""
        if len(self.leaf) > 0:
            raise ValueError("This block is not a leaf, can not split it")

        self.leaf = [None] * self.max_num_leaves

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
                    leaf_index = ox1 + ox2 * 2 + ox3 * 4
                    if refine:
                        self.leaf[leaf_index] = self.generate_leaf_refine(
                            ox1, ox2, ox3)
                    else:
                        self.leaf[leaf_index] = self.generate_leaf(
                            ox1, ox2, ox3)

        all_none = True
        for leaf in self.leaf:
            if leaf is not None:
                all_none = False
                break
        if all_none:
            self.leaf = []

    def merge_blocks(self):
        """Merge children blocks into a parent block."""

    def find_node(self, mblock: MeshBlock) -> Optional[Self]:
        """Find the node that contains the mesh block."""

    # TODO: implement this
    def find_neighbors(self, cubic_offset: (int, int, int)) -> [Self]:
        """Find neighbors of the block."""
        # 0, 1, 2, 3 represent left, right, up, down, if it is on board, return None

    def get_leaf(self, ox1: int, ox2: int = 0, ox3: int = 0) -> Optional[Self]:
        """Get the leaf block."""
        leaf_index = ox1 + ox2 * 2 + ox3 * 4
        return self.leaf[leaf_index]

    def create_tree(self) -> None:
        """Create the tree."""
        nx1, nx2, nx3 = MeshBlockTree.block_size
        if self.size.nx3 % nx3 != 0 or \
           self.size.nx2 % nx2 != 0 or \
           self.size.nx1 % nx1 != 0:
            raise ValueError("region size is not divisible by block size")

        self.split_block(refine=False)

        for leaf in self.leaf:
            if leaf is not None:
                leaf.create_tree()

    def print_tree(self) -> None:
        """Print the tree."""
        print(self)
        for leaf in self.leaf:
            if leaf is not None:
                leaf.print_tree()

    def __str__(self):
        """Return a string representation of the node."""
        return f"\nlevel={self.level}\nsize={self.size}\n" + \
               f"lx1={bin(self.lx1)},lx2={bin(self.lx2)},lx3={bin(self.lx3)}\nleaves={self.leaf}"


if __name__ == "__main__":
    MeshBlockTree.set_block_size(nx1=2, nx2=2, nx3=1)
    rs = RegionSize(x1dim=(0, 120., 8), x2dim=(0, 120., 4))
    root = MeshBlockTree(rs)
    root.create_tree()
    root.print_tree()

    print("\n\n===== split block =====")
    root.leaf[3].leaf[1].split_block()
    print(root.leaf[3].leaf[1])
    print(root.leaf[3].leaf[1].leaf[0])
    print(root.leaf[3].leaf[1].leaf[1])
    print(root.leaf[3].leaf[1].leaf[2])
    print(root.leaf[3].leaf[1].leaf[3])
