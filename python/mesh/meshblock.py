# pylint: disable = import-error, cyclic-import, too-many-instance-attributes, redefined-outer-name, too-many-locals, too-many-branches, too-many-statements, fixme
"""Module containing the MeshBlock class."""

from bisect import bisect
from typing import Any
from typing_extensions import Self
import numpy as np
from numpy.lib.stride_tricks import as_strided
from scipy.interpolate import interpn
from .region_size import RegionSize
from .coordinates_factory import CoordinateFactory
from .coordinates import Coordinates


class MeshBlock:
    """A class representing a mesh block."""

    def __init__(self, size: RegionSize, coordinate_type: str = "cartesian"):
        """Initialize MeshBlock with size, coordinate type, and optional nghost."""
        self.size = size
        self.coord = CoordinateFactory.create(size, coordinate_type)
        self.data = None
        self.view = {}
        self.ghost = {}

    def allocate(self, nvar: int = 1) -> Self:
        """Allocate memory for the mesh block."""
        self.size.nvar = int(nvar)
        if self.size.nvar < 1:
            raise ValueError("nvar must be >= 1")

        self.data = np.zeros(
            (self.size.nc3, self.size.nc2, self.size.nc1, self.size.nvar))

        # Interior slices
        for o3 in [-1, 0, 1]:
            for o2 in [-1, 0, 1]:
                for o1 in [-1, 0, 1]:
                    self.create_strided_view(o3, o2, o1, is_ghost=False)

        # Ghost zones
        for o3 in [-1, 0, 1]:
            for o2 in [-1, 0, 1]:
                for o1 in [-1, 0, 1]:
                    self.create_strided_view(o3, o2, o1, is_ghost=True)

        return self

    def create_strided_view(self, o3, o2, o1, is_ghost):
        """Create a strided view of the mesh block."""
        nghost = self.size.nghost
        start1, len1 = (nghost * (1 - is_ghost), nghost) if o1 == -1 \
            else (nghost, self.size.nx1) if o1 == 0 \
            else (nghost * is_ghost + self.size.nx1, nghost)

        nghost = 0 if self.size.nx2 == 1 else self.size.nghost
        start2, len2 = (nghost * (1 - is_ghost), nghost) if o2 == -1 \
            else (nghost, self.size.nx2) if o2 == 0 \
            else (nghost * is_ghost + self.size.nx2, nghost)

        nghost = 0 if self.size.nx3 == 1 else self.size.nghost
        start3, len3 = (nghost * (1 - is_ghost), nghost) if o3 == -1 \
            else (nghost, self.size.nx3) if o3 == 0 \
            else (nghost * is_ghost + self.size.nx3, nghost)

        target_array = self.ghost if is_ghost else self.view

        byte = self.data.itemsize
        target_array[(o3, o2, o1)] = as_strided(
            self.data[start3:, start2:, start1:, :],
            shape=(len3, len2, len1, self.size.nvar),
            strides=(self.size.nc2 * self.size.nc1 * self.size.nvar * byte,
                     self.size.nc1 * self.size.nvar * byte, self.size.nvar * byte, byte),
            writeable=is_ghost
        )

    def fill_random(self) -> Self:
        """Fill interior zones with random values."""
        self.data.fill(-1)
        #np.random.seed(seed) # remove me after testing
        self.ghost[(0, 0, 0)][:] = np.random.uniform(0, 1, size=(
            self.size.nx3, self.size.nx2, self.size.nx1, self.size.nvar))
        return self


    def prolongated_view(self, my_offsets: (int, int, int),
                         finer: Coordinates) -> np.ndarray:
        """Prolongate a view to a finer mesh block."""
        # Extract finer block coordinates
        finer_offsets = tuple(-x for x in my_offsets)
        si, ei, sj, ej, sk, ek = self.size.ghost_range(finer_offsets)
        gx1v, gx2v, gx3v = finer.x1v[si:ei], finer.x2v[sj:ej], finer.x3v[sk:ek]

        # Create coordinates for interpolation
        gcoord = np.array(np.meshgrid(gx3v, gx2v, gx1v)).T.reshape(-1, 3)
        points = (self.coord.x3v, self.coord.x2v, self.coord.x1v)

        # Interpolate data to finer ghoze zone
        view = np.zeros((ek - sk, ej - sj, ei - si, self.size.nvar))
        for i in range(self.size.nvar):
            view[:, :, :, i] = interpn(points, self.data[:, :, :, i],
                                       gcoord, method='linear',
                                       bounds_error=True).reshape(ek - sk, ej - sj, ei - si)
        return view

    def restricted_view(self, my_offsets: (int, int, int),
                        coarser: Coordinates) -> np.ndarray:
        """Restrict a view to a coarser mesh block."""
        # Extract coordinates for the coarser mesh block from the finer
        coarser_offsets = tuple(-x for x in my_offsets)
        si, ei, sj, ej, sk, ek = self.size.ghost_range(coarser_offsets)
        gx1v, gx2v, gx3v = coarser.x1v[si:ei], coarser.x2v[sj:ej], coarser.x3v[sk:ek]

        # Find indices for self interior points
        isi, iei, isj, iej, isk, iek = self.size.ghost_range((0, 0, 0))

        # Find the assemble_ of coarser mesh block that is inside the self mesh block
        si = bisect(gx1v, self.coord.x1v[isi])
        ei = bisect(gx1v, self.coord.x1v[iei - 1])

        sj = bisect(gx2v, self.coord.x2v[isj]) if self.size.nx2 > 1 else 0
        ej = bisect(gx2v, self.coord.x2v[iej - 1]) if self.size.nx2 > 1 else 1

        sk = bisect(gx3v, self.coord.x3v[isk]) if self.size.nx3 > 1 else 0
        ek = bisect(gx3v, self.coord.x3v[iek - 1]) if self.size.nx3 > 1 else 1

        gcoord = np.array(np.meshgrid(
            gx3v[sk:ek], gx2v[sj:ej], gx1v[si:ei])).T.reshape(-1, 3)
        points = (self.coord.x3v[isk:iek],
                  self.coord.x2v[isj:iej], self.coord.x1v[isi:iei])

        # Interpolate data to coarser mesh
        view = np.zeros((ek - sk, ej - sj, ei - si, self.size.nvar))
        for i in range(self.size.nvar):
            view[:, :, :, i] = interpn(points, self.view[(0, 0, 0)][:, :, :, i],
                                       gcoord, method='linear',
                                       bounds_error=True).reshape(ek - sk, ej - sj, ei - si)
        return view

    def part(self, offsets: (int, int, int), logicloc: (int, int, int)) -> np.ndarray:
        """Extract a part of the ghost zone during restriction."""
        si, ei, sj, ej, sk, ek = self.size.ghost_range(offsets)
        o3, o2, o1 = offsets
        f3, f2, f1 = logicloc[0] % 2, logicloc[1] % 2, logicloc[2] % 2

        len1, len2, len3 = ei - si, ej - sj, ek - sk
        ng1 = min(self.size.nx1 // 2, self.size.nghost)
        ng2 = min(self.size.nx2 // 2, self.size.nghost)
        ng3 = min(self.size.nx3 // 2, self.size.nghost)

        if f1 != 0:
            si = ei - ng1 if o1 != 0 else ei - len1 // 2
        else:
            ei = si + ng1 if o1 != 0 else si + len1 // 2

        if self.size.nx2 > 1:
            if f2 != 0:
                sj = ej - ng2 if o2 != 0 else ej - len2 // 2
            else:
                ej = sj + ng2 if o2 != 0 else sj + len2 // 2

        if self.size.nx3 > 1:
            if f3 != 0:
                sk = ek - ng3 if o3 != 0 else ek - len3 // 2
            else:
                ek = sk + ng3 if o3 != 0 else sk + len3 // 2

        byte = self.data.itemsize
        view = as_strided(
            self.data[sk:, sj:, si:, :],
            shape=(ek - sk, ej - sj, ei - si, self.size.nvar),
            strides=(self.size.nc2 * self.size.nc1 * self.size.nvar * byte,
                     self.size.nc1 * self.size.nvar * byte, self.size.nvar * byte, byte),
            writeable=True
        )
        return view

    def print_data(self) -> None:
        """Print the data in the mesh block."""
        for n in range(self.size.nvar):
            print(f"var {n} = ")
            for k, z_value in enumerate(self.coord.x3v):
                print(f"(z = {z_value:.2f})")
                for j in range(self.size.nc2):
                    for i in range(self.size.nc1):
                        print(f"{self.data[k, j, i, n]:.3f}, ", end="")
                    print()
                if k < self.size.nc3 - 1:
                    print()

    def __str__(self) -> str:
        """Return a string representation of the mesh block."""
        if self.data is None:
            return f"{self.size}\n{self.coord}\ndata=None"
        return f"{self.size}\n{self.coord}\ndata_shape={self.data.shape}"

    def __eq__(self, other: Any) -> bool:
        """Check if two MeshBlock instances are equal."""
        if isinstance(other, MeshBlock):
            return self.coord == other.coord and self.data == other.data
        return False


if __name__ == "__main__":
    # test 1
    size = RegionSize(x1dim=(0, 1, 2), x2dim=(0, 1, 4), nghost=2)
    mb = MeshBlock(size)
    mb.allocate()
    print(mb)

    # test 2
    mb = MeshBlock(size, "cylindrical")
    mb.allocate(2)
    print(mb)

    # test 3
    size1 = RegionSize(x1dim=(0, 40., 4), x2dim=(0, 40., 2))
    mb1 = MeshBlock(size1)
    mb1.allocate(2).fill_random()
    print("===== mb1 =====")
    mb1.print_data()
    print(mb1.view[(0, 0, 1)][0, :, :, 0])
    print(mb1.view[(0, 0, 1)][0, :, :, 1])
    print(mb1.view[(0, -1, -1)][0, :, :, 0])
    print(mb1.view[(0, -1, -1)][0, :, :, 1])
    print("===============")

    size2 = RegionSize(x1dim=(40., 120., 4), x2dim=(0, 40., 2))
    mb2 = MeshBlock(size2)
    mb2.allocate(2).fill_random()
    print("===== mb2 =====")
    mb2.print_data()
    print(mb2.view[(0, 1, 0)][0, :, :, 0])
    print(mb2.view[(0, 1, 0)][0, :, :, 1])
    print(mb2.view[(0, -1, 1)][0, :, :, 0])
    print(mb2.view[(0, -1, 1)][0, :, :, 1])
    print("===============")

    size3 = RegionSize(x1dim=(0., 40., 4), x2dim=(40, 120., 2))
    mb3 = MeshBlock(size3)
    mb3.allocate(2).fill_random()
    print("===== mb3 =====")
    mb3.print_data()
    print(mb3.view[(0, 1, -1)][0, :, :, 0])
    print(mb3.view[(0, 1, -1)][0, :, :, 1])
    print("===============")

    size4 = RegionSize(x1dim=(40., 120., 4), x2dim=(40, 120., 2))
    mb4 = MeshBlock(size4)
    mb4.allocate(2).fill_random()
    print("===== mb4 =====")
    mb4.print_data()
    print(mb4.view[(0, 1, 1)][0, :, :, 0])
    print(mb4.view[(0, 1, 1)][0, :, :, 1])
    print("===============")

    # test 4 (prolongation)
    print("==== Test Prolongation ====")
    from .tree import Tree
    Tree.set_block_size(nx1=2, nx2=2, nx3=1)
    size = RegionSize(x1dim=(0, 120., 8), x2dim=(0, 120., 8))
    root = Tree(size)
    root.create_tree()
    root.print_tree()
    coarse_node = root.children[3].children[0]
    print(f"coarse_node = {coarse_node}")
    coarse_block = MeshBlock(coarse_node.size)
    coarse_block.allocate().fill_random()
    coarse_block.print_data()
    print("\n===== split block =====")
    root.children[3].children[1].split_block()
    fine_node = root.children[3].children[1].children[0]
    print(f"fine_node = {fine_node}")
    fine_block = MeshBlock(fine_node.size)
    fine_block.allocate().fill_random()
    fine_block.print_data()

    view = coarse_block.prolongated_view((0, 0, 1), fine_block.coord)
    print(view.shape)
    print(view)

    # test 5 (restriction)
    print("==== Test Restriction ====")
    view = fine_block.restricted_view((0, 0, -1), coarse_block.coord)
    print(view.shape)
    print(view)
