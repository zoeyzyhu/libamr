# pylint: disable = import-error, too-many-instance-attributes, too-many-branches, too-many-statements, fixme
"""Module containing the MeshBlock class."""

from typing import Any
from typing_extensions import Self
import numpy as np
from numpy.lib.stride_tricks import as_strided
from .region_size import RegionSize
from .coordinate_factory import CoordinateFactory


class MeshBlock:
    """A class representing a mesh block."""

    def __init__(self, size: RegionSize, coordinate_type: str, nghost: int = 0):
        """Initialize MeshBlock with size, coordinate type, and optional ghost zones."""
        self.coord = CoordinateFactory.create(size, coordinate_type, nghost)
        self.nvar = 0
        self.data = None
        self.view = {}
        self.ghost = {}

        self.nx3 = size.nx3
        self.nx2 = size.nx2
        self.nx1 = size.nx1

        if size.nx3 > 1:
            self.nc3 = size.nx3 + 2 * nghost
        else:
            self.nc3 = 1

        if size.nx2 > 1:
            self.nc2 = size.nx2 + 2 * nghost
        else:
            self.nc2 = 1

        self.nc1 = size.nx1 + 2 * nghost

    def allocate(self, nvar: int = 1) -> Self:
        """Allocate memory for the mesh block."""
        self.nvar = int(nvar)
        if self.nvar < 1:
            raise ValueError("nvar must be >= 1")

        self.data = np.zeros((self.nc3, self.nc2, self.nc1, self.nvar))

        # interior slices
        for o3 in [-1, 0, 1]:
            for o2 in [-1, 0, 1]:
                for o1 in [-1, 0, 1]:
                    nghost = self.coord.nghost
                    if o1 == -1:
                        offset1, len1 = nghost, nghost
                    elif o1 == 0:
                        offset1, len1 = nghost, self.nx1
                    else:
                        offset1, len1 = self.nx1, nghost

                    if self.nx2 == 1:
                        nghost = 0
                    else:
                        nghost = self.coord.nghost

                    if o2 == -1:
                        offset2, len2 = nghost, nghost
                    elif o2 == 0:
                        offset2, len2 = nghost, self.nx2
                    else:
                        offset2, len2 = self.nx2, nghost

                    if self.nx3 == 1:
                        nghost = 0
                    else:
                        nghost = self.coord.nghost

                    if o3 == -1:
                        offset3, len3 = nghost, nghost
                    elif o3 == 0:
                        offset3, len3 = nghost, self.nx3
                    else:
                        offset3, len3 = self.nx3, nghost

                    byte = self.data.itemsize

                    self.view[(o3, o2, o1)] = as_strided(
                        self.data[offset3: offset3 + len3,
                                  offset2: offset2 + len2,
                                  offset1: offset1 + len1,
                                  0: self.nvar],
                        shape=(len3, len2, len1, self.nvar),
                        strides=(self.nc2 * self.nc1 * self.nvar * byte, self.nc1 *
                                 self.nvar * byte, self.nvar * byte, byte),
                        writeable=False
                    )

        # ghost zones
        for o3 in [-1, 0, 1]:
            for o2 in [-1, 0, 1]:
                for o1 in [-1, 0, 1]:
                    nghost = self.coord.nghost
                    if o1 == -1:
                        offset1, len1 = 0, nghost
                    elif o1 == 0:
                        offset1, len1 = nghost, self.nx1
                    else:
                        offset1, len1 = nghost + self.nx1, nghost

                    if self.nx2 == 1:
                        nghost = 0
                    else:
                        nghost = self.coord.nghost

                    if o2 == -1:
                        offset2, len2 = 0, nghost
                    elif o2 == 0:
                        offset2, len2 = nghost, self.nx2
                    else:
                        offset2, len2 = nghost + self.nx2, nghost

                    if self.nx3 == 1:
                        nghost = 0
                    else:
                        nghost = self.coord.nghost

                    if o3 == -1:
                        offset3, len3 = 0, nghost
                    elif o3 == 0:
                        offset3, len3 = nghost, self.nx3
                    else:
                        offset3, len3 = nghost + self.nx3, nghost

                    byte = self.data.itemsize

                    self.ghost[(o3, o2, o1)] = as_strided(
                        self.data[offset3: offset3 + len3,
                                  offset2: offset2 + len2,
                                  offset1: offset1 + len1,
                                  0: self.nvar],
                        shape=(len3, len2, len1, self.nvar),
                        strides=(self.nc2 * self.nc1 * self.nvar * byte,
                                 self.nc1 * self.nvar * byte, self.nvar * byte, byte)
                    )
        return self

    def part(self, cubic_offset: (int, int, int), of1: int, of2: int) -> np.ndarray:
        """Extract a part of the mesh block."""

    def fill_random(self) -> Self:
        """Fill ghost zones with random values."""
        # set ghost zones to zero
        self.data[:] = -np.ones(self.data.shape)
        self.ghost[(0, 0, 0)][:] = \
            np.random.uniform(0, 1, size=(
                self.nx3, self.nx2, self.nx1, self.nvar))

        return self

    def print_data(self) -> None:
        """Print the data in the mesh block."""
        for n in range(self.nvar):
            print(f"var {n} = ")
            for k in range(self.nc3):
                print(f"(z = {self.coord.x3v[k]:.2f})")
                for j in range(self.nc2):
                    for i in range(self.nc1):
                        print(f"{self.data[k, j, i, n]:.3f}, ", end="")
                    print()
                if k < self.nc3 - 1:
                    print()

    # TODO: implement prolongation
    def prolongated_view(self, cubic_offset: (int, int, int),
                         ox1: int, ox2: int, ox3: int) -> np.ndarray:
        """Prolongate a view to a finer mesh block."""

    # TODO: implement restriction
    def restrict_view(self, cubic_offset: (int, int, int),
                      ox1: int, ox2: int, ox3: int) -> np.ndarray:
        """Restrict a view to a coarser mesh block."""

    def __str__(self) -> str:
        """Return a string representation of the mesh block."""
        if self.data is None:
            return f"{self.coord}\ndata=None"
        return f"{self.coord}\ndata_shape={self.data.shape}"

    def __eq__(self, other: Any) -> bool:
        """Check if two MeshBlock instances are equal."""
        if isinstance(other, MeshBlock):
            return self.coord == other.coord and self.data == other.data
        return False


if __name__ == "__main__":
    # test 1
    rs = RegionSize(x1dim=(0, 1, 2), x2dim=(0, 1, 4))
    mb = MeshBlock(rs, "cartesian")
    mb.allocate()
    print(mb)

    # test 2
    mb = MeshBlock(rs, "cylindrical", nghost=2)
    mb.allocate(2)
    print(mb)

    # test 3
    rs1 = RegionSize(x1dim=(0, 40., 4), x2dim=(0, 40., 2))
    mb1 = MeshBlock(rs1, "cartesian", nghost=1)
    mb1.allocate(2).fill_random()
    print("===== mb1 =====")
    mb1.print_data()
    print(mb1.view[(0, 0, 1)][0, :, :, 0])
    print(mb1.view[(0, 0, 1)][0, :, :, 1])
    print(mb1.view[(0, -1, -1)][0, :, :, 0])
    print(mb1.view[(0, -1, -1)][0, :, :, 1])
    print("===============")

    rs2 = RegionSize(x1dim=(40., 120., 4), x2dim=(0, 40., 2))
    mb2 = MeshBlock(rs2, "cartesian", nghost=1)
    mb2.allocate(2).fill_random()
    print("===== mb2 =====")
    mb2.print_data()
    print(mb2.view[(0, 1, 0)][0, :, :, 0])
    print(mb2.view[(0, 1, 0)][0, :, :, 1])
    print(mb2.view[(0, -1, 1)][0, :, :, 0])
    print(mb2.view[(0, -1, 1)][0, :, :, 1])
    print("===============")

    rs3 = RegionSize(x1dim=(0., 40., 4), x2dim=(40, 120., 2))
    mb3 = MeshBlock(rs3, "cartesian", nghost=1)
    mb3.allocate(2).fill_random()
    print("===== mb3 =====")
    mb3.print_data()
    print(mb3.view[(0, 1, -1)][0, :, :, 0])
    print(mb3.view[(0, 1, -1)][0, :, :, 1])
    print("===============")

    rs4 = RegionSize(x1dim=(40., 120., 4), x2dim=(40, 120., 2))
    mb4 = MeshBlock(rs4, "cartesian", nghost=1)
    mb4.allocate(2).fill_random()
    print("===== mb4 =====")
    mb4.print_data()
    print(mb4.view[(0, 1, 1)][0, :, :, 0])
    print(mb4.view[(0, 1, 1)][0, :, :, 1])
    print("===============")
