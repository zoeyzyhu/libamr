# pylint: disable = too-many-instance-attributes, too-many-arguments
"""This module contains the RegionSize class."""


class RegionSize:
    """A class for representing the size of a multidimensional block."""

    def __init__(self, x1dim: (float, float, int),
                 x2dim: (float, float, int) = (-0.5, 0.5, 1),
                 x3dim: (float, float, int) = (-0.5, 0.5, 1),
                 nghost: int = 1, nvar: int = 1):
        """Initialize RegionSize with x1, x2, and x3 dimensions."""
        self.nghost = nghost
        self.nvar = nvar

        self.x1min, self.x1max, self.nx1 = map(float, x1dim)
        self.x2min, self.x2max, self.nx2 = map(float, x2dim)
        self.x3min, self.x3max, self.nx3 = map(float, x3dim)
        self.nx1, self.nx2, self.nx3 = map(int, [self.nx1, self.nx2, self.nx3])

        self.nc3 = self.nx3 + 2 * self.nghost if self.nx3 > 1 else 1
        self.nc2 = self.nx2 + 2 * self.nghost if self.nx2 > 1 else 1
        self.nc1 = self.nx1 + 2 * self.nghost

        for dim, name in [(self.nx1, 'x1'), (self.nx2, 'x2'), (self.nx3, 'x3')]:
            if dim < 1:
                raise ValueError(f"{name} dimension (n{name}) must be >= 1")

        for dmin, dmax, name in [(self.x1min, self.x1max, 'x1'),
                                 (self.x2min, self.x2max, 'x2'),
                                 (self.x3min, self.x3max, 'x3')]:
            if dmin > dmax:
                raise ValueError(f"{name}min = {dmin} > {name}max = {dmax}")

    def center(self) -> tuple[float]:
        """Return the center of the region."""
        return ((self.x3max + self.x3min) / 2.,
                (self.x2max + self.x2min) / 2.,
                (self.x1max + self.x1min) / 2.)

    def ghost_range(self, offsets: (int, int, int)) -> tuple[int]:
        """Return the range of ghost zone specified by the cubic offsets."""
        def get_range(nx, nc, offset):
            if offset == -1:
                start, end = 0, self.nghost
            elif offset == 0:
                start, end = self.nghost, nx + self.nghost
            else:
                start, end = nx + self.nghost, nc
            return start, end

        o3, o2, o1 = offsets
        si, ei = get_range(self.nx1, self.nc1, o1)

        if self.nx2 > 1:
            sj, ej = get_range(self.nx2, self.nc2, o2)
        elif o2 == 0:
            sj, ej = 0, 1
        else:
            sj, ej = 0, 0

        if self.nx3 > 1:
            sk, ek = get_range(self.nx3, self.nc3, o3)
        elif o3 == 0:
            sk, ek = 0, 1
        else:
            sk, ek = 0, 0

        return si, ei, sj, ej, sk, ek

    def __str__(self) -> str:
        """Return a string representation of the region size."""
        dimensions = [
            f"[{getattr(self, f'x{i}min')}, {getattr(self, f'x{i}max')}]"
            for i in range(1, 4)
        ]
        dim_str = f"{dimensions[0]} x {dimensions[1]} x {dimensions[2]}, " \
            f"nx1 ({self.nx1}) x nx2 ({self.nx2}) x nx3 ({self.nx3}), "
        if self.nx3 == 1:
            dim_str = f"{dimensions[0]} x {dimensions[1]}, " \
                f"nx1 ({self.nx1}) x nx2 ({self.nx2}), "
            if self.nx2 == 1:
                dim_str = f"{dimensions[0]}, nx1 ({self.nx1}), "
        dim_str += f"nvar = {self.nvar}, nghost = {self.nghost}"
        return dim_str

    def __eq__(self, other) -> bool:
        """Check if two RegionSize instances are equal."""
        if isinstance(other, RegionSize):
            return self.x1min == other.x1min and \
                self.x1max == other.x1max and self.nx1 == other.nx1 and \
                self.x2min == other.x2min and \
                self.x2max == other.x2max and self.nx2 == other.nx2 and \
                self.x3min == other.x3min and \
                self.x3max == other.x3max and self.nx3 == other.nx3 and \
                self.nghost == other.nghost and self.nvar == other.nvar
        return False


if __name__ == "__main__":
    region_size1 = RegionSize((0, 1, 2))
    print(region_size1)

    region_size2 = RegionSize((0, 1, 2), (0, 1, 2))
    print(region_size2)

    region_size3 = RegionSize(
        x1dim=(0, 1, 2), x2dim=(0, 1, 2), x3dim=(-2, 3, 3))
    print(region_size3)

    region_size4 = RegionSize(
        x1dim=(0, 1, 2), x3dim=(-2, 3, 3), x2dim=(0, 1, 2))
    print(region_size4)

    print(region_size2 == region_size3)
    print(region_size4 == region_size3)
