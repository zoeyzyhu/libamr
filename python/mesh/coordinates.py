# pylint: disable = import-error, pointless-string-statement, redefined-outer-name
"""Module containing the Coordinates, Cartesian, and Cylindrical classes."""

import numpy as np
from .region_size import RegionSize

"""
This is the staggering of a coordinate axis
+ : cell center
| : cell face
----|-----+------|-----+------|----+------|-----+------|-----+------|---
----|--x1v(i-2)--|--x1v(i-1)--|--x1v(i)---|--x1v(i+1)--|--x1v(i+2)--|---
----|-----+------|-----+------|----+------|-----+------|-----+------|---
x1f(i-2)--+--x1f(i-1)--+--x1f(i)---+--x1f(i+i)--+--x1f(i+2)--+--x1f(i+3)
----|-----+------|-----+------|----+------|-----+------|-----+------|---
"""


class Coordinates:
    """Base class for coordinate representation."""

    def __init__(self, size: RegionSize, nghost: int):
        """Initialize coordinates with region size and ghost zones."""
        def generate_coordinate_array(xmin, xmax, nx):
            delta = (xmax - xmin) / nx
            extended_min = xmin - nghost * delta
            extended_max = xmax + nghost * delta
            return np.linspace(extended_min, extended_max,
                               num=nx + 1 + 2 * nghost)

        self.nghost = nghost
        self.x1f = generate_coordinate_array(size.x1min, size.x1max, size.nx1)
        nghost = 0 if size.nx2 == 1 else self.nghost
        self.x2f = generate_coordinate_array(size.x2min, size.x2max, size.nx2)
        nghost = 0 if size.nx3 == 1 else self.nghost
        self.x3f = generate_coordinate_array(size.x3min, size.x3max, size.nx3)

    def __str__(self) -> str:
        """Return a string representation of the coordinates."""
        x1f_str = ", ".join(f"{x:.2f}" for x in self.x1f)
        x2f_str = ", ".join(f"{x:.2f}" for x in self.x2f)
        x3f_str = ", ".join(f"{x:.2f}" for x in self.x3f)
        return f"Coordinates:\n" \
            f"x1f=[{x1f_str}]\n" \
            f"x2f=[{x2f_str}]\n" \
            f"x3f=[{x3f_str}]\n" \
            f"nghost={self.nghost}"

    def __eq__(self, other) -> bool:
        """Check if two Coordinates instances are equal."""
        if isinstance(other, Coordinates):
            return np.array_equal(self.x1f, other.x1f) and \
                np.array_equal(self.x2f, other.x2f) and \
                np.array_equal(self.x3f, other.x3f) and \
                self.nghost == other.nghost
        return False


class Cartesian(Coordinates):
    """Class representing Cartesian coordinates."""

    def __init__(self, size: RegionSize, nghost: int = 0):
        """Initialize Cartesian coordinates."""
        super().__init__(size, nghost)
        self.x1v = (self.x1f[1:] + self.x1f[:-1]) / 2.0
        self.x2v = (self.x2f[1:] + self.x2f[:-1]) / 2.0
        self.x3v = (self.x3f[1:] + self.x3f[:-1]) / 2.0

    def __str__(self) -> str:
        """Return a string representation of the Cartesian coordinates."""
        coordinates_str = super().__str__()
        x1v_str = ", ".join(f"{x:.2f}" for x in self.x1v)
        x2v_str = ", ".join(f"{x:.2f}" for x in self.x2v)
        x3v_str = ", ".join(f"{x:.2f}" for x in self.x3v)

        return f"{coordinates_str}\n" \
            f"x1v=[{x1v_str}]\n" \
            f"x2v=[{x2v_str}]\n" \
            f"x3v=[{x3v_str}]"

    def __eq__(self, other) -> bool:
        """Check if two Cartesian instances are equal."""
        if isinstance(other, Cartesian):
            return super().__eq__(other) and \
                np.array_equal(self.x1v, other.x1v) and \
                np.array_equal(self.x2v, other.x2v) and \
                np.array_equal(self.x3v, other.x3v)
        return False


class Cylindrical(Coordinates):
    """Class representing cylindrical coordinates."""

    def __init__(self, size: RegionSize, nghost: int = 0):
        """Initialize cylindrical coordinates."""
        super().__init__(size, nghost)
        if size.x1min < 0:
            raise ValueError(
                "x1min (minimum radius) must be >= 0 for cylindrical coordinates")

        self.x1v = 2. / 3. * (pow(self.x1f[1:], 3) - pow(self.x1f[:-1], 3)) / (
            pow(self.x1f[1:], 2) - pow(self.x1f[:-1], 2))
        self.x2v = (self.x2f[1:] + self.x2f[:-1]) / 2.0
        self.x3v = (self.x3f[1:] + self.x3f[:-1]) / 2.0

    def __str__(self) -> str:
        """Return a string representation of the Cartesian coordinates."""
        coordinates_str = super().__str__()
        x1v_str = ", ".join(f"{x:.2f}" for x in self.x1v)
        x2v_str = ", ".join(f"{x:.2f}" for x in self.x2v)
        x3v_str = ", ".join(f"{x:.2f}" for x in self.x3v)

        return f"{coordinates_str}\n" \
            f"x1v=[{x1v_str}]\n" \
            f"x2v=[{x2v_str}]\n" \
            f"x3v=[{x3v_str}]"

    def __eq__(self, other) -> bool:
        """Check if two Cartesian instances are equal."""
        if isinstance(other, Cartesian):
            return super().__eq__(other) and \
                np.array_equal(self.x1v, other.x1v) and \
                np.array_equal(self.x2v, other.x2v) and \
                np.array_equal(self.x3v, other.x3v)
        return False


if __name__ == "__main__":
    size = RegionSize((-1.0, 1.0, 5))
    coords = Cartesian(size, nghost=2)
    print(coords)

    size.x1min = 0.0
    coords = Cylindrical(size, nghost=2)
    print(coords)
