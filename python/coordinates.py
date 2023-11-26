# pylint: disable = import-error, pointless-string-statement, redefined-outer-name
"""Module containing the Coordinates, Cartesian, and Cylindrical classes."""

import numpy as np
from region_size import RegionSize

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

    def __init__(self, rs: RegionSize, nghost: int):
        """Initialize coordinates with region size and ghost zones."""
        self.nghost = nghost

        dx1 = (rs.x1max - rs.x1min) / rs.nx1
        x1min = rs.x1min - nghost * dx1
        x1max = rs.x1max + nghost * dx1
        self.x1f = np.linspace(x1min, x1max, num=rs.nx1 + 1 + 2 * nghost)

        if rs.nx2 == 1:
            nghost = 0
        else:
            nghost = self.nghost
        dx2 = (rs.x2max - rs.x2min) / rs.nx2
        x2min = rs.x2min - nghost * dx2
        x2max = rs.x2max + nghost * dx2
        self.x2f = np.linspace(x2min, x2max, num=rs.nx2 + 1 + 2 * nghost)

        if rs.nx3 == 1:
            nghost = 0
        else:
            nghost = self.nghost
        dx3 = (rs.x3max - rs.x3min) / rs.nx3
        x3min = rs.x3min - nghost * dx3
        x3max = rs.x3max + nghost * dx3
        self.x3f = np.linspace(x3min, x3max, num=rs.nx3 + 1 + 2 * nghost)

    def __str__(self) -> str:
        """Return a string representation of the coordinates."""
        mystr = "Coordinates:\n"
        x1f_str = " ".join(f"{x:.2f}" for x in self.x1f)
        x2f_str = " ".join(f"{x:.2f}" for x in self.x2f)
        x3f_str = " ".join(f"{x:.2f}" for x in self.x3f)
        mystr += f"x1f=[{x1f_str}]\nx2f=[{x2f_str}]\nx3f=[{x3f_str}]\n" + \
                 f"nghost={self.nghost}"
        return mystr

    def __eq__(self, other) -> bool:
        """Check if two Coordinates instances are equal."""
        if isinstance(other, Coordinates):
            return self.x1f == other.x1f and self.x2f == other.x2f and \
                self.x3f == other.x3f and self.nghost == other.nghost
        return False


class Cartesian(Coordinates):
    """Class representing Cartesian coordinates."""

    def __init__(self, rs: RegionSize, nghost: int = 0):
        """Initialize Cartesian coordinates."""
        super().__init__(rs, nghost)
        self.x1v = (self.x1f[1:] + self.x1f[:-1]) / 2.0
        self.x2v = (self.x2f[1:] + self.x2f[:-1]) / 2.0
        self.x3v = (self.x3f[1:] + self.x3f[:-1]) / 2.0

    def __str__(self) -> str:
        """Return a string representation of the Cartesian coordinates."""
        mystr = super().__str__()
        x1v_str = " ".join(f"{x:.2f}" for x in self.x1v)
        x2v_str = " ".join(f"{x:.2f}" for x in self.x2v)
        x3v_str = " ".join(f"{x:.2f}" for x in self.x3v)

        mystr += f"\nx1v=[{x1v_str}]\nx2v=[{x2v_str}]\nx3v=[{x3v_str}]"
        return f"{mystr}"

    def __eq__(self, other) -> bool:
        """Check if two Cartesian instances are equal."""
        if isinstance(other, Cartesian):
            return super().__eq__(other) and self.x1v == other.x1v and \
                self.x2v == other.x2v and self.x3v == other.x3v
        return False


class Cylindrical(Coordinates):
    """Class representing cylindrical coordinates."""

    def __init__(self, rs: RegionSize, nghost: int = 0):
        """Initialize cylindrical coordinates."""
        super().__init__(rs, nghost)
        if rs.x1min < 0:
            raise ValueError(
                "x1min (minimum radius) must be >= 0 for cylindrical coordinates")

        self.x1v = 2. / 3. * (pow(self.x1f[1:], 3) - pow(self.x1f[:-1], 3)) / (
            pow(self.x1f[1:], 2) - pow(self.x1f[:-1], 2))
        self.x2v = (self.x2f[1:] + self.x2f[:-1]) / 2.0
        self.x3v = (self.x3f[1:] + self.x3f[:-1]) / 2.0

    def __str__(self) -> str:
        """Return a string representation of the cylindrical coordinates."""
        mystr = super().__str__()
        x1v_str = " ".join(f"{x:.2f}" for x in self.x1v)
        x2v_str = " ".join(f"{x:.2f}" for x in self.x2v)
        x3v_str = " ".join(f"{x:.2f}" for x in self.x3v)
        mystr += f"\nx1v=[{x1v_str}]\nx2v=[{x2v_str}]\nx3v=[{x3v_str}]"
        return f"{mystr}"

    def __eq__(self, other) -> bool:
        """Check if two Cylindrical instances are equal."""
        if isinstance(other, Cylindrical):
            return super().__eq__(other) and self.x1v == other.x1v and \
                self.x2v == other.x2v and self.x3v == other.x3v
        return False


if __name__ == "__main__":
    rs = RegionSize((-1.0, 1.0, 5))
    coords = Cartesian(rs, nghost=2)
    print(coords)

    rs.x1min = 0.0
    coords = Cylindrical(rs, nghost=2)
    print(coords)
