# pylint: disable = too-many-instance-attributes
"""This module contains the RegionSize class."""


class RegionSize:
    """A class for representing the size of a region."""

    def __init__(self, x1dim: (float, float, int),
                 x2dim: (float, float, int) = (-0.5, 0.5, 1),
                 x3dim: (float, float, int) = (-0.5, 0.5, 1)):
        """Initialize RegionSize with x1, x2, and x3 dimensions."""
        self.x1min, self.x1max, self.nx1 = float(
            x1dim[0]), float(x1dim[1]), int(x1dim[2])
        if self.x1min > self.x1max:
            raise ValueError(f"x1min = {self.x1min} > x1max = {self.x1max}")
        if self.nx1 < 1:
            raise ValueError(f"nx1 = {self.nx1} < 1")

        self.x2min, self.x2max, self.nx2 = float(
            x2dim[0]), float(x2dim[1]), int(x2dim[2])
        if self.x2min > self.x2max:
            raise ValueError(f"x2min = {self.x2min} > x2max = {self.x2max}")
        if self.nx2 < 1:
            raise ValueError(f"nx2 = {self.nx2} < 1")

        self.x3min, self.x3max, self.nx3 = float(
            x3dim[0]), float(x3dim[1]), int(x3dim[2])
        if self.x3min > self.x3max:
            raise ValueError(f"x3min = {self.x3min} > x3max = {self.x3max}")
        if self.nx3 < 1:
            raise ValueError(f"nx3 = {self.nx3} < 1")

    def __str__(self) -> str:
        """Return a string representation of the region size."""
        if self.nx3 == 1:
            if self.nx2 == 1:
                return f"[{self.x1min}, {self.x1max}], nx1 = {self.nx1}"
            return f"[{self.x1min}, {self.x1max}] x [{self.x2min}, " + \
                f"{self.x2max}], nx1 = {self.nx1}, nx2 = {self.nx2}"
        return f"[{self.x1min}, {self.x1max}] x [{self.x2min}, " + \
            f"{self.x2max}] x [{self.x3min}, {self.x3max}], " + \
            f"nx1 = {self.nx1}, nx2 = {self.nx2}, nx3 = {self.nx3}"

    def __eq__(self, other) -> bool:
        """Check if two RegionSize instances are equal."""
        if isinstance(other, RegionSize):
            return self.x1min == other.x1min and \
                self.x1max == other.x1max and self.nx1 == other.nx1 and \
                self.x2min == other.x2min and \
                self.x2max == other.x2max and self.nx2 == other.nx2 and \
                self.x3min == other.x3min and \
                self.x3max == other.x3max and self.nx3 == other.nx3
        return False


if __name__ == "__main__":
    region_size = RegionSize((0, 1, 2))
    print(region_size)

    region_size = RegionSize((0, 1, 2), (0, 1, 2))
    print(region_size)

    region_size = RegionSize(
        x1dim=(0, 1, 2), x3dim=(-2, 3, 3), x2dim=(0, 1, 2))
    print(region_size)
