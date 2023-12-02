# pylint: disable = import-error, too-few-public-methods
"""Module containing the CoordinateFactory class."""

from .region_size import RegionSize
from .coordinates import Cartesian, Cylindrical


class CoordinateFactory:
    """A factory class for creating coordinate objects."""

    @staticmethod
    def create(size: RegionSize, coordinate_type: str = "cartesian"):
        """Create a coordinate object based on the specified type."""
        if coordinate_type == "cartesian":
            coord = Cartesian(size)
        elif coordinate_type == "cylindrical":
            coord = Cylindrical(size)
        else:
            raise ValueError(f"Invalid coordinate type: {coordinate_type}")
        return coord
