from region_size import RegionSize
from coordinates import Cartesian, Cylindrical

class CoordinateFactory:
    @staticmethod
    def create(size: RegionSize, coordinate_type: str, nghost: int = 0):
        if coordinate_type == "cartesian":
            coord = Cartesian(size, nghost)
        elif coordinate_type == "cylindrical":
            coord = Cylindrical(size, nghost)
        else:
            raise ValueError(f"Invalid coordinate type: {coordinate_type}")
        return coord
