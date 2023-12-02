"""Test the coordinate systems"""

import sys
sys.path.append('../')
import mesh as me


if __name__ == "__main__":
    size = me.RegionSize((-1.0, 1.0, 5), nghost=2)
    coords = me.Cartesian(size)
    print(coords)

    size.x1min = 0.0
    coords = me.Cylindrical(size)
    print(coords)