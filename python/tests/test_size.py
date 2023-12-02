
import sys
sys.path.append('../')
import mesh as me

if __name__ == "__main__":
    region_size1 = me.RegionSize((0, 1, 2))
    print(region_size1)

    region_size2 = me.RegionSize((0, 1, 2), (0, 1, 2))
    print(region_size2)

    region_size3 = me.RegionSize(
        x1dim=(0, 1, 2), x2dim=(0, 1, 2), x3dim=(-2, 3, 3))
    print(region_size3)

    region_size4 = me.RegionSize(
        x1dim=(0, 1, 2), x3dim=(-2, 3, 3), x2dim=(0, 1, 2))
    print(region_size4)

    print(region_size2 == region_size3)
    print(region_size4 == region_size3)