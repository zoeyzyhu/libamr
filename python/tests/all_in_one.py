# pylint: disable = too-many-arguments, redefined-outer-name, unused-variable, too-many-locals, no-member, too-many-instance-attributes, invalid-name
"""
Test Ray actors with an all-in-one toy script.

1. Initiates mesh blocks with geometric info.
2. Sets memory views for interior zones.
3. Fills ghost zones of a specific position.
"""

import numpy as np
import ray
from numpy.lib.stride_tricks import as_strided


@ray.remote
class MeshActor:
    """A class representing a mesh block."""

    def __init__(self, NGHOST, nx1, nx2, x1min, x2min, x1max, x2max):
        """Initialize MeshBlock with size, coordinate type, and optional ghost zones."""
        # dimensions ---------------------------------------------------------
        self.NGHOST = NGHOST
        self.nx1, self.nx2 = nx1, nx2
        self.nc1, self.nc2 = nx1 + 2 * NGHOST, nx2 + 2 * NGHOST
        # indices ------------------------------------------------------------
        self.iis, self.ie = NGHOST, NGHOST + nx1 - 1
        self.js, self.je = NGHOST, NGHOST + nx2 - 1
        # coordinates --------------------------------------------------------
        self.x1min, self.x2min = x1min, x2min
        self.x1max, self.x2max = x1max, x2max
        self.x1delta = (x1max - x1min) // nx1
        self.x2delta = (x2max - x2min) // nx2
        self.x1v = [x1min + self.x1delta / 2 - NGHOST * self.x1delta +
                    i * self.x1delta for i in range(nx1 + 2 * NGHOST)]
        self.x2v = [x2min + self.x2delta / 2 - NGHOST * self.x2delta +
                    i * self.x2delta for i in range(nx2 + 2 * NGHOST)]
        self.x1f = [x1min - NGHOST * self.x1delta + i * self.x1delta
                    for i in range(nx1 + 2 * NGHOST + 1)]
        self.x2f = [x2min - NGHOST * self.x2delta + i * self.x2delta
                    for i in range(nx2 + 2 * NGHOST + 1)]
        # matrix -------------------------------------------------------------
        self.matrix = np.random.uniform(0, 1, size=(nx2 + 2 * NGHOST,
                                                    nx1 + 2 * NGHOST))
        # - set first & last NGHOST rows to -1
        for i in range(1, NGHOST + 1):
            self.matrix[i - 1] = [0 for _ in range(nx1 + 2 * NGHOST)]
            self.matrix[-i] = [0 for _ in range(nx1 + 2 * NGHOST)]
        # - set first & last NGHOST cols to -1
        for i in range(nx2 + 2 * NGHOST):
            for j in range(1, NGHOST + 1):
                self.matrix[i][j - 1] = 0
                self.matrix[i][-j] = 0
        # memory views for domain area ---------------------------------------
        self.views = {}

    def get_all_info(self):
        """Return all info of the mesh block."""
        return (self.x1min, self.x2min, self.x1max, self.x2max,
                self.matrix, self.x1v, self.x2v, self.x1f, self.x2f,
                self.iis, self.ie, self.js, self.je, self.x1delta, self.x2delta)

    def set_mview(self):
        """Set memory views for interior zones."""
        for o1 in range(-1, 2):
            for o2 in range(-1, 2):
                if o1 == -1:
                    offset1, len1 = self.NGHOST, self.NGHOST
                if o1 == 0:
                    offset1, len1 = self.NGHOST, self.nx1
                if o1 == 1:
                    offset1, len1 = self.nx1, self.NGHOST
                if o2 == -1:
                    offset2, len2 = self.NGHOST, self.NGHOST
                if o2 == 0:
                    offset2, len2 = self.NGHOST, self.nx2
                if o2 == 1:
                    offset2, len2 = self.nx2, self.NGHOST

                byte = self.matrix.itemsize
                self.views[(o2, o1)] = as_strided(
                    self.matrix[offset2: offset2 + len2,
                                offset1: offset1 + len1],
                    shape=(len2, len1),
                    strides=(self.nc1 * byte, byte)
                )

    def get_volume(self, index1, index2):
        """Return the volume of the mesh block at a specific index."""
        return self.x1v[index1], self.x2v[index2]

    def get_ghost_dimensions(self, cubic_offsets):
        """Return the dimensions of ghost zones."""
        o2, o1 = cubic_offsets
        if o1 == -1:
            offset1, len1 = 0, self.NGHOST
        if o1 == 0:
            offset1, len1 = self.NGHOST, self.nx1
        if o1 == 1:
            offset1, len1 = self.NGHOST + self.nx1, self.NGHOST
        if o2 == -1:
            offset2, len2 = 0, self.NGHOST
        if o2 == 0:
            offset2, len2 = self.NGHOST, self.nx2
        if o2 == 1:
            offset2, len2 = self.NGHOST + self.nx2, self.NGHOST

        if o1 == 0 and o2 == 0:
            pass  # currently not using it

        return offset1, len1, offset2, len2

    def get_mview(self, cubic_offsets):
        """Return the memory view of ghost zones."""
        return self.views[cubic_offsets]

    def update_with_mview(self, neighbor, my_cubic_offsets):
        """Update ghost zones with memory views from neighbors."""
        offset1, len1, offset2, len2 = \
            self.get_ghost_dimensions(my_cubic_offsets)

        o2, o1 = my_cubic_offsets

        ready_refs, remain_refs = ray.wait(
            [neighbor.get_mview.remote((-o2, -o1))])

        self.matrix[offset2: offset2 + len2,
                    offset1: offset1 + len1] = ray.get(ready_refs)[0]


# ============================================================================


def find_mesh_scope(meshblock_limits):
    """Find the scope of the mesh."""
    mesh_scope = [float('-inf'), float('-inf')]
    for limit in meshblock_limits:
        if limit[2] > mesh_scope[0]:  # x1max
            mesh_scope[0] = limit[2]
        if limit[3] > mesh_scope[1]:  # x2max
            mesh_scope[1] = limit[3]
    return mesh_scope


def find_meshblock(x1: float, x2: float, meshblock_limits, actors):
    """Find the meshblock that contains the point (x1, x2)."""
    for i, limit in enumerate(meshblock_limits):
        if limit[0] < x1 < limit[2] and limit[1] < x2 < limit[3]:
            return actors[i]
    return None


def print_actor_info(actor):
    """Print all info of the mesh block."""
    (
        x1min, x2min, x1max, x2max,
        matrix, x1v, x2v, x1f, x2f,
        iis, ie, js, je, x1delta, x2delta
    ) = ray.get(actor.get_all_info.remote())

    # Range
    print(f"x1min: {x1min} - x1max: {x1max};")
    print(f"x2min: {x2min} - x2max {x2max};")
    print(f"delta x1: {x1delta}; delta x2: {x2delta}")

    # Matrix
    print("Matrix:")
    for row in matrix:
        print([round(v, 2) for v in row])

    # Volume, Face, Index
    print(f"x1f: {x1f}")
    print(f"x1v: {x1v}")
    print(f"x2f: {x2f}")
    print(f"x2v: {x2v}")
    print(f"is: {iis}, js: {js}; ie: {ie}, je: {je}\n")


# ============================================================================

meshblock_limits = [
    # (x1min, x2min, x1max, x2max)
    (0., 0., 40., 40.),
    (40., 0., 120., 40.),
    (0., 40., 40., 120.),
    (40., 40., 120., 120.)
]

nx1, nx2 = (4, 2)

NGHOST = 2

# 7, 5
gindex1, gindex2 = 3, 1
coffset = (-1, 0)

# ============================================================================

# Get mesh scope
mesh1_max, mesh2_max = find_mesh_scope(meshblock_limits)

# Initiate meshblocks, set up views
actors = [MeshActor.remote(NGHOST, nx1, nx2, *limit)
          for i, limit in enumerate(meshblock_limits)]
for actor in actors:
    ray.get(actor.set_mview.remote())

# Get x1v, x2v
volumes = []
for actor in actors:
    volumes.append(ray.get(actor.get_volume.remote(gindex1, gindex2)))

# Found consumers and owners
consumers = []
coordinates = []
for i, volume in enumerate(volumes):
    x1v, x2v = volume
    if 0 < x1v < mesh1_max and 0 < x2v < mesh2_max:
        consumers.append(actors[i])
        coordinates.append((x1v, x2v))
        print(f"\nBlock # {i} needs data from coordinate {(x1v, x2v)}.")

owners = []
for (x1v, x2v) in coordinates:
    owners.append(find_meshblock(x1v, x2v, meshblock_limits, actors))

# Print original matrices
for i, owner in enumerate(owners):
    print(f"\n ===== The data consumer # {i} ===== ")
    print_actor_info(consumers[i])
    print(f"\n ===== The data owner # {i} ===== ")
    print_actor_info(owner)

# Fill ghose zones
for i, owner in enumerate(owners):
    ray.get(consumers[i].update_with_mview.remote(owner, coffset))

# Print updated matrices
for i, consumer in enumerate(consumers):
    print(f"\n ===== The updated data consumer # {i} ===== ")
    print_actor_info(consumer)
