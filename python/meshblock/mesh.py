# pylint: disable = import-error, too-few-public-methods, unused-argument, unused-variable, unused-import, undefined-variable
"""Mesh class and related functions."""

import ray
from typing_extensions import Self
from .meshblock_tree import MeshBlockTree
from .coordinate_factory import CoordinateFactory


@ray.remote
class Mesh:
    """A class representing a mesh."""

    def __init__(self, tree: MeshBlockTree, coordinate_type: str, nghost: int):
        """Initialize Mesh with tree, coordinate type, and optional ghost zones."""
        mblocks = []


if __name__ == "__main__":
    mesh = Mesh(tree, "cartesian", 1)
