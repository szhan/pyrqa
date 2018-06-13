#!/usr/bin/python
#
# This file is part of PyRQA.
# Copyright 2015 Tobias Rawald, Mike Sips.

"""
Neighbourhoods.
"""

from pyrqa.abstract_classes import AbstractNeighbourhood


class FixedRadius(AbstractNeighbourhood):
    """
    Fixed radius neighbourhood.

    :ivar radius: Radius.
    """
    def __init__(self, radius=1.0):
        self.radius = radius

    def contains(self, distance):
        """ See AbstractNeighbourhood. """
        if distance < self.radius:
            return True

        return False

    def __str__(self):
        return "Fixed Radius\n" \
               "------------\n" \
               "Radius: %.2f\n" % self.radius


class RadiusCorridor(AbstractNeighbourhood):
    """
    Radius corridor neighbourhood.

    :ivar inner_radius: Inner radius.
    :ivar outer_radius: Outer radius.
    """
    def __init__(self, inner_radius=0.1, outer_radius=1.0):
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius

    def contains(self, distance):
        """ See AbstractNeighbourhood. """
        if self.inner_radius < distance < self.outer_radius:
            return True

        return False

    def __str__(self):
        return "Radius Corridor\n" \
               "---------------\n" \
               "Inner Radius: %.2f\n" \
               "Outer Radius: %.2f\n" % (self.inner_radius,
                                         self.outer_radius)


class FAN(AbstractNeighbourhood):
    """
    Fixed amount of nearest neighbours neighbourhood.

    :ivar k: Number of nearest neighbours.
    :ivar indices: Indices of neighbours.
    :ivar distances: Distance of neighbours.
    """
    def __init__(self, k=5):
        self.k = k
        self.indices = None
        self.distances = None

    def contains(self, idx):
        """ See AbstractNeighbourhood. """
        if idx in self.indices:
            return True

        return False

    def __str__(self):
        return "Fixed Amount of Nearest Neighbours\n" \
               "----------------------------------\n" \
               "Amount of Nearest Neighbours (k): %d\n" % self.k
