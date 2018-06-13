#!/usr/bin/python
#
# This file is part of PyRQA.
# Copyright 2015 Tobias Rawald, Mike Sips.

"""
Baseline RQA implementation
"""

import math
import numpy as np

from ....abstract_classes import AbstractRunnable
from ....recurrence_analysis import RQA, Carryover
from ....result import RQAResult
from ....runtimes import Runtimes


class Baseline(RQA, Carryover, AbstractRunnable):
    """
    Full Matrix
    No Sub Matrices
    Plain Python
    """
    def __init__(self,
                 settings,
                 verbose=True):
        RQA.__init__(self, settings, verbose)
        Carryover.__init__(self, settings, verbose)

        self.__initialise()

    def __initialise(self):
        # Bypass possible symmetry of similarity matrix
        self.diagonal_length_carryover = np.zeros(self.settings.number_of_vectors * 2 - 1, dtype=np.uint32)

    def reset(self):
        RQA.reset(self)
        Carryover.reset(self)
        self.__initialise()

    def run(self):
        self.reset()

        for index_x in np.arange(self.settings.number_of_vectors):
            vertical_index = index_x

            for index_y in np.arange(self.settings.number_of_vectors):
                diagonal_index = self.settings.number_of_vectors - 1 + (index_y - index_x)

                distance = self.settings.similarity_measure.get_distance_time_series(self.settings.time_series,
                                                                                     self.settings.time_series,
                                                                                     self.settings.embedding_dimension,
                                                                                     self.settings.time_delay,
                                                                                     index_x,
                                                                                     index_y)

                if self.settings.neighbourhood.contains(distance):
                    # Recurrence points
                    self.recurrence_points[vertical_index] += 1

                    # Diagonal lines
                    if math.fabs(index_y - index_x) >= self.settings.theiler_corrector:
                        self.diagonal_length_carryover[diagonal_index] += 1

                    # Vertical lines
                    self.vertical_length_carryover[vertical_index] += 1

                    # White vertical lines
                    if self.white_vertical_length_carryover[vertical_index] > 0:
                        self.white_vertical_frequency_distribution[self.white_vertical_length_carryover[vertical_index] - 1] += 1

                    self.white_vertical_length_carryover[vertical_index] = 0
                else:
                    # Diagonal lines
                    if self.diagonal_length_carryover[diagonal_index] > 0:
                        self.diagonal_frequency_distribution[self.diagonal_length_carryover[diagonal_index] - 1] += 1

                    self.diagonal_length_carryover[diagonal_index] = 0

                    # Vertical lines
                    if self.vertical_length_carryover[vertical_index] > 0:
                        self.vertical_frequency_distribution[self.vertical_length_carryover[vertical_index] - 1] += 1

                    self.vertical_length_carryover[vertical_index] = 0

                    # White vertical lines
                    self.white_vertical_length_carryover[vertical_index] += 1

        for line_length in self.diagonal_length_carryover:
            if line_length > 0:
                self.diagonal_frequency_distribution[line_length - 1] += 1

        for line_length in self.vertical_length_carryover:
            if line_length > 0:
                self.vertical_frequency_distribution[line_length - 1] += 1

        for line_length in self.white_vertical_length_carryover:
            if line_length > 0:
                self.white_vertical_frequency_distribution[line_length - 1] += 1

        result = RQAResult(self.settings,
                           Runtimes(),
                           recurrence_points=self.recurrence_points,
                           diagonal_frequency_distribution=self.diagonal_frequency_distribution,
                           vertical_frequency_distribution=self.vertical_frequency_distribution,
                           white_vertical_frequency_distribution=self.white_vertical_frequency_distribution)

        return result
