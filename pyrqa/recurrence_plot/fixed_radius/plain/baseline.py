#!/usr/bin/python
#
# This file is part of PyRQA.
# Copyright 2015 Tobias Rawald, Mike Sips.

"""
Recurrence Plot (Baseline)
"""

import numpy as np
import time

from ....abstract_classes import AbstractRunnable
from ....recurrence_analysis import RecurrencePlot
from ....result import RecurrencePlotResult
from ....runtimes import Runtimes


class Baseline(RecurrencePlot, AbstractRunnable):
    """
    Input Data Representation: Column-Store
    Similarity Value Representation: Byte
    Division of Matrix: No
    """
    def __init__(self,
                 settings,
                 verbose=True):
        RecurrencePlot.__init__(self,
                                settings=settings,
                                verbose=verbose)

    def reset(self):
        RecurrencePlot.reset(self)

    def create_matrix(self):
        """
        Create matrix
        """
        for index_x in np.arange(self.settings.number_of_vectors):
            for index_y in np.arange(self.settings.number_of_vectors):
                distance = self.settings.similarity_measure.get_distance_time_series(self.settings.time_series,
                                                                                     self.settings.time_series,
                                                                                     self.settings.embedding_dimension,
                                                                                     self.settings.time_delay,
                                                                                     index_x,
                                                                                     index_y)

                if self.settings.neighbourhood.contains(distance):
                    self.recurrence_matrix[index_y][index_x] = 1

    def run(self):
        self.reset()

        start = time.time()
        self.create_matrix()
        end = time.time()

        runtimes = Runtimes(create_matrix=end-start)


        result = RecurrencePlotResult(self.settings,
                                      runtimes,
                                      recurrence_matrix=self.recurrence_matrix)



        return result
