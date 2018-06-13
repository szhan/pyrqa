#!/usr/bin/python
#
# This file is part of PyRQA.
# Copyright 2015 Tobias Rawald, Mike Sips.

"""
Testing Recurrence Plot, Fixed Radius, OpenCL.
"""

import numpy as np
import unittest

from ....settings import Settings
from ....neighbourhood import FixedRadius
from ....metric import EuclideanMetric, MaximumMetric, TaxicabMetric

from ..plain.baseline import Baseline
from ..opencl.column_byte import ColumnByte


class RecurrencePlotFixedRadiusOpenCLTest(unittest.TestCase):
    """
    Testing Recurrence Plot, Fixed Radius, OpenCL.
    """
    @classmethod
    def setUpClass(cls):
        """
        Set up test.

        :cvar time_series: Random time series consisting of floating point values.
        """
        cls.time_series = np.array(np.random.rand(1000),
                                   dtype=np.float32)

    def evaluate_results(self,
                         result_1,
                         result_2):
        """
        Evaluate computing results.

        :param result_1: First result object.
        :param result_2: Second result object.
        """
        self.assertEqual(result_1.settings.number_of_vectors, result_2.settings.number_of_vectors)

        self.assertFalse(False in (result_1.recurrence_matrix == result_2.recurrence_matrix))

        self.assertFalse(False in (result_1.recurrence_matrix_reverse == result_2.recurrence_matrix_reverse))

    def perform_computations(self,
                             settings,
                             verbose=False,
                             edge_length=1000):
        """
        Perform computations

        :param settings: Recurrence analysis settings.
        :param verbose: Verbosity of print outputs.
        :param edge_length: Initial edge length of the sub matrices.
        """
        baseline = Baseline(settings,
                            verbose=verbose)
        result_baseline = baseline.run()

        column_byte = ColumnByte(settings,
                                 verbose=verbose,
                                 edge_length=edge_length)
        result_column_byte = column_byte.run()

        self.evaluate_results(result_baseline,
                              result_column_byte)

    def test_default(self):
        """ Test using the default recurrence analysis settings. """
        for metric in [EuclideanMetric, MaximumMetric, TaxicabMetric]:
            settings = Settings(self.time_series,
                                similarity_measure=metric)
            self.perform_computations(settings)

    def test_partition_100(self):
        """ Test partition 100x100 (and residual sub matrix sizes). """
        for metric in [EuclideanMetric, MaximumMetric, TaxicabMetric]:
            settings = Settings(self.time_series,
                                similarity_measure=metric)
            self.perform_computations(settings,
                                      edge_length=100)

    def test_partition_60(self):
        """ Test partition 60x60 (and residual sub matrix sizes). """
        for metric in [EuclideanMetric, MaximumMetric, TaxicabMetric]:
            settings = Settings(self.time_series,
                                similarity_measure=metric)
            self.perform_computations(settings,
                                      edge_length=60)

    def test_embedding_parameters(self):
        """ Test using different than the default recurrence analysis settings. """
        for metric in [EuclideanMetric, MaximumMetric, TaxicabMetric]:
            settings = Settings(self.time_series,
                                similarity_measure=metric,
                                embedding_dimension=5,
                                time_delay=3,
                                neighbourhood=FixedRadius(0.5))
            self.perform_computations(settings)

if __name__ == "__main__":
    unittest.main()
