#!/usr/bin/python
#
# This file is part of PyRQA.
# Copyright 2015 Tobias Rawald, Mike Sips.

"""
Tests for RQA, Fixed Radius, OpenCL.
"""

import numpy as np
import unittest

from ....metric import EuclideanMetric, MaximumMetric, TaxicabMetric
from ....neighbourhood import FixedRadius
from ....settings import Settings

from ..plain.baseline import Baseline

from ..opencl.row_mat_byte_no_rec import RowMatByteNoRec
from ..opencl.row_mat_byte_rec import RowMatByteRec
from ..opencl.row_mat_bit_no_rec import RowMatBitNoRec
from ..opencl.row_mat_bit_rec import RowMatBitRec
from ..opencl.row_no_mat import RowNoMat
from ..opencl.column_mat_byte_no_rec import ColumnMatByteNoRec
from ..opencl.column_mat_byte_rec import ColumnMatByteRec
from ..opencl.column_mat_bit_no_rec import ColumnMatBitNoRec
from ..opencl.column_mat_bit_rec import ColumnMatBitRec
from ..opencl.column_no_mat import ColumnNoMat

class RQAFixedRadiusOpenclTest(unittest.TestCase):
    """
    Tests for RQA, Fixed Radius, OpenCL.
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

        self.assertFalse(False in (result_1.recurrence_points == result_2.recurrence_points))

        self.assertFalse(False in (result_1.diagonal_frequency_distribution == result_2.diagonal_frequency_distribution))

        self.assertFalse(False in (result_1.vertical_frequency_distribution == result_2.vertical_frequency_distribution))

        self.assertFalse(False in (result_1.white_vertical_frequency_distribution == result_2.white_vertical_frequency_distribution))

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

        row_1 = RowMatByteNoRec(settings,
                                verbose=verbose,
                                edge_length=edge_length)
        result_row_1 = row_1.run()

        self.evaluate_results(result_baseline,
                              result_row_1)

        row_2 = RowMatByteRec(settings,
                              verbose=verbose,
                              edge_length=edge_length)
        result_row_2 = row_2.run()

        self.evaluate_results(result_baseline,
                              result_row_2)

        row_3 = RowMatBitNoRec(settings,
                               verbose=verbose,
                               edge_length=edge_length)
        result_row_3 = row_3.run()

        self.evaluate_results(result_baseline,
                              result_row_3)

        row_4 = RowMatBitRec(settings,
                             verbose=verbose,
                             edge_length=edge_length)
        result_row_4 = row_4.run()

        self.evaluate_results(result_baseline,
                              result_row_4)

        row_5 = RowNoMat(settings,
                         verbose=verbose,
                         edge_length=edge_length)
        result_row_5 = row_5.run()

        self.evaluate_results(result_baseline,
                              result_row_5)

        column_1 = ColumnMatByteNoRec(settings,
                                      verbose=verbose,
                                      edge_length=edge_length)
        result_column_1 = column_1.run()

        self.evaluate_results(result_baseline,
                              result_column_1)

        column_2 = ColumnMatByteRec(settings,
                                    verbose=verbose,
                                    edge_length=edge_length)
        result_column_2 = column_2.run()

        self.evaluate_results(result_baseline,
                              result_column_2)

        column_3 = ColumnMatBitNoRec(settings,
                                     verbose=verbose,
                                     edge_length=edge_length)
        result_column_3 = column_3.run()

        self.evaluate_results(result_baseline,
                              result_column_3)

        column_4 = ColumnMatBitRec(settings,
                                   verbose=verbose,
                                   edge_length=edge_length)
        result_column_4 = column_4.run()

        self.evaluate_results(result_baseline,
                              result_column_4)

        column_5 = ColumnNoMat(settings,
                               verbose=verbose,
                               edge_length=edge_length)
        result_column_5 = column_5.run()

        self.evaluate_results(result_baseline,
                              result_column_5)

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
