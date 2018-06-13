#!/usr/bin/python
#
# This file is part of PyRQA.
# Copyright 2015 Tobias Rawald, Mike Sips.

"""
Recurrence analysis
"""

import abc
import math
import Queue

import numpy as np

import processing_order as po
from abstract_classes import AbstractSettings, \
    AbstractVerbose


class RecurrencePlot(AbstractSettings, AbstractVerbose):
    """
    Recurrence Plot.

    :ivar recurrence_matrix: Recurrence matrix.
    """
    def __init__(self, settings, verbose):
        AbstractSettings.__init__(self, settings)
        AbstractVerbose.__init__(self, verbose)

        self.__initialise()

    def __initialise(self):
        self.recurrence_matrix = np.zeros((self.settings.number_of_vectors, self.settings.number_of_vectors), dtype=np.uint8)

    def reset(self):
        self.__initialise()


class RQA(AbstractSettings, AbstractVerbose):
    """
    Recurrence quantification analysis.

    :ivar recurrence_points: Local recurrence points.
    :ivar diagonal_frequency_distribution: Frequency distribution of diagonal lines
    :ivar vertical_frequency_distribution: Frequency distribution of vertical lines
    :ivar white_vertical_frequency_distribution: Frequency distribution of white vertical lines
    """
    def __init__(self, settings, verbose):
        AbstractSettings.__init__(self, settings)
        AbstractVerbose.__init__(self, verbose)

        self.__initialise()

    def __initialise(self):
        self.recurrence_points = self.get_emtpy_recurrence_points()
        self.diagonal_frequency_distribution = self.get_emtpy_global_frequency_distribution()
        self.vertical_frequency_distribution = self.get_emtpy_global_frequency_distribution()
        self.white_vertical_frequency_distribution = self.get_emtpy_global_frequency_distribution()

    def reset(self):
        self.__initialise()

    def get_emtpy_recurrence_points(self):
        """
        Get empty recurrence points.

        :return: Empty recurrence points.
        :rtype: 1D array.
        """
        return np.zeros(self.settings.number_of_vectors, dtype=np.uint32)

    def get_emtpy_global_frequency_distribution(self):
        """
        Get empty frequency distribution.

        :returns: Empty global frequency distribution.
        :rtype: 1D array.
        """
        return np.zeros(self.settings.number_of_vectors, dtype=np.uint64)

    def extent_diagonal_frequency_distribution(self):
        """ Extent the content of the diagonal frequency distribution. """
        if self.settings.is_matrix_symmetric:
            self.diagonal_frequency_distribution += self.diagonal_frequency_distribution
            if not self.settings.theiler_corrector:
                self.diagonal_frequency_distribution[-1] -= 1


class SubMatrices(AbstractSettings):
    """
    Processing of sub matrices.

    :ivar edge_length: Inital edge length of sub matrix.
    :ivar processing_order: Processing order of the sub matrices.
    """
    class SubMatrix(object):
        """
        Sub matrix.

        :ivar partition_index_x: X index of sub matrix in partitioned global recurrence matrix.
        :ivar partition_index_y: Y index of sub matrix in partitioned global recurrence matrix.
        :ivar start_x: Global index for first vector (X dimension).
        :ivar start_y: Global index for first vector (Y dimension).
        :ivar dim_x: Number of vectors (X dimension).
        :ivar dim_y: Number of vectors (Y dimension).
        """
        def __init__(self, partition_index_x, partition_index_y, dim_x, start_x, dim_y, start_y):
            self.partition_index_x = partition_index_x
            self.partition_index_y = partition_index_y
            self.start_x = start_x
            self.start_y = start_y
            self.dim_x = dim_x
            self.dim_y = dim_y

        def __str__(self):
            return "Sub Matrix\n" \
                   "----------\n" \
                   "\n" \
                   "Partition Index X: %d\n" \
                   "Partition Index Y: %d\n" \
                   "Start X: %d\n" \
                   "Start Y: %d\n"\
                   "Dim X: %d\n" \
                   "Dim Y: %d\n" % (self.partition_index_x,
                                    self.partition_index_y,
                                    self.start_y,
                                    self.start_x,
                                    self.dim_x,
                                    self.dim_y)

    def __init__(self, settings, edge_length, processing_order):
        AbstractSettings.__init__(self, settings)

        self.edge_length = edge_length
        self.processing_order = processing_order

        self.__initialise()

    def __initialise(self):
        self.create_sub_matrices()

    def reset(self):
        self.__initialise()

    def create_sub_matrices(self):
        """
        Create sub matrices according to the given processing order.
        Each task queue represents an execution level.
        """
        max_edge_length = math.pow(2, 16) - 1
        self.edge_length = max_edge_length if self.edge_length > max_edge_length else self.edge_length
        number_of_partitions = int(math.ceil(float(self.settings.number_of_vectors) / self.edge_length))

        self.sub_matrix_queues = []
        for partition_index_x in np.arange(number_of_partitions):
            for partition_index_y in np.arange(number_of_partitions):
                if partition_index_x == number_of_partitions - 1:
                    dim_x = self.settings.number_of_vectors - partition_index_x * self.edge_length
                    start_x = partition_index_x * self.edge_length
                else:
                    dim_x = self.edge_length
                    start_x = partition_index_x * dim_x

                if partition_index_y == number_of_partitions - 1:
                    dim_y = self.settings.number_of_vectors - partition_index_y * self.edge_length
                    start_y = partition_index_y * self.edge_length
                else:
                    dim_y = self.edge_length
                    start_y = partition_index_y * dim_y

                sub_matrix = SubMatrices.SubMatrix(partition_index_x,
                                                   partition_index_y,
                                                   dim_x,
                                                   start_x,
                                                   dim_y,
                                                   start_y)

                queue_index = None
                if self.processing_order == po.Diagonal:
                    queue_index = partition_index_x + partition_index_y

                elif self.processing_order == po.Vertical:
                    queue_index = partition_index_y

                elif self.processing_order == po.Bulk:
                    queue_index = 0

                if len(self.sub_matrix_queues) <= queue_index:
                    self.sub_matrix_queues.append(Queue.Queue())

                self.sub_matrix_queues[queue_index].put(sub_matrix)

    def get_vectors_x(self, sub_matrix):
        """
        Get vectors (X dimension).

        :param sub_matrix: Sub matrix.
        :returns: Extracted vectors (X dimension).
        :rtype: 1D array.
        """
        return self.settings.get_vectors(sub_matrix.start_x, sub_matrix.dim_x)

    def get_vectors_y(self, sub_matrix):
        """
        Get vectors (Y dimension).

        :param sub_matrix: Sub matrix.
        :returns: Extracted vectors (Y dimension).
        :rtype: 1D array.
        """
        return self.settings.get_vectors(sub_matrix.start_y, sub_matrix.dim_y)

    def get_vectors_x_as_2d_array(self, sub_matrix):
        """
        Get vectors as 2D array (X dimension).

        :param sub_matrix: Sub matrix.
        :returns: Extracted vectors (X dimension).
        :rtype: 2D array.
        """
        return self.settings.get_vectors_as_2d_array(sub_matrix.start_x, sub_matrix.dim_x)

    def get_vectors_y_as_2d_array(self, sub_matrix):
        """
        Get vectors as 2D array (Y dimension).

        :param sub_matrix: Sub matrix.
        :returns: Extracted vectors (Y dimension).
        :rtype: 2D array.
        """
        return self.settings.get_vectors_as_2d_array(sub_matrix.start_y, sub_matrix.dim_y)

    def get_time_series_x(self, sub_matrix):
        """
        Get time series (X dimension).

        :param sub_matrix: Sub matrix.
        :returns: Sub time series (X dimension).
        :rtype: 1D array.
        """
        return self.settings.get_time_series(sub_matrix.start_x, sub_matrix.dim_x)

    def get_time_series_y(self, sub_matrix):
        """
        Get time series (Y dimension).

        :param sub_matrix: Sub matrix.
        :returns: Sub time series (Y dimension).
        :rtype: 1D array.
        """
        return self.settings.get_time_series(sub_matrix.start_y, sub_matrix.dim_y)

    def get_recurrence_matrix(self, sub_matrix, data_type):
        """
        Get sub recurrence matrix.

        :param sub_matrix: Sub matrix.
        :param data_type: Data type of array.
        :returns: Sub recurrence matrix.
        :rtype: 1D array.
        """
        return np.zeros(sub_matrix.dim_x * sub_matrix.dim_y, dtype=data_type)

    def get_bit_recurrence_matrix(self, sub_matrix, data_type):
        """
        Get bit sub recurrence matrix.

        :param sub_matrix: Sub matrix.
        :param data_type: Data type of array.
        :returns: Sub recurrence matrix in bit format.
        :rtype: 1D array.
        """
        bits_per_element = np.dtype(data_type).itemsize * 8
        number_of_elements = sub_matrix.dim_x * math.ceil(float(sub_matrix.dim_y) / bits_per_element)

        return np.zeros(number_of_elements, dtype=data_type)

    def get_recurrence_matrix_size(self, sub_matrix, data_type):
        """
        Get size of sub recurrence matrix.

        :param sub_matrix: Sub matrix.
        :param data_type: Data type of array.
        :returns: Size of sub recurrence matrix.
        :rtype: Integer value.
        """
        return sub_matrix.dim_x * sub_matrix.dim_y * np.dtype(data_type).itemsize

    def get_bit_matrix_size(self, sub_matrix, data_type):
        """
        Get size of sub bit recurrence matrix.

        :param sub_matrix: Sub matrix.
        :param data_type: Data type of array.
        :returns: Size of sub recurrence matrix in bit format.
        :rtype: 1D array.
        """
        bits_per_element = np.dtype(data_type).itemsize * 8
        number_of_elements = sub_matrix.dim_x * math.ceil(float(sub_matrix.dim_y) / bits_per_element)
        size = number_of_elements * np.dtype(data_type).itemsize

        return size, number_of_elements

    @abc.abstractmethod
    def process_sub_matrix(self):
        """
        Processing of a single sub matrix.
        """
        pass


class Carryover(AbstractSettings, AbstractVerbose):
    """
    Perform recurrence quantification analysis based on multiple sub matrices

    :ivar diagonal_length_carryover: Diagonal line length carryover.
    :ivar diagonal_index_carryover: Diagonal index carryover.
    :ivar vertical_length_carryover: Vertical line length carryover.
    :ivar vertical_index_carryover: Vertical index carryover.
    :ivar white_vertical_length_carryover: White vertical line length carryover.
    :ivar white_vertical_index_carryover: White vertical index carryover.
    """
    def __init__(self, settings, verbose):
        AbstractSettings.__init__(self, settings)
        AbstractVerbose.__init__(self, verbose)

        self.__initialise()

    def __initialise(self):
        if self.settings.is_matrix_symmetric:
            self.diagonal_length_carryover = np.zeros(self.settings.number_of_vectors, dtype=np.uint32)
            self.diagonal_index_carryover = np.zeros(self.settings.number_of_vectors, dtype=np.uint32)
        else:
            self.diagonal_length_carryover = np.zeros(self.settings.number_of_vectors * 2 - 1, dtype=np.uint32)
            self.diagonal_index_carryover = np.zeros(self.settings.number_of_vectors * 2 - 1, dtype=np.uint32)

        self.vertical_length_carryover = np.zeros(self.settings.number_of_vectors, dtype=np.uint32)
        self.vertical_index_carryover = np.zeros(self.settings.number_of_vectors, dtype=np.uint32)

        self.white_vertical_length_carryover = np.zeros(self.settings.number_of_vectors, dtype=np.uint32)
        self.white_vertical_index_carryover = np.zeros(self.settings.number_of_vectors, dtype=np.uint32)

    def reset(self):
        self.__initialise()


class RecurrencePlotSubMatrices(RecurrencePlot, SubMatrices):
    """
    Combination of:
    - RecurrencePlot and
    - SubMatrices.
    """
    def __init__(self, settings, verbose, edge_length, processing_order):
        RecurrencePlot.__init__(self, settings, verbose)
        SubMatrices.__init__(self, settings, edge_length, processing_order)

    def reset(self):
        RecurrencePlot.reset(self)
        SubMatrices.reset(self)

    def insert_sub_matrix(self, sub_matrix, data):
        """
        Insert sub matrix in global recurrence matrix
        """
        data = data.reshape(sub_matrix.dim_y, sub_matrix.dim_x)
        self.recurrence_matrix[sub_matrix.start_y:sub_matrix.start_y + sub_matrix.dim_y, sub_matrix.start_x:sub_matrix.start_x + sub_matrix.dim_x] = data


class RQASubMatricesCarryover(RQA, SubMatrices, Carryover):
    """
    Combination of:
    - RQA,
    - SubMatrices,
    - OpenCL and
    - Carryover.
    """
    def __init__(self, settings, verbose, edge_length, processing_order):
        RQA.__init__(self, settings, verbose)
        SubMatrices.__init__(self, settings, edge_length, processing_order)
        Carryover.__init__(self, settings, verbose)

    def reset(self):
        RQA.reset(self)
        SubMatrices.reset(self)
        Carryover.reset(self)

    def get_recurrence_points(self, sub_matrix):
        """
        Get sub matrix specific part of global recurrence points array.

        :param sub_matrix: Sub matrix.
        :returns: Sub array of global recurrence points array.
        :rtype: 1D array.
        """
        recurrence_points_start = sub_matrix.start_x
        recurrence_points_end = sub_matrix.start_x + sub_matrix.dim_x
        recurrence_points = self.recurrence_points[recurrence_points_start:recurrence_points_end]

        return recurrence_points, recurrence_points_start, recurrence_points_end

    def get_empty_local_frequency_distribution(self):
        """
        Get empty local frequency distribution.

        :returns: Empty local frequency distribution.
        :rtype: 1D array.
        """
        return np.zeros(self.settings.number_of_vectors, dtype=np.uint32)

    def get_vertical_length_carryover(self, sub_matrix):
        """
        Get sub matrix specific part of global vertical length carryover array.

        :param sub_matrix: Sub matrix.
        :returns: Sub array of global vertical length carryover array.
        :rtype: 1D array.
        """
        carryover_start = sub_matrix.start_x
        carryover_end = sub_matrix.start_x + sub_matrix.dim_x
        carryover = self.vertical_length_carryover[carryover_start:carryover_end]

        return carryover, carryover_start, carryover_end

    def get_vertical_index_carryover(self, sub_matrix):
        """
        Get sub matrix specific part of global vertical index carryover array.

        :param sub_matrix: Sub matrix.
        :returns: Sub array of global vertical index carryover array.
        :rtype: 1D array.
        """
        carryover_start = sub_matrix.start_x
        carryover_end = sub_matrix.start_x + sub_matrix.dim_x
        carryover = self.vertical_index_carryover[carryover_start:carryover_end]

        return carryover, carryover_start, carryover_end

    def get_white_vertical_length_carryover(self, sub_matrix):
        """
        Get sub matrix specific part of global white vertical length carryover array.

        :param sub_matrix: Sub matrix.
        :returns: Sub array of global white vertical length carryover array.
        :rtype: 1D array.
        """
        carryover_start = sub_matrix.start_x
        carryover_end = sub_matrix.start_x + sub_matrix.dim_x
        carryover = self.white_vertical_length_carryover[carryover_start:carryover_end]

        return carryover, carryover_start, carryover_end

    def get_white_vertical_index_carryover(self, sub_matrix):
        """
        Get sub matrix specific part of global white vertical index carryover array.

        :param sub_matrix: Sub matrix.
        :returns: Sub array of global white vertical index carryover array.
        :rtype: 1D array.
        """
        carryover_start = sub_matrix.start_x
        carryover_end = sub_matrix.start_x + sub_matrix.dim_x
        carryover = self.white_vertical_index_carryover[carryover_start:carryover_end]

        return carryover, carryover_start, carryover_end

    def get_diagonal_length_carryover(self, sub_matrix):
        """
        Get sub matrix specific part of global diagonal length carryover array.

        :param sub_matrix: Sub matrix.
        :returns: Sub array of global diagonal length carryover array.
        :rtype: 1D array.
        """
        if self.settings.is_matrix_symmetric:
            if sub_matrix.partition_index_x < sub_matrix.partition_index_y:
                carryover_start = sub_matrix.start_y - (sub_matrix.start_x + sub_matrix.dim_x)
            else:
                carryover_start = sub_matrix.start_x - sub_matrix.start_y

            carryover_end = carryover_start + sub_matrix.dim_x
        else:
            carryover_start = (self.settings.number_of_vectors - 1) + (sub_matrix.start_x - sub_matrix.dim_y + 1) - sub_matrix.start_y
            carryover_end = carryover_start + (sub_matrix.dim_x + sub_matrix.dim_y - 1)

        carryover = self.diagonal_length_carryover[carryover_start:carryover_end]

        return carryover, carryover_start, carryover_end

    def get_diagonal_index_carryover(self, sub_matrix):
        """
        Get sub matrix specific part of global diagonal index carryover array.

        :param sub_matrix: Sub matrix.
        :returns: Sub array of global diagonal index carryover array.
        :rtype: 1D array.
        """
        if self.settings.is_matrix_symmetric:
            if sub_matrix.partition_index_x < sub_matrix.partition_index_y:
                carryover_start = sub_matrix.start_y - (sub_matrix.start_x + sub_matrix.dim_x)
            else:
                carryover_start = sub_matrix.start_x - sub_matrix.start_y

            carryover_end = carryover_start + sub_matrix.dim_x
        else:
            carryover_start = (self.settings.number_of_vectors - 1) + (sub_matrix.start_x - sub_matrix.dim_y + 1) - sub_matrix.start_y
            carryover_end = carryover_start + (sub_matrix.dim_x + sub_matrix.dim_y - 1)

        carryover = self.diagonal_index_carryover[carryover_start:carryover_end]

        return carryover, carryover_start, carryover_end

    def post_process_length_carryovers(self):
        """ Post process length carryover buffers. """
        for line_length in self.diagonal_length_carryover[self.diagonal_length_carryover > 0]:
            self.diagonal_frequency_distribution[line_length - 1] += 1

        for line_length in self.vertical_length_carryover[self.vertical_length_carryover > 0]:
            self.vertical_frequency_distribution[line_length - 1] += 1

        for line_length in self.white_vertical_length_carryover[self.white_vertical_length_carryover > 0]:
            self.white_vertical_frequency_distribution[line_length - 1] += 1

    def post_process_white_vertical_index_carryover(self):
        """ Post process white vertical index carryover buffer. """
        for idx in self.white_vertical_index_carryover:
            line_length = self.settings.number_of_vectors - idx - 1
            if line_length > 0:
                self.white_vertical_frequency_distribution[line_length - 1] += 1

    @staticmethod
    def get_diagonal_offset(sub_matrix):
        """
        Get diagonal offset.

        :param sub_matrix: Sub matrix.
        :returns: Diagonal offset.
        :rtype: Integer value.
        """
        if sub_matrix.partition_index_x < sub_matrix.partition_index_y:
            return 1
        else:
            return 0
