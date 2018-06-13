#!/usr/bin/python
#
# This file is part of PyRQA.
# Copyright 2015 Tobias Rawald, Mike Sips.

"""
Settings
"""

import inspect
import numpy as np
import os

from config_parser import ConfigurationParser
from exceptions import NoOpenCLKernelsFoundException
from metric import EuclideanMetric
from neighbourhood import FixedRadius, RadiusCorridor


class Settings(object):
    """
    Settings of recurrence analysis computations.

    :ivar time_series: Time series to be analyzed.
    :ivar embedding_dimension: Embedding dimension.
    :ivar time_delay: Time delay.
    :ivar similarity_measure: Similarity measure, e.g., EuclideanMetric.
    :ivar neighbourhood: Neighbourhood for detecting neighbours, e.g., FixedRadius(1.0).
    :ivar theiler_corrector: Theiler corrector.
    :ivar min_diagonal_line_length: Minimum diagonal line length.
    :ivar min_vertical_line_length: Minimum vertical line length.
    :ivar min_white_vertical_line_length: Minimum white vertical line length.
    :ivar config_file_path: Path to the configuration file that specifies the names of the kernel files.
    """
    def __init__(self,
                 time_series,
                 embedding_dimension=2,
                 time_delay=2,
                 similarity_measure=EuclideanMetric,
                 neighbourhood=FixedRadius(),
                 theiler_corrector=1,
                 min_diagonal_line_length=2,
                 min_vertical_line_length=2,
                 min_white_vertical_line_length=2,
                 config_file_path=os.path.join(os.path.dirname(os.path.relpath(__file__)), "config.json")):
        self.time_series = np.array(time_series, dtype=np.float32)
        self.embedding_dimension = embedding_dimension
        self.time_delay = time_delay
        self.similarity_measure = similarity_measure
        self.neighbourhood = neighbourhood
        self.theiler_corrector = theiler_corrector
        self.min_diagonal_line_length = min_diagonal_line_length
        self.min_vertical_line_length = min_vertical_line_length
        self.min_white_vertical_line_length = min_white_vertical_line_length
        self.config_data = ConfigurationParser.parse(config_file_path)

    @property
    def base_path(self):
        """ Base path of the project. """
        return os.path.dirname(os.path.abspath(__file__))

    @property
    def time_series_length(self):
        """ Length of the input time series. """
        return len(self.time_series)

    @property
    def offset(self):
        """ Time series offset based on embedding dimension and embedding delay. """
        return (self.embedding_dimension - 1) * self.time_delay

    @property
    def number_of_vectors(self):
        """ Number of vectors extracted from series. """
        return self.time_series_length - self.offset

    @property
    def is_matrix_symmetric(self):
        """ Is the recurrence matrix symmetric? """
        if self.similarity_measure.is_symmetric() and \
                (isinstance(self.neighbourhood, FixedRadius) or isinstance(self.neighbourhood, RadiusCorridor)):
            return True

        return False

    @property
    def diagonal_kernel_name(self):
        """ Name of the kernel function to detect the diagonal lines. """
        if self.is_matrix_symmetric:
            return "diagonal_symmetric"
        else:
            return "diagonal"

    def get_time_series(self, start, cnt):
        """
        Get sub time series from the original time series.

        :param start: Start index within the original time series.
        :param cnt: Number of data points to be extracted.
        :returns: Extracted sub time series.
        :rtype: 1D array.
        """
        return self.time_series[start:start + cnt + self.offset]

    def get_vectors(self, start, cnt):
        """
        Get vectors from the original time series.

        :param start: Start index within the original time series.
        :param cnt: Number of vectors to be extracted.
        :returns: Extracted vectors.
        :rtype: 1D array.
        """
        recurrence_vectors = []

        for idx in np.arange(start, start + cnt):
            for dim in np.arange(self.embedding_dimension):
                recurrence_vectors.append(self.time_series[idx + dim * self.time_delay])

        return np.array(recurrence_vectors, dtype=np.float32)

    def get_vectors_as_2d_array(self, start, count):
        """
        Get vectors from the original time series as 2D array.

        :param start: Start index within the original time series.
        :param cnt: Number of vectors to be extracted.
        :returns: Extracted vectors.
        :rtype: 2D array.
        """
        recurrence_vectors = self.get_vectors(start, count)
        recurrence_vectors.shape = (recurrence_vectors.size / self.embedding_dimension, self.embedding_dimension)

        return recurrence_vectors

    def get_kernel_file_names(self, obj):
        """
        Get kernel function file names.

        :param obj: Computation object.
        :returns: Kernel file names.
        :rtype: 1D array.
        """

        cls_list = [cls.__name__ for cls in inspect.getmro(obj.__class__)]
        for element in self.config_data:
            if (element['computation_class'] in cls_list) \
                    and (element['neighbourhood_class'] == self.neighbourhood.__class__.__name__) \
                    and (element['class'] == obj.__class__.__name__):
                return tuple(element['kernel_file_names'])
        raise NoOpenCLKernelsFoundException("Kernels for class '%s' could not be found." % obj.__class__.__name__)

    def __str__(self):
        return "Recurrence Analysis Settings\n" \
               "----------------------------\n" \
               "Embedding dimension: %d\n" \
               "Time delay: %d\n" \
               "Similarity measure: %s\n" \
               "Neighbourhood: %s\n" \
               "Theiler corrector: %d\n" \
               "Minimum diagonal line length: %d\n" \
               "Minimum vertical line length: %d\n" \
               "Minimum white vertical line length: %d\n" \
               "Time series length: %d\n" \
               "Offset: %d\n" \
               "Number of vectors: %d\n" \
               "Matrix symmetry: %r\n" % (self.embedding_dimension,
                                          self.time_delay,
                                          self.similarity_measure,
                                          self.neighbourhood,
                                          self.theiler_corrector,
                                          self.min_diagonal_line_length,
                                          self.min_vertical_line_length,
                                          self.min_white_vertical_line_length,
                                          self.time_series_length,
                                          self.offset,
                                          self.number_of_vectors,
                                          self.is_matrix_symmetric)
