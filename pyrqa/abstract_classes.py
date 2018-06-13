#!/usr/bin/python
#
# This file is part of PyRQA.
# Copyright 2015 Tobias Rawald, Mike Sips.

"""
Collection of abstract classes.
"""

import abc


class abstractclassmethod(classmethod):
    """
    Abstract class method
    """
    __isabstractmethod__ = True

    def __init__(self, cllble):
        cllble.__isabstractmethod__ = True
        super(abstractclassmethod, self).__init__(cllble)


class AbstractSettings(object):
    """
    Abstract settings.

    :ivar settings: Recurrence analysis settings.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, settings):
        self.settings = settings


class AbstractRuntimes(object):
    """
    Abstract runtimes

    :ivar runtimes: Computing runtimes.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, runtimes):
        self.runtimes = runtimes


class AbstractRunnable(object):
    """
    Abstract runnable.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def run(self):
        """ Perform computations. """
        pass

    def run_single_device(self):
        """ Perform computations using a single computing device. """
        pass

    def run_multiple_devices(self):
        """ Perform computations using multiple computing devices. """
        pass


class AbstractVerbose(object):
    """
    Abstract verbose.

    :ivar verbose: Boolean value indicating the verbosity of print outs.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, verbose):
        self.verbose = verbose

    def print_out(self, string):
        """ Print string if verbose is true. """
        if self.verbose:
            print string


class AbstractMetric(object):
    """
    Abstract metric.
    """
    __metaclass__ = abc.ABCMeta
    name = 'metric'

    @classmethod
    def is_symmetric(cls):
        """ Is the metric symmetric? """
        return True

    @abstractclassmethod
    def get_distance_time_series(cls, time_series_x, time_series_y, embedding_dimension, time_delay, index_x, index_y):
        """
        Get distance between two vectors (time series representation)

        :param time_series_x: Time series on X axis.
        :param time_series_y: Time series on Y axis.
        :param embedding_dimension: Embedding dimension.
        :param time_delay: Time delay.
        :param index_x: Index on X axis.
        :param index_y: Index on Y axis.
        :returns: Distance between two vectors.
        :rtype: Float value.
        """
        pass

    @abstractclassmethod
    def get_distance_vectors(cls, vectors_x, vectors_y, embedding_dimension, index_x, index_y):
        """
        Get distance between two vectors (vectors representation)

        :param vectors_x: Vectors on X axis.
        :param vectors_y: Vectors on Y axis.
        :param embedding_dimension: Embedding dimension.
        :param index_x: Index on X axis.
        :param index_y: Index on Y axis.
        :returns: Distance between two vectors.
        :rtype: Float value.
        """
        pass


class AbstractNeighbourhood(object):
    """
    Abstract neighbourhood.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def contains(self, sample):
        """
        Check whether neighbourhood contains sample object.

        :param sample: Sample object, e.g. distance.
        """
        pass
