#!/usr/bin/python
#
# This file is part of PyRQA.
# Copyright 2015 Tobias Rawald, Mike Sips.

"""
Custom exceptions.
"""


class UnsupportedNeighbourhoodException(Exception):
    """ Neighbourhood chosen is not supported. """
    def __init__(self, message):
        super(UnsupportedNeighbourhoodException, self).__init__(message)


class NoOpenCLPlatformDetectedException(Exception):
    """ No OpenCL platform could be detected. """
    def __init__(self, message):
        super(NoOpenCLPlatformDetectedException, self).__init__(message)


class NoOpenCLDeviceDetectedException(Exception):
    """ No OpenCL device could be detected. """
    def __init__(self, message):
        super(NoOpenCLDeviceDetectedException, self).__init__(message)


class OpenCLPlatformIndexOutOfBoundsException(Exception):
    """ OpenCL Platform index is out of bounds. """
    def __init__(self, message):
        super(OpenCLPlatformIndexOutOfBoundsException, self).__init__(message)


class OpenCLDeviceIndexOutOfBoundsException(Exception):
    """ OpenCL Device index is out of bounds. """
    def __init__(self, message):
        super(OpenCLDeviceIndexOutOfBoundsException, self).__init__(message)


class NoOpenCLKernelsFoundException(Exception):
    """ No OpenCL kernels have been found. """
    def __init__(self, message):
        super(NoOpenCLKernelsFoundException, self).__init__(message)
