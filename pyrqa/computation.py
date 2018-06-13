#!/usr/bin/python
#
# This file is part of PyRQA.
# Copyright 2015 Tobias Rawald, Mike Sips.

"""
Factories for creating recurrence analysis computations.
"""

from .exceptions import UnsupportedNeighbourhoodException
from .neighbourhood import FixedRadius, RadiusCorridor, FAN

from .rqa.fixed_radius.opencl.column_mat_byte_no_rec import ColumnMatByteNoRec as RQAFixedRadius

from .recurrence_plot.fixed_radius.opencl.column_byte import ColumnByte as RecurrencePlotFixedRadius


class RecurrencePlotComputation(object):
    """
    Factory for creating a RQA computation.
    """
    @classmethod
    def create(cls, settings, **kwargs):
        """
        Create RQA computation.

        :param settings: Recurrence analysis settings.
        """
        if isinstance(settings.neighbourhood, FixedRadius):
            return RecurrencePlotFixedRadius(settings, **kwargs)
        elif isinstance(settings.neighbourhood, RadiusCorridor):
            raise UnsupportedNeighbourhoodException("Neighbourhood '%s' is not yet supported!" % settings.neighbourhood.__class__.__name__)
        elif isinstance(settings.neighbourhood, FAN):
            raise UnsupportedNeighbourhoodException("Neighbourhood '%s' is not yet supported!" % settings.neighbourhood.__class__.__name__)
        else:
            raise UnsupportedNeighbourhoodException("Neighbourhood '%s' is not supported!" % settings.neighbourhood.__class__.__name__)


class RQAComputation(object):
    """
    Factory for creating a recurrence plot computation.
    """
    @classmethod
    def create(cls, settings, **kwargs):
        """
        Create recurrence plot computation.

        :param settings: Recurrence analysis settings.
        """
        if isinstance(settings.neighbourhood, FixedRadius):
            return RQAFixedRadius(settings, **kwargs)
        elif isinstance(settings.neighbourhood, RadiusCorridor):
            raise UnsupportedNeighbourhoodException("Neighbourhood '%s' is not yet supported!" % settings.neighbourhood.__class__.__name__)
        elif isinstance(settings.neighbourhood, FAN):
            raise UnsupportedNeighbourhoodException("Neighbourhood '%s' is not yet supported!" % settings.neighbourhood.__class__.__name__)
        else:
            raise UnsupportedNeighbourhoodException("Neighbourhood '%s' is not supported!" % settings.neighbourhood.__class__.__name__)
