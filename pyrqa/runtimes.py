#!/usr/bin/python
#
# This file is part of PyRQA.
# Copyright 2015 Tobias Rawald, Mike Sips.

"""
Recurrence analysis runtimes.
"""


class Runtimes(object):
    """
    Recurrence analysis runtimes.

    :ivar transfer_to_device: Data transfer from the host to the computing device.
    :ivar transfer_from_device: Data transfer from the computing device to the host.
    :ivar create_matrix: Creation of the recurrence matrix.
    :ivar detect_vertical_lines: Detection of vertical lines (including white vertical lines).
    :ivar detect_diagonal_lines: Detection of diagonal lines.
    """
    def __init__(self, transfer_to_device=.0, transfer_from_device=.0, create_matrix=.0, detect_vertical_lines=.0, detect_diagonal_lines=.0):
        self.transfer_to_device = transfer_to_device
        self.transfer_from_device = transfer_from_device
        self.create_matrix = create_matrix
        self.detect_vertical_lines = detect_vertical_lines
        self.detect_diagonal_lines = detect_diagonal_lines

    def __add__(self, other):
        return Runtimes(transfer_to_device=self.transfer_to_device + other.transfer_to_device,
                        transfer_from_device=self.transfer_from_device + other.transfer_from_device,
                        create_matrix=self.create_matrix + other.create_matrix,
                        detect_vertical_lines=self.detect_vertical_lines + other.detect_vertical_lines,
                        detect_diagonal_lines=self.detect_diagonal_lines + other.detect_diagonal_lines)

    def __radd__(self, other):
        return Runtimes(transfer_to_device=self.transfer_to_device + other.transfer_to_device,
                        transfer_from_device=self.transfer_from_device + other.transfer_from_device,
                        create_matrix=self.create_matrix + other.create_matrix,
                        detect_vertical_lines=self.detect_vertical_lines + other.detect_vertical_lines,
                        detect_diagonal_lines=self.detect_diagonal_lines + other.detect_diagonal_lines)

    def __str__(self):
        return "Runtimes\n" \
               "--------\n" \
               "Transfer to Device: %.4fs\n" \
                "Transfer from Device: %.4fs\n" \
                "Create Matrix: %.4fs\n" \
                "Detect Vertical Lines: %.4fs\n" \
                "Detect Diagonal Lines: %.4fs\n" % (self.transfer_to_device,
                                                    self.transfer_from_device,
                                                    self.create_matrix,
                                                    self.detect_vertical_lines,
                                                    self.detect_diagonal_lines)
