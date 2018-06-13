#!/usr/bin/python
#
# This file is part of PyRQA.
# Copyright 2015 Tobias Rawald, Mike Sips.

"""
Results
"""

import numpy as np

from abstract_classes import AbstractSettings, AbstractRuntimes


class RecurrencePlotResult(AbstractSettings, AbstractRuntimes):
    """
    Recurrence plot result

    :ivar recurrence_matrix: initial value: empty 2D array
    """
    def __init__(self,
                 settings,
                 runtimes,
                 recurrence_matrix=np.array([], dtype=np.uint32)):
        AbstractSettings.__init__(self, settings)
        AbstractRuntimes.__init__(self, runtimes)

        self.recurrence_matrix = recurrence_matrix

    @property
    def recurrence_matrix_reverse(self):
        """ Reverted recurrence matrix """
        return self.recurrence_matrix[::-1]


class RQAResult(AbstractSettings, AbstractRuntimes):
    """
    Recurrence quantification analysis result

    :ivar settings: Recurrence analysis settings.
    :ivar runtimes: Computing runtimes.
    :ivar recurrence_points: 1D array containing the local recurrence points counts.
    :ivar diagonal_frequency_distribution: 1D array containing the counts of the lengths of diagonal lines starting at 1.
    :ivar vertical_frequency_distribution: 1D array containing the counts of the lengths of vertical lines starting at 1.
    :ivar white_vertical_frequency_distribution: 1D array containing the counts of the lengths of white vertical lines starting at 1.
    :ivar number_of_recurrence_points: Total number of recurrence points.
    :ivar number_of_diagonal_lines: Total number of diagonal lines.
    :ivar number_of_diagonal_lines_points: Total number of recurrence points that form diagonal lines.
    :ivar longest_diagonal_line: Longest diagonal line.
    :ivar entropy_diagonal_lines: Entropy of diagonal lines.
    :ivar number_of_vertical_lines: Total number of vertical lines.
    :ivar number_of_vertical_lines_points: Total number of recurrence points that form vertical lines.
    :ivar longest_vertical_line: Longest vertical line.
    :ivar entropy_vertical_lines: Entropy of vertical lines.
    :ivar number_of_white_vertical_lines: Total number of white vertical lines.
    :ivar number_of_white_vertical_lines_points: Total number of recurrence points that form white vertical lines.
    :ivar longest_white_vertical_line: Longest white vertical line.
    :ivar entropy_white_vertical_lines: Entropy of white vertical lines.
    """
    def __init__(self,
                 settings,
                 runtimes,
                 recurrence_points=np.array([], dtype=np.uint32),
                 diagonal_frequency_distribution=np.array([], dtype=np.uint32),
                 vertical_frequency_distribution=np.array([], dtype=np.uint32),
                 white_vertical_frequency_distribution=np.array([], dtype=np.uint32)):
        AbstractSettings.__init__(self, settings)
        AbstractRuntimes.__init__(self, runtimes)

        self.recurrence_points = recurrence_points
        self.diagonal_frequency_distribution = diagonal_frequency_distribution
        self.vertical_frequency_distribution = vertical_frequency_distribution
        self.white_vertical_frequency_distribution = white_vertical_frequency_distribution

        self.number_of_recurrence_points = 0

        self.number_of_diagonal_lines = 0
        self.number_of_diagonal_lines_points = 0
        self.longest_diagonal_line = 0
        self.entropy_diagonal_lines = .0

        self.number_of_vertical_lines = 0
        self.number_of_vertical_lines_points = 0
        self.longest_vertical_line = 0
        self.entropy_vertical_lines = .0

        self.number_of_white_vertical_lines = 0
        self.number_of_white_vertical_lines_points = 0
        self.longest_white_vertical_line = 0
        self.entropy_white_vertical_lines = .0

        self.update()

    def update(self):
        """ Update all instance attributes.  """
        self.set_number_of_recurrence_points()

        self.update_diagonal_lines()
        self.update_vertical_lines()
        self.update_white_vertical_lines()

    def update_diagonal_lines(self):
        """ Update instance attributes regarding the detection of diagonal lines. """
        self.set_number_of_diagonal_lines()
        self.set_number_of_diagonal_lines_points()
        self.set_longest_diagonal_line()
        self.set_entropy_diagonal_lines()

    def update_vertical_lines(self):
        """ Update instance attributes regarding the detection of vertical lines. """
        self.set_number_of_vertical_lines()
        self.set_number_of_vertical_lines_points()
        self.set_longest_vertical_line()
        self.set_entropy_vertical_lines()

    def update_white_vertical_lines(self):
        """ Update instance attributes regarding the detection of white vertical lines. """
        self.set_number_of_white_vertical_lines()
        self.set_number_of_white_vertical_lines_points()
        self.set_longest_white_vertical_line()
        self.set_entropy_white_vertical_lines()

    def set_number_of_recurrence_points(self):
        """ Set the total number of recurrence points. """
        self.number_of_recurrence_points = np.sum(self.recurrence_points)

    def set_number_of_diagonal_lines(self):
        """ Set total number of diagonal lines. """
        if self.settings.min_diagonal_line_length > 0:
            self.number_of_diagonal_lines = np.sum(self.diagonal_frequency_distribution[self.settings.min_diagonal_line_length - 1:])
        else:
            raise RuntimeError

    def set_number_of_diagonal_lines_points(self):
        """ Set total number of recurrence points that form diagonal lines. """
        if self.settings.min_diagonal_line_length > 0:
            self.number_of_diagonal_lines_points = np.sum(((np.arange(self.diagonal_frequency_distribution.size) + 1) * self.diagonal_frequency_distribution)[self.settings.min_diagonal_line_length - 1:])
        else:
            raise RuntimeError

    def set_longest_diagonal_line(self):
        """ Set longest diagonal line length (L_max). """
        if self.settings.min_diagonal_line_length > 0:
            non_zero = self.diagonal_frequency_distribution.nonzero()[0]
            if non_zero.size > 0:
                self.longest_diagonal_line = np.uint32(np.max(non_zero) + 1)
        else:
            raise RuntimeError

    def set_entropy_diagonal_lines(self):
        """ Set entropy of diagonal lines (L_entr). """
        if self.settings.min_diagonal_line_length > 0:
            line_lengths = np.array(self.diagonal_frequency_distribution[self.settings.min_diagonal_line_length - 1:], dtype=np.float32)
            non_zero = line_lengths.nonzero()[0]
            if non_zero.size > 0:
                line_lengths = line_lengths[non_zero]
                intermediate_sum = np.sum((line_lengths / self.number_of_diagonal_lines) * (np.log(line_lengths / self.number_of_diagonal_lines)))
                if intermediate_sum != .0:
                    self.entropy_diagonal_lines = - intermediate_sum
                else:
                    self.entropy_diagonal_lines = intermediate_sum
        else:
            raise RuntimeError

    def set_number_of_vertical_lines(self):
        """ Set total number of vertical lines. """
        if self.settings.min_vertical_line_length > 0:
            self.number_of_vertical_lines = np.sum(self.vertical_frequency_distribution[self.settings.min_vertical_line_length - 1:])
        else:
            raise RuntimeError

    def set_number_of_vertical_lines_points(self):
        """ Set total number of recurrence points that form vertical lines. """
        if self.settings.min_vertical_line_length > 0:
            self.number_of_vertical_lines_points = np.sum(((np.arange(self.vertical_frequency_distribution.size) + 1) * self.vertical_frequency_distribution)[self.settings.min_vertical_line_length - 1:])
        else:
            raise RuntimeError

    def set_longest_vertical_line(self):
        """ Set longest vertical line length (V_max). """
        if self.settings.min_vertical_line_length > 0:
            non_zero = self.vertical_frequency_distribution.nonzero()[0]
            if non_zero.size > 0:
                self.longest_vertical_line = np.uint32(np.max(non_zero) + 1)
        else:
            raise RuntimeError

    def set_entropy_vertical_lines(self):
        """ Set entropy of vertical lines (V_entr). """
        if self.settings.min_vertical_line_length > 0:
            line_lenghts = np.array(self.vertical_frequency_distribution[self.settings.min_vertical_line_length - 1:], dtype=np.float32)
            non_zero = line_lenghts.nonzero()[0]
            if non_zero.size > 0:
                line_lengths = line_lenghts[non_zero]
                intermediate_sum = np.sum((line_lengths / self.number_of_vertical_lines) * (np.log(line_lengths / self.number_of_vertical_lines)))
                if intermediate_sum != .0:
                    self.entropy_vertical_lines = - intermediate_sum
                else:
                    self.entropy_vertical_lines = intermediate_sum
        else:
            raise RuntimeError

    def set_number_of_white_vertical_lines(self):
        """ Set total number of white vertical lines. """
        if self.settings.min_white_vertical_line_length > 0:
            self.number_of_white_vertical_lines = np.sum(self.white_vertical_frequency_distribution[self.settings.min_white_vertical_line_length - 1:])
        else:
            raise RuntimeError

    def set_number_of_white_vertical_lines_points(self):
        """ Set total number of recurrence points that form white vertical lines. """
        if self.settings.min_white_vertical_line_length > 0:
            self.number_of_white_vertical_lines_points = np.sum(((np.arange(self.white_vertical_frequency_distribution.size) + 1) * self.white_vertical_frequency_distribution)[self.settings.min_white_vertical_line_length - 1:])
        else:
            raise RuntimeError

    def set_longest_white_vertical_line(self):
        """ Set longest vertical line length (V_max). """
        if self.settings.min_white_vertical_line_length > 0:
            non_zero = self.white_vertical_frequency_distribution.nonzero()[0]
            if non_zero.size > 0:
                self.longest_white_vertical_line = np.uint32(np.max(non_zero) + 1)
        else:
            raise RuntimeError

    def set_entropy_white_vertical_lines(self):
        """ Set entropy of white vertical lines (V_entr). """
        if self.settings.min_white_vertical_line_length > 0:
            line_lenghts = np.array(self.white_vertical_frequency_distribution[self.settings.min_white_vertical_line_length - 1:], dtype=np.float32)
            non_zero = line_lenghts.nonzero()[0]
            if non_zero.size > 0:
                line_lengths = line_lenghts[non_zero]
                self.entropy_white_vertical_lines = - np.sum((line_lengths / self.number_of_white_vertical_lines) * (np.log(line_lengths / self.number_of_white_vertical_lines)))
        else:
            raise RuntimeError

    @property
    def min_diagonal_line_length(self):
        """ Get/Set minimum diagonal line length (L_min). """
        return self.settings.min_diagonal_line_length

    @min_diagonal_line_length.setter
    def min_diagonal_line_length(self, value):
        self.settings.min_diagonal_line_length = value
        self.update_diagonal_lines()

    @property
    def min_vertical_line_length(self):
        """ Get/Set minimum vertical line length (V_min). """
        return self.settings.min_vertical_line_length

    @min_vertical_line_length.setter
    def min_vertical_line_length(self, value):
        self.settings.min_vertical_line_length = value
        self.update_vertical_lines()

    @property
    def min_white_vertical_line_length(self):
        """ Get/Set minimum white vertical line length (W_min). """
        return self.settings.min_white_vertical_line_length

    @min_white_vertical_line_length.setter
    def min_white_vertical_line_length(self, value):
        self.settings.min_white_vertical_line_length = value
        self.update_white_vertical_lines()

    @property
    def recurrence_rate(self):
        """ Recurrence rate (RR). """
        recurrence_rate = np.float(0)
        if self.settings.number_of_vectors > 0:
            recurrence_rate = np.float(self.number_of_recurrence_points) / pow(self.settings.number_of_vectors, 2)
        return recurrence_rate

    @property
    def average_local_recurrence_rate(self):
        """ Average local recurrence rate. """
        average_local_recurrence_rate = np.float(0)
        if self.settings.number_of_vectors > 0:
            average_local_recurrence_rate = np.float(self.number_of_recurrence_points) / self.settings.number_of_vectors
        return average_local_recurrence_rate

    @property
    def determinism(self):
        """ Determinism (DET). """
        determinism = np.float(0)
        if self.number_of_recurrence_points > 0:
            determinism = np.float(self.number_of_diagonal_lines_points) / self.number_of_recurrence_points
        return determinism

    @property
    def average_diagonal_line(self):
        """ Average diagonal line length (L). """
        average_diagonal_line = np.float(0)
        if self.number_of_diagonal_lines > 0:
            average_diagonal_line = np.float(self.number_of_diagonal_lines_points) / self.number_of_diagonal_lines
        return average_diagonal_line

    @property
    def divergence(self):
        """ Divergence (DIV). """
        divergence = np.float(0)
        if self.longest_diagonal_line > 0:
            divergence = np.float(1) / self.longest_diagonal_line
        return divergence

    @property
    def laminarity(self):
        """ Laminarity (LAM). """
        laminarity = np.float(0)
        if self.number_of_recurrence_points > 0:
            laminarity = np.float(self.number_of_vertical_lines_points) / self.number_of_recurrence_points
        return laminarity

    @property
    def trapping_time(self):
        """ Trapping time (TT). """
        trapping_time = np.float(0)
        if self.number_of_vertical_lines > 0:
            trapping_time = np.float(self.number_of_vertical_lines_points) / self.number_of_vertical_lines
        return trapping_time

    @property
    def average_white_vertical_line(self):
        """ Average white vertical line length (W). """
        average_white_vertical_line = np.float(0)
        if self.number_of_white_vertical_lines > 0:
            average_white_vertical_line = np.float(self.number_of_white_vertical_lines_points) / self.number_of_white_vertical_lines
        return average_white_vertical_line

    @property
    def ratio_determinism_recurrence_rate(self):
        """ Ratio determinism / recurrence rate (DET/RR). """
        ratio_determinism_recurrence_rate = np.float(0)
        if self.recurrence_rate > 0:
            ratio_determinism_recurrence_rate = self.determinism / self.recurrence_rate
        return ratio_determinism_recurrence_rate

    @property
    def ratio_laminarity_determinism(self):
        """ Ratio laminarity / determinism (LAM/DET). """
        ratio_laminarity_determinism = np.float(0)
        if self.determinism > 0:
            ratio_laminarity_determinism = self.laminarity / self.determinism
        return ratio_laminarity_determinism

    def indices_by_local_recurrence_rate(self, threshold):
        """
        Indices of recurrence vectors, having a local recurrence rate equal-or-smaller to the threshold.

        :param threshold: local recurrence rate threshold.
        :returns: List of recurrence vector indices.
        :rtype: 1D array.
        """
        local_recurrence_rate = np.float64(self.recurrence_points) / self.settings.number_of_vectors
        return np.nonzero(local_recurrence_rate <= threshold)[0]

    def indices_by_number_of_local_recurrence_points(self, threshold):
        """
        Indices of recurrence vectors, having an equal-or-smaller number of local recurrence points.

        :param threshold: Local recurrence points threshold.
        :returns: List of recurrence vectors indices.
        :rtype: 1D array.
        """
        return np.nonzero(self.recurrence_points <= threshold)[0]

    def persist_diagonal_frequency_distribution(self, file_path):
        """
        Persist diagonal frequency distribution.

        :param file_path: Path to output file.
        """
        with open(file_path, 'w') as output:
            for length_index in np.arange(self.diagonal_frequency_distribution.size):
                line = "%d: %d\n" % (length_index + self.settings.min_diagonal_line_length, self.diagonal_frequency_distribution[length_index])
                output.write(line)

    def persist_vertical_frequency_distribution(self, file_path):
        """
        Persist vertical frequency distribution.

        :param file_path: Path to output file.
        """
        with open(file_path, 'w') as output:
            for length_index in np.arange(self.vertical_frequency_distribution.size):
                line = "%d: %d\n" % (length_index + self.settings.min_vertical_line_length, self.vertical_frequency_distribution[length_index])
                output.write(line)

    def persist_white_vertical_frequency_distribution(self, file_path):
        """
        Persist white vertical frequency distribution.

        :param file_path: Path to output file.
        """
        with open(file_path, 'w') as output:
            for length_index in np.arange(self.white_vertical_frequency_distribution.size):
                line = "%d: %d\n" % (length_index + self.settings.min_white_vertical_line_length, self.white_vertical_frequency_distribution[length_index])
                output.write(line)

    def __str__(self):
        return "RQA Result:\n" \
               "-----------\n" \
               "Minimum diagonal line length (L_min): %d\n" \
               "Minimum vertical line length (V_min): %d\n" \
               "Minimum white vertical line length (W_min): %d\n" \
               "\n" \
               "Recurrence rate (RR): %f\n" \
               "Determinism (DET): %f\n" \
               "Average diagonal line length (L): %f\n" \
               "Longest diagonal line length (L_max): %d\n" \
               "Divergence (DIV): %f\n" \
               "Entropy diagonal lines (L_entr): %f\n" \
               "Laminarity (LAM): %f\n" \
               "Trapping time (TT): %f\n" \
               "Longest vertical line length (V_max): %d\n" \
               "Entropy vertical lines (V_entr): %f\n" \
               "Average white vertical line length (W): %f\n" \
               "Longest white vertical line length (W_max): %d\n" \
               "Entropy white vertical lines (W_entr): %f\n" \
               "\n" \
               "Ratio determinism / recurrence rate (DET/RR): %f\n"\
               "Ratio laminarity / determinism (LAM/DET): %f\n"        %  (self.min_diagonal_line_length,
                                                                           self.min_vertical_line_length,
                                                                           self.min_white_vertical_line_length,
                                                                           self.recurrence_rate,
                                                                           self.determinism,
                                                                           self.average_diagonal_line,
                                                                           self.longest_diagonal_line,
                                                                           self.divergence,
                                                                           self.entropy_diagonal_lines,
                                                                           self.laminarity,
                                                                           self.trapping_time,
                                                                           self.longest_vertical_line,
                                                                           self.entropy_vertical_lines,
                                                                           self.average_white_vertical_line,
                                                                           self.longest_white_vertical_line,
                                                                           self.entropy_white_vertical_lines,
                                                                           self.ratio_determinism_recurrence_rate,
                                                                           self.ratio_laminarity_determinism)
