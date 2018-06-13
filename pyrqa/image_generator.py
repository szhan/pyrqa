#!/usr/bin/python
#
# This file is part of PyRQA.
# Copyright 2015 Tobias Rawald, Mike Sips.

"""
Generate recurrence plot from recurrence matrix
"""
from PIL import Image, ImageOps


class ImageGenerator(object):
    """
    Image generator
    """
    @classmethod
    def generate_recurrence_plot(cls, recurrence_matrix):
        """
        Generate recurrence plot from recurrence matrix

        :param recurrence_matrix: Recurrence matrix.
        :returns: Recurrence plot.
        :rtype: PIL image.
        """
        pil_image = Image.fromarray(recurrence_matrix * 255)
        pil_image = ImageOps.invert(pil_image)
        pil_image = pil_image.convert(mode='RGBA', palette=Image.ADAPTIVE)

        return pil_image

    @classmethod
    def save_recurrence_plot(cls, recurrence_matrix, path):
        """
        Generate and save recurrence plot from recurrence matrix

        :param recurrence_matrix: Recurrence matrix.
        :param path: Path to output file.
        """
        pil_image = ImageGenerator.generate_recurrence_plot(recurrence_matrix)
        pil_image.save(path)
