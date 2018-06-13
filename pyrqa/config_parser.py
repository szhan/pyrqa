#!/usr/bin/python
#
# This file is part of PyRQA.
# Copyright 2015 Tobias Rawald, Mike Sips.

"""
Configuration parser.
"""

import json


class ConfigurationParser(object):
    """
    Configuration parser.
    """
    @staticmethod
    def parse(config_file_path):
        """
        Parse configuration data from JSON file.

        :param config_file_path: Path to configuration file.
        :returns: Configuration data.
        """
        with open(config_file_path, 'r') as config_file:
            config_file_data = config_file.read()
            return json.loads(config_file_data)['config_data']

