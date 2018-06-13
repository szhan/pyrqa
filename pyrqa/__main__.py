#!/usr/bin/python
#
# This file is part of PyRQA.
# Copyright 2015 Tobias Rawald, Mike Sips

"""
Command line recurrence analysis.
"""

import argparse
import numpy as np
import os

from computation import RecurrencePlotComputation, RQAComputation
from file_reader import FileReader
from image_generator import ImageGenerator
from neighbourhood import FixedRadius, RadiusCorridor, FAN
from settings import Settings

def parse_arguments():
    """
    Parse command line arguments.
    """
    prog = "pyrqa"

    usage = '%(prog)s [options] TYPE INPUT_FILE\n' \
            'TYPE of computation. Data is read from INPUT_FILE.\n'

    parser = argparse.ArgumentParser(prog=prog, usage=usage)

    # Positional arguments
    parser.add_argument('type',
                        type=str,
                        choices=['rp', 'rqa'],
                        help='type of computation: rp (recurrence plot), '
                             'rqa (recurrence quantification analysis)')

    parser.add_argument('input_file',
                        help='read data from file',
                        type=str)

    # Optional arguments
    parser.add_argument('-n',
                        '--neighbourhood',
                        default='fr',
                        type=str,
                        choices=['fr', 'rc', 'fan'],
                        help='neighbourhood (choices: fr (fixed radius), rc '
                             '(radius corridor), fan (fixed amount of nearest '
                             'neighbours); default: %(default)s)',
                        metavar='NEIGHBOURHOOD',
                        dest='neighbourhood')

    parser.add_argument('-d'
                        '--delimiter',
                        default=',',
                        help='delimiter of columns (default: %(default)s)',
                        type=str,
                        metavar='DELIMITER',
                        dest='delimiter')

    parser.add_argument('-c'
                        '--column',
                        default=0,
                        help='data column within file (default: %(default)s)',
                        type=np.uint32,
                        metavar='COLUMN',
                        dest='column')

    parser.add_argument('-s'
                        '--offset',
                        default=0,
                        help='offset within data column (default: %(default)s)',
                        type=np.uint32,
                        metavar='OFFSET',
                        dest='offset')

    parser.add_argument('-o',
                        '--output_file',
                        type=str,
                        help='write data to OUTPUT_FILE',
                        metavar='OUTPUT_FILE',
                        dest='output_file')

    parser.add_argument('-m',
                        '--embedding_dimension',
                        default=2,
                        help='embedding dimension',
                        type=np.uint32,
                        metavar='EMBEDDING_DIMENSION',
                        dest='embedding_dimension')

    parser.add_argument('-t',
                        '--time_delay',
                        default=2,
                        help='time delay',
                        type=np.uint32,
                        metavar='TIME_DELAY',
                        dest='time_delay')

    parser.add_argument('-l_min',
                        '--min_diagonal_line_length',
                        default=2,
                        help='minimum diagonal line_length',
                        type=np.uint32,
                        metavar='MIN_DIAGONAL_LINE_LENGTH',
                        dest='min_diagonal_line_length')

    parser.add_argument('-v_min',
                        '--min_vertical_line_length',
                        default=2,
                        help='minimum vertical line length',
                        type=np.uint32,
                        metavar='MIN_VERTICAL_LINE_LENGTH',
                        dest='min_vertical_line_length')

    parser.add_argument('-w_min',
                        '--min_white_vertical_line_length',
                        default=2,
                        help='minimum white vertical line length',
                        type=np.uint32,
                        metavar='MIN_WHITE_VERTICAL_LINE_LENGTH',
                        dest='min_white_vertical_line_length')

    parser.add_argument('-w',
                        '--theiler_corrector',
                        default=1,
                        help='Theiler corrector',
                        type=np.uint32,
                        metavar='THEILER_CORRECTOR',
                        dest='theiler_corrector')

    parser.add_argument('-z'
                        '--edge_length',
                        default=10240,
                        help='edge length of sub matrices',
                        type=np.uint32,
                        metavar='EDGE_LENGTH',
                        dest='edge_length')

    parser.add_argument('-opt',
                        '--optimisations_enabled',
                        default=True,
                        help='Optimisation status',
                        type=np.bool,
                        metavar='OPTIMISATIONS_ENABLED',
                        dest='optimisations_enabled')

    # Argument groups
    fixed_radius = parser.add_argument_group('fixed_radius',
                                             'fixed radius neighbourhood')

    fixed_radius.add_argument('-r',
                              '--radius',
                              default=1.0,
                              type=np.float32,
                              help='radius',
                              metavar='RADIUS',
                              dest='radius')

    radius_corridor = parser.add_argument_group('radius_corridor',
                                                'radius corridor neighbourhood')

    radius_corridor.add_argument('-ri',
                                 '--inner_radius',
                                 default=0.1,
                                 type=np.float32,
                                 help='inner radius',
                                 metavar='INNER_RADIUS',
                                 dest='inner_radius')

    radius_corridor.add_argument('-ro',
                                 '--outer_radius',
                                 default=1.0,
                                 type=np.float32,
                                 help='outer radius',
                                 metavar='OUTER_RADIUS',
                                 dest='outer_radius')

    fan = parser.add_argument_group('fan',
                                    'fixed amount of nearest neighbours neighbourhood')

    fan.add_argument('-k',
                     '--k_nearest_neighbours',
                     default=10,
                     type=np.uint32,
                     help='amount of nearest neighbours',
                     metavar='K_NEAREST_NEIGHBOURS',
                     dest='k_nearest_neighbours')

    return parser.parse_args()


def extend_settings(settings, arguments):
    """
    Extend settings
    """
    if arguments.neighbourhood == 'fr':
        settings.neighbourhood = FixedRadius(arguments.radius)
    elif arguments.neighbourhood == 'rc':
        settings.neighbourhood = RadiusCorridor(arguments.inner_radius,
                                                arguments.outer_radius)
    elif arguments.neighbourhood == 'fan':
        settings.neighbourhood = FAN(arguments.k_nearest_neighbours)


def generate_output_file(result, output_path, args):
    """
    Generate output file
    """
    if args.type == 'rp':
        ImageGenerator.save_recurrence_plot(result.recurrence_matrix_reverse,
                                            output_path)
    elif args.type == 'rqa':
        with open(output_path, 'w') as f:
            for line in result.to_string():
                f.write(line)


if __name__ == "__main__":
    arguments = parse_arguments()

    input_path = os.path.abspath(arguments.input_file)
    time_series = FileReader.file_as_float_array(input_path,
                                                 delimiter=arguments.delimiter,
                                                 column=arguments.column,
                                                 offset=arguments.offset)

    settings = Settings(time_series,
                        embedding_dimension=arguments.embedding_dimension,
                        time_delay=arguments.time_delay,
                        min_diagonal_line_length=arguments.min_diagonal_line_length,
                        min_vertical_line_length=arguments.min_vertical_line_length,
                        min_white_vertical_line_length=arguments.min_white_vertical_line_length,
                        theiler_corrector=arguments.theiler_corrector)

    extend_settings(settings, arguments)

    if arguments.type == 'rp':
        computation = RecurrencePlotComputation.create(settings,
                                                       command_line=True,
                                                       edge_length=arguments.edge_length,
                                                       optimisations_enabled=arguments.optimisations_enabled)
    elif arguments.type == 'rqa':
        computation = RQAComputation.create(settings,
                                            command_line=True,
                                            edge_length=arguments.edge_length,
                                            optimisations_enabled=arguments.optimisations_enabled)

    result = computation.run()

    if arguments.output_file:
        output_path = os.path.abspath(arguments.output_file)
        generate_output_file(result,
                             output_path,
                             arguments)
    else:
        print result
        print result.runtimes
