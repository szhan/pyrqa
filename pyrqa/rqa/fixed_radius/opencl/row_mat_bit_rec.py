#!/usr/bin/python
#
# This file is part of PyRQA.
# Copyright 2015 Tobias Rawald, Mike Sips.

"""
RQA, Fixed Radius, OpenCL, RowMatBitRec
"""

import numpy as np
import os
import pyopencl as cl
import threading

import Queue

from ....abstract_classes import AbstractRunnable
from ....opencl import OpenCL
from ....processing_order import Diagonal
from ....recurrence_analysis import RQASubMatricesCarryover
from ....result import RQAResult
from ....runtimes import Runtimes


class RowMatBitRec(RQASubMatricesCarryover, AbstractRunnable):
    """
    Input Data Representation: Column-Store
    Similarity Value Materialisation: Yes
    Similarity Value Representation: Byte
    Intermediate Results Recycling: Yes
    """
    def __init__(self,
                 settings,
                 opencl=None,
                 verbose=False,
                 command_line=False,
                 edge_length=10240,
                 processing_order=Diagonal,
                 optimisations_enabled=False,
                 data_type=np.uint32):
        RQASubMatricesCarryover.__init__(self, settings, verbose, edge_length, processing_order)

        self.opencl = opencl
        self.command_line = command_line
        self.optimisations_enabled = optimisations_enabled
        self.data_type = data_type

        self.__initialise()

    def __initialise(self):
        self.validate_opencl()

        self.data_size = np.dtype(self.data_type).itemsize * 8

        self.threads_runtimes = {}
        self.threads_diagonal_frequency_distribution = {}
        self.threads_vertical_frequency_distribution = {}
        self.threads_white_vertical_frequency_distribution = {}

        for device in self.opencl.devices:
            self.threads_runtimes[device] = Runtimes()
            self.threads_diagonal_frequency_distribution[device] = self.get_emtpy_global_frequency_distribution()
            self.threads_vertical_frequency_distribution[device] = self.get_emtpy_global_frequency_distribution()
            self.threads_white_vertical_frequency_distribution[device] = self.get_emtpy_global_frequency_distribution()

    def reset(self):
        RQASubMatricesCarryover.reset(self)
        self.__initialise()

    def validate_opencl(self):
        if not self.opencl:
            self.opencl = OpenCL(verbose=self.verbose,
                                 command_line=self.command_line,
                                 optimisations_enabled=self.optimisations_enabled)

        if not self.opencl.programs_created:
            self.opencl.create_programs(kernel_file_names=self.settings.get_kernel_file_names(self),
                                        similarity_measure_name=self.settings.similarity_measure.name,
                                        leaf_path=os.path.dirname(os.path.abspath(__file__)),
                                        root_path=self.settings.base_path)

    def process_sub_matrix(self, *args, **kwargs):
        device = kwargs['device']
        sub_matrix_queue = kwargs['sub_matrix_queue']

        context = self.opencl.contexts[device]
        command_queue = self.opencl.command_queues[device]
        program = self.opencl.programs[device]

        vertical_kernel = cl.Kernel(program, 'vertical')
        diagonal_kernel = cl.Kernel(program, self.settings.diagonal_kernel_name)
        clear_buffer_kernel = cl.Kernel(program, 'clear_buffer')

        while True:
            try:
                sub_matrix = sub_matrix_queue.get(False)

                transfer_from_device_events = []
                transfer_to_device_events = []
                create_matrix_events = []
                vertical_events = []
                diagonal_events = []

                # Vectors X
                vectors_x = self.get_vectors_x(sub_matrix)

                vectors_x_buffer = cl.Buffer(context,
                                             cl.mem_flags.READ_ONLY,
                                             vectors_x.size * vectors_x.itemsize)

                transfer_to_device_events.append(cl.enqueue_write_buffer(command_queue,
                                                                         vectors_x_buffer,
                                                                         vectors_x,
                                                                         device_offset=0,
                                                                         wait_for=None,
                                                                         is_blocking=False))

                # Vectors Y
                vectors_y = self.get_vectors_y(sub_matrix)

                vectors_y_buffer = cl.Buffer(context,
                                             cl.mem_flags.READ_ONLY,
                                             vectors_y.size * vectors_y.itemsize)

                transfer_to_device_events.append(cl.enqueue_write_buffer(command_queue,
                                                                         vectors_y_buffer,
                                                                         vectors_y,
                                                                         device_offset=0,
                                                                         wait_for=None,
                                                                         is_blocking=False))

                # Recurrence matrix
                # matrix = self.get_bit_matrix(sub_matrix, self.data_type)
                # matrix_buffer = cl.Buffer(context, cl.mem_flags.READ_WRITE, matrix.size * matrix.itemsize)
                # transfer_to_device_events.append(cl.enqueue_write_buffer(command_queue, matrix_buffer, matrix, device_offset=0, wait_for=None, is_blocking=False))

                matrix_size, matrix_elements = self.get_bit_matrix_size(sub_matrix,
                                                                        self.data_type)

                matrix = np.zeros(1,
                                  dtype=self.data_type)

                matrix_buffer = cl.Buffer(context,
                                          cl.mem_flags.READ_WRITE,
                                          int(matrix_size))

                transfer_to_device_events.append(cl.enqueue_write_buffer(command_queue,
                                                                         matrix_buffer,
                                                                         matrix,
                                                                         device_offset=0,
                                                                         wait_for=None,
                                                                         is_blocking=False))

                # Recurrence points
                recurrence_points, \
                    recurrence_points_start, \
                    recurrence_points_end = self.get_recurrence_points(sub_matrix)

                recurrence_points_buffer = cl.Buffer(context,
                                                     cl.mem_flags.READ_WRITE,
                                                     recurrence_points.size * recurrence_points.itemsize)

                transfer_to_device_events.append(cl.enqueue_write_buffer(command_queue,
                                                                         recurrence_points_buffer,
                                                                         recurrence_points,
                                                                         device_offset=0,
                                                                         wait_for=None,
                                                                         is_blocking=False))

                # Vertical frequency distribution
                vertical_frequency_distribution = self.get_empty_local_frequency_distribution()

                vertical_frequency_distribution_buffer = cl.Buffer(context,
                                                                   cl.mem_flags.READ_WRITE,
                                                                   vertical_frequency_distribution.size * vertical_frequency_distribution.itemsize)

                transfer_to_device_events.append(cl.enqueue_write_buffer(command_queue,
                                                                         vertical_frequency_distribution_buffer,
                                                                         vertical_frequency_distribution,
                                                                         device_offset=0,
                                                                         wait_for=None,
                                                                         is_blocking=False))

                # White vertical frequency distribution
                white_vertical_frequency_distribution = self.get_empty_local_frequency_distribution()

                white_vertical_frequency_distribution_buffer = cl.Buffer(context,
                                                                         cl.mem_flags.READ_WRITE,
                                                                         white_vertical_frequency_distribution.size * white_vertical_frequency_distribution.itemsize)

                transfer_to_device_events.append(cl.enqueue_write_buffer(command_queue,
                                                                         white_vertical_frequency_distribution_buffer,
                                                                         white_vertical_frequency_distribution,
                                                                         device_offset=0,
                                                                         wait_for=None,
                                                                         is_blocking=False))

                # Diagonal frequency distribution
                diagonal_frequency_distribution = self.get_empty_local_frequency_distribution()

                diagonal_frequency_distribution_buffer = cl.Buffer(context,
                                                                   cl.mem_flags.READ_WRITE,
                                                                   diagonal_frequency_distribution.size * diagonal_frequency_distribution.itemsize)

                transfer_to_device_events.append(cl.enqueue_write_buffer(command_queue,
                                                                         diagonal_frequency_distribution_buffer,
                                                                         diagonal_frequency_distribution,
                                                                         device_offset=0,
                                                                         wait_for=None,
                                                                         is_blocking=False))

                # Vertical carryover
                vertical_carryover, \
                    vertical_carryover_start,\
                    vertical_carryover_end = self.get_vertical_length_carryover(sub_matrix)

                vertical_carryover_buffer = cl.Buffer(context,
                                                      cl.mem_flags.READ_WRITE,
                                                      vertical_carryover.size * vertical_carryover.itemsize)

                transfer_to_device_events.append(cl.enqueue_write_buffer(command_queue,
                                                                         vertical_carryover_buffer,
                                                                         vertical_carryover,
                                                                         device_offset=0,
                                                                         wait_for=None,
                                                                         is_blocking=False))

                # White vertical carryover
                white_vertical_carryover, \
                    white_vertical_carryover_start,\
                    white_vertical_carryover_end = self.get_white_vertical_length_carryover(sub_matrix)

                white_vertical_carryover_buffer = cl.Buffer(context,
                                                            cl.mem_flags.READ_WRITE,
                                                            white_vertical_carryover.size * white_vertical_carryover.itemsize)

                transfer_to_device_events.append(cl.enqueue_write_buffer(command_queue,
                                                                         white_vertical_carryover_buffer,
                                                                         white_vertical_carryover,
                                                                         device_offset=0,
                                                                         wait_for=None,
                                                                         is_blocking=False))

                # Diagonal carryover
                diagonal_carryover, \
                    diagonal_carryover_start, \
                    diagonal_carryover_end = self.get_diagonal_length_carryover(sub_matrix)

                diagonal_carryover_buffer = cl.Buffer(context,
                                                      cl.mem_flags.READ_WRITE,
                                                      diagonal_carryover.size * diagonal_carryover.itemsize)

                transfer_to_device_events.append(cl.enqueue_write_buffer(command_queue,
                                                                         diagonal_carryover_buffer,
                                                                         diagonal_carryover,
                                                                         device_offset=0,
                                                                         wait_for=None,
                                                                         is_blocking=False))

                command_queue.finish()

                # Clear buffer kernel
                clear_buffer_args = [matrix_buffer]

                OpenCL.set_kernel_args(clear_buffer_kernel,
                                       clear_buffer_args)

                global_work_size = [int(matrix_elements)]
                local_work_size = None

                vertical_events.append(cl.enqueue_nd_range_kernel(command_queue,
                                                                  clear_buffer_kernel,
                                                                  global_work_size,
                                                                  local_work_size))

                command_queue.finish()

                # Vertical kernel
                vertical_args = [vectors_x_buffer,
                                 vectors_y_buffer,
                                 np.uint32(sub_matrix.dim_x),
                                 np.uint32(sub_matrix.dim_y),
                                 np.uint32(self.settings.embedding_dimension),
                                 np.uint32(self.settings.time_delay),
                                 np.float32(self.settings.neighbourhood.radius),
                                 np.uint32(self.data_size),
                                 recurrence_points_buffer,
                                 vertical_frequency_distribution_buffer,
                                 vertical_carryover_buffer,
                                 white_vertical_frequency_distribution_buffer,
                                 white_vertical_carryover_buffer,
                                 matrix_buffer]

                OpenCL.set_kernel_args(vertical_kernel, vertical_args)

                global_work_size = [int(sub_matrix.dim_x + (device.max_work_group_size - (sub_matrix.dim_x % device.max_work_group_size)))]
                local_work_size = None

                vertical_events.append(cl.enqueue_nd_range_kernel(command_queue,
                                                                  vertical_kernel,
                                                                  global_work_size,
                                                                  local_work_size))

                command_queue.finish()

                # Diagonal kernel
                if self.settings.is_matrix_symmetric:
                    diagonal_args = [matrix_buffer,
                                     np.uint32(sub_matrix.dim_x),
                                     np.uint32(sub_matrix.dim_y),
                                     np.uint32(sub_matrix.start_x),
                                     np.uint32(sub_matrix.start_y),
                                     np.uint32(self.settings.theiler_corrector),
                                     np.uint32(self.data_size),
                                     np.uint32(self.get_diagonal_offset(sub_matrix)),
                                     diagonal_frequency_distribution_buffer,
                                     diagonal_carryover_buffer]

                    global_work_size = [int(sub_matrix.dim_x + (device.max_work_group_size - (sub_matrix.dim_x % device.max_work_group_size)))]

                else:
                    diagonal_args = [matrix_buffer,
                                     np.uint32(sub_matrix.dim_x),
                                     np.uint32(sub_matrix.dim_y),
                                     np.uint32(sub_matrix.dim_x + sub_matrix.dim_y - 1),
                                     np.uint32(sub_matrix.start_x),
                                     np.uint32(sub_matrix.start_y),
                                     np.uint32(self.settings.theiler_corrector),
                                     np.uint32(self.data_size),
                                     diagonal_frequency_distribution_buffer,
                                     diagonal_carryover_buffer]

                    global_work_size_x = sub_matrix.dim_x + sub_matrix.dim_y - 1
                    global_work_size = [int(global_work_size_x + (device.max_work_group_size - (global_work_size_x % device.max_work_group_size)))]

                OpenCL.set_kernel_args(diagonal_kernel,
                                       diagonal_args)

                local_work_size = None

                diagonal_events.append(cl.enqueue_nd_range_kernel(command_queue,
                                                                  diagonal_kernel,
                                                                  global_work_size,
                                                                  local_work_size))

                command_queue.finish()

                # Read buffer
                transfer_from_device_events.append(cl.enqueue_read_buffer(command_queue,
                                                                          recurrence_points_buffer,
                                                                          self.recurrence_points[recurrence_points_start:recurrence_points_end],
                                                                          device_offset=0,
                                                                          wait_for=None,
                                                                          is_blocking=False))

                transfer_from_device_events.append(cl.enqueue_read_buffer(command_queue,
                                                                          vertical_frequency_distribution_buffer,
                                                                          vertical_frequency_distribution,
                                                                          device_offset=0,
                                                                          wait_for=None,
                                                                          is_blocking=False))

                transfer_from_device_events.append(cl.enqueue_read_buffer(command_queue,
                                                                          vertical_carryover_buffer,
                                                                          self.vertical_length_carryover[vertical_carryover_start:vertical_carryover_end],
                                                                          device_offset=0,
                                                                          wait_for=None,
                                                                          is_blocking=False))

                transfer_from_device_events.append(cl.enqueue_read_buffer(command_queue,
                                                                          white_vertical_frequency_distribution_buffer,
                                                                          white_vertical_frequency_distribution,
                                                                          device_offset=0,
                                                                          wait_for=None,
                                                                          is_blocking=False))

                transfer_from_device_events.append(cl.enqueue_read_buffer(command_queue,
                                                                          white_vertical_carryover_buffer,
                                                                          self.white_vertical_length_carryover[white_vertical_carryover_start:white_vertical_carryover_end],
                                                                          device_offset=0,
                                                                          wait_for=None,
                                                                          is_blocking=False))

                transfer_from_device_events.append(cl.enqueue_read_buffer(command_queue,
                                                                          diagonal_frequency_distribution_buffer,
                                                                          diagonal_frequency_distribution,
                                                                          device_offset=0,
                                                                          wait_for=None,
                                                                          is_blocking=False))

                transfer_from_device_events.append(cl.enqueue_read_buffer(command_queue,
                                                                          diagonal_carryover_buffer,
                                                                          self.diagonal_length_carryover[diagonal_carryover_start:diagonal_carryover_end],
                                                                          device_offset=0,
                                                                          wait_for=None,
                                                                          is_blocking=False))

                command_queue.finish()

                # Update frequency distributions
                self.threads_vertical_frequency_distribution[device] += vertical_frequency_distribution
                self.threads_white_vertical_frequency_distribution[device] += white_vertical_frequency_distribution
                self.threads_diagonal_frequency_distribution[device] += diagonal_frequency_distribution

                # Get events runtimes
                runtimes = Runtimes()
                runtimes.transfer_to_device = self.opencl.convert_events_runtime(transfer_to_device_events)
                runtimes.transfer_from_device = self.opencl.convert_events_runtime(transfer_from_device_events)
                runtimes.create_matrix = self.opencl.convert_events_runtime(create_matrix_events)
                runtimes.detect_vertical_lines = self.opencl.convert_events_runtime(vertical_events)
                runtimes.detect_diagonal_lines = self.opencl.convert_events_runtime(diagonal_events)

                self.threads_runtimes[device] += runtimes

            except Queue.Empty:
                break

    def run_single_device(self):
        for sub_matrix_queue in self.sub_matrix_queues:
            self.process_sub_matrix(device=self.opencl.devices[0],
                                    sub_matrix_queue=sub_matrix_queue)

    def run_multiple_devices(self):
        for sub_matrix_queue in self.sub_matrix_queues:
            threads = []
            for device in self.opencl.devices:
                kwargs = {'device': device,
                          'sub_matrix_queue': sub_matrix_queue}

                thread = threading.Thread(group=None, target=self.process_sub_matrix, name=None, args=(), kwargs=kwargs)
                thread.start()

                threads.append(thread)

            for thread in threads:
                thread.join()

    def run(self):
        self.reset()

        runtimes = Runtimes()

        if len(self.opencl.devices) == 0:
            print 'No device specified!'
            return 0
        elif len(self.opencl.devices) == 1:
            self.run_single_device()
        elif len(self.opencl.devices) > 1:
            self.run_multiple_devices()

        self.post_process_length_carryovers()

        for device in self.opencl.devices:
            runtimes += self.threads_runtimes[device]
            self.diagonal_frequency_distribution += self.threads_diagonal_frequency_distribution[device]
            self.vertical_frequency_distribution += self.threads_vertical_frequency_distribution[device]
            self.white_vertical_frequency_distribution += self.threads_white_vertical_frequency_distribution[device]

        if self.settings.is_matrix_symmetric:
            self.extent_diagonal_frequency_distribution()

        result = RQAResult(self.settings,
                           runtimes,
                           recurrence_points=self.recurrence_points,
                           diagonal_frequency_distribution=self.diagonal_frequency_distribution,
                           vertical_frequency_distribution=self.vertical_frequency_distribution,
                           white_vertical_frequency_distribution=self.white_vertical_frequency_distribution)

        return result
