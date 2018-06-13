#!/usr/bin/python
#
# This file is part of PyRQA.
# Copyright 2015 Tobias Rawald, Mike Sips.

"""
Recurrence Plot (Column Byte)
"""

import numpy as np
import os
import pyopencl as cl
import threading

import Queue

from ....abstract_classes import AbstractRunnable
from ....opencl import OpenCL
from ....processing_order import Diagonal
from ....recurrence_analysis import RecurrencePlotSubMatrices
from ....result import RecurrencePlotResult
from ....runtimes import Runtimes


class ColumnByte(RecurrencePlotSubMatrices, AbstractRunnable):
    """
    Input Data Representation: Column-Store
    Similarity Value Representation: Byte
    """
    def __init__(self,
                 settings,
                 opencl=None,
                 verbose=False,
                 command_line=False,
                 edge_length=10240,
                 processing_order=Diagonal,
                 optimisations_enabled=False,
                 data_type=np.uint8):
        RecurrencePlotSubMatrices.__init__(self,
                                           settings,
                                           verbose,
                                           edge_length,
                                           processing_order)
        self.opencl = opencl
        self.command_line = command_line
        self.optimisations_enabled = optimisations_enabled
        self.data_type = data_type

        self.__initialise()

    def __initialise(self):
        self.validate_opencl()

        self.threads_runtimes = {}

        for device in self.opencl.devices:
            self.threads_runtimes[device] = Runtimes()

    def reset(self):
        RecurrencePlotSubMatrices.reset(self)
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

        create_matrix_kernel = cl.Kernel(program, 'create_matrix')

        while True:
            try:
                sub_matrix = sub_matrix_queue.get(False)

                transfer_from_device_events = []
                transfer_to_device_events = []
                create_matrix_events = []

                # Time series X
                time_series_x = self.get_time_series_x(sub_matrix)
                time_series_x_buffer = cl.Buffer(context,
                                                 cl.mem_flags.READ_ONLY,
                                                 time_series_x.size * time_series_x.itemsize)

                transfer_to_device_events.append(cl.enqueue_write_buffer(command_queue,
                                                                         time_series_x_buffer,
                                                                         time_series_x,
                                                                         device_offset=0,
                                                                         wait_for=None,
                                                                         is_blocking=False))

                # Time series Y
                time_series_y = self.get_time_series_y(sub_matrix)
                time_series_y_buffer = cl.Buffer(context,
                                                 cl.mem_flags.READ_ONLY,
                                                 time_series_y.size * time_series_y.itemsize)

                transfer_to_device_events.append(cl.enqueue_write_buffer(command_queue,
                                                                         time_series_y_buffer,
                                                                         time_series_y,
                                                                         device_offset=0,
                                                                         wait_for=None,
                                                                         is_blocking=False))

                # Recurrence matrix

                matrix = self.get_recurrence_matrix(sub_matrix,
                                                    data_type=self.data_type)
                matrix_buffer = cl.Buffer(context,
                                          cl.mem_flags.READ_WRITE,
                                          matrix.size * matrix.itemsize)

                transfer_to_device_events.append(cl.enqueue_write_buffer(command_queue,
                                                                         matrix_buffer,
                                                                         matrix,
                                                                         device_offset=0,
                                                                         wait_for=None,
                                                                         is_blocking=False))

                # matrix = np.zeros(1, dtype=self.data_type)
                # matrix_buffer = cl.Buffer(context, cl.mem_flags.READ_WRITE, int(self.get_matrix_size(sub_matrix, self.data_type)))
                # transfer_to_device_events.append( cl.enqueue_write_buffer(command_queue, matrix_buffer, matrix, device_offset=0, wait_for=None, is_blocking=False) )

                # Create matrix kernel
                create_matrix_args = [time_series_x_buffer,
                                      time_series_y_buffer,
                                      np.uint32(sub_matrix.dim_x),
                                      np.uint32(self.settings.embedding_dimension),
                                      np.uint32(self.settings.time_delay),
                                      np.float32(self.settings.neighbourhood.radius),
                                      matrix_buffer]

                OpenCL.set_kernel_args(create_matrix_kernel,
                                       create_matrix_args)

                global_work_size = [int(sub_matrix.dim_x + (device.max_work_group_size - (sub_matrix.dim_x % device.max_work_group_size))), int(sub_matrix.dim_y)]
                local_work_size = None

                create_matrix_events.append(cl.enqueue_nd_range_kernel(command_queue,
                                                                       create_matrix_kernel,
                                                                       global_work_size,
                                                                       local_work_size))

                command_queue.finish()

                # Read buffer
                transfer_from_device_events.append(cl.enqueue_read_buffer(command_queue,
                                                                          matrix_buffer,
                                                                          matrix,
                                                                          device_offset=0,
                                                                          wait_for=None,
                                                                          is_blocking=False))

                command_queue.finish()

                # Insert in recurrence matrix
                self.insert_sub_matrix(sub_matrix, matrix)

                # Get events runtimes
                runtimes = Runtimes()
                runtimes.transfer_to_device = self.opencl.convert_events_runtime(transfer_to_device_events)
                runtimes.transfer_from_device = self.opencl.convert_events_runtime(transfer_from_device_events)
                runtimes.create_matrix = self.opencl.convert_events_runtime(create_matrix_events)

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

        for device in self.opencl.devices:
            runtimes += self.threads_runtimes[device]

        result = RecurrencePlotResult(self.settings,
                                      runtimes,
                                      recurrence_matrix=self.recurrence_matrix)

        return result
