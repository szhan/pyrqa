#!/usr/bin/python
#
# This file is part of PyRQA.
# Copyright 2015 Tobias Rawald, Mike Sips.

"""
OpenCL
"""

import math
import numpy as np
import os
import pyopencl as cl

from abstract_classes import AbstractVerbose
from exceptions import NoOpenCLPlatformDetectedException, \
    NoOpenCLDeviceDetectedException, \
    OpenCLPlatformIndexOutOfBoundsException, \
    OpenCLDeviceIndexOutOfBoundsException
from file_reader import FileReader


class OpenCL(AbstractVerbose):
    """
    OpenCL computation

    :ivar command_line: Is the computation conducted via command line?
    :ivar optimisations_enabled: Are the default OpenCL compiler optimisations enabled?
    :ivar platform_id: ID of OpenCL platform.
    :ivar device_ids: IDs of OpenCL devices.
    :ivar platform: OpenCL platform.
    :ivar devices: OpenCL devices.
    :ivar contexts: OpenCL contexts.
    :ivar command_queues: OpenCL command queues.
    :ivar programs: OpenCL programs.
    :ivar programs_created: Are the device specific OpenCL programs created?
    :ivar kernel_file_names: Names of OpenCL kernels.
    :ivar similarity_measure_name: Name of similarity measure
    :ivar leaf_path: Leaf path to search for OpenCL kernel files.
    :ivar root_path: Root path to search for OpenCL kernel files.
    """
    def __init__(self,
                 verbose=False,
                 command_line=False,
                 optimisations_enabled=False,
                 platform_id=0,
                 device_ids=(0,)):
        AbstractVerbose.__init__(self,
                                 verbose)

        self.command_line = command_line
        self.optimisations_enabled = optimisations_enabled
        self.platform_id = platform_id
        self.device_ids = device_ids

        self.__initialise()

    def __initialise(self):
        """ Initialize the instance attributes. """
        self.platform = None
        self.devices = None
        self.contexts = {}
        self.command_queues = {}
        self.programs = {}

        if self.command_line:
            self.create_environment_command_line()
        else:
            self.create_environment()

        self.programs_created = False
        self.kernel_file_names = ()
        self.similarity_measure_name = ''
        self.leaf_path = os.path.abspath('')
        self.root_path = os.path.abspath('')

    def search_paths(self, search_path, kernel_file_name):
        """
        Search a given search path and sub paths.

        :param search_path: Path to search for kernel files.
        :param kernel_file_name: Name of kernel file.
        :returns: Content of kernel file.
        :rtype: String.
        """
        try:
            return FileReader.file_as_string(os.path.join(search_path, 'kernels'), kernel_file_name)
        except IOError:
            try:
                return FileReader.file_as_string(os.path.join(search_path, 'kernels', self.similarity_measure_name), kernel_file_name)
            except IOError:
                return ''

    def locate_kernels(self):
        """
        Locate OpenCL kernel files based on the kernel file names.

        :returns: Kernel file names and their corresponding sources.
        :rtype: Dictionary.
        """
        kernel_dict = dict((kernel_file_name, '') for kernel_file_name in self.kernel_file_names)

        search_path = self.leaf_path
        while search_path != self.root_path:
            search_path = os.path.abspath(os.path.join(search_path, os.pardir))

            for kernel_file_name in self.kernel_file_names:
                kernel_source = kernel_dict[kernel_file_name]
                if not kernel_source:
                    kernel_source = self.search_paths(search_path, kernel_file_name)
                    if not kernel_source:
                        kernel_source = self.search_paths(search_path, kernel_file_name)

                kernel_dict[kernel_file_name] = kernel_source

        return kernel_dict

    def get_program_source(self):
        """
        Get program source for all given kernel file names.

        :returns: Program source.
        :rtype: String.
        """
        program_source = ''
        for kernel_file_name, kernel_source in self.locate_kernels().iteritems():
            if kernel_source:
                program_source += kernel_source
            else:
                print "Kernel with file name '%s' could not be found!" % kernel_file_name

        return program_source

    def create_programs(self,
                        kernel_file_names,
                        similarity_measure_name,
                        leaf_path,
                        root_path):
        """
        Create device specific OpenCL programs.

        :param kernel_file_names: Names of the OpenCL kernel files.
        :param similarity_measure_name: Name of the similarity measure.
        :param leaf_path: Leaf path to search for OpenCL kernel files.
        :param root_path: Root path to search for OpenCL kernel files.
        """
        self.kernel_file_names = kernel_file_names
        self.similarity_measure_name = similarity_measure_name
        self.leaf_path = leaf_path
        self.root_path = root_path

        for device in self.devices:
            program_source = self.get_program_source()
            os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
            program = cl.Program(self.contexts[device], program_source)

            if not self.optimisations_enabled:
                program.build(options=["-cl-opt-disable"])
            else:
                program.build()

            self.programs[device] = program

        self.programs_created = True

    def create_contexts(self):
        """ Create device specific OpenCL contexts. """
        for device in self.devices:
            self.print_out(OpenCL.get_device_info(device))

            context = cl.Context(devices=[device])
            self.contexts[device] = context

            command_queue = cl.CommandQueue(context, properties=cl.command_queue_properties.PROFILING_ENABLE)
            self.command_queues[device] = command_queue

    def create_environment(self):
        """ Create OpenCL environment. """
        self.platform = OpenCL.get_platform(self.platform_id)
        self.print_out(OpenCL.get_platform_info(self.platform))

        self.devices = OpenCL.get_devices(self.platform, self.device_ids)

        self.create_contexts()

    def create_environment_command_line(self):
        """ Create OpenCL environment from command line """
        platforms = cl.get_platforms()
        if platforms:
            platform_strings = []
            for platform_idx in np.arange(len(platforms)):
                platform_strings.append("[%d] %s" % (platform_idx, platforms[platform_idx]))

            platform_select = int(raw_input("\nAvailable platform(s):\n%s\n\nChoose platform, e.g., '0': " % "\n".join(platform_strings)))

            if platform_select not in np.arange(len(platforms)):
                raise OpenCLPlatformIndexOutOfBoundsException("Platform index '%d' is out of bounds." % platform_select)

            self.platform = platforms[platform_select]

            devices = self.platform.get_devices(cl.device_type.ALL)
            if devices:
                device_strings = []
                for device_idx in np.arange(len(devices)):
                    device_strings.append("[%d] %s" % (device_idx, devices[device_idx]))

                device_select = raw_input("\nAvailable device(s):\n%s\n\nChoose device(s), comma-separated, e.g., '0,1': " % "\n".join(device_strings))

                device_indices = [int(x) for x in device_select.split(',')]

                self.devices = []
                for device_idx in device_indices:
                    if device_idx not in np.arange(len(devices)):
                        raise OpenCLDeviceIndexOutOfBoundsException("Device index '%d' is out of bounds.")
                    else:
                        self.devices.append(devices[device_idx])

                self.create_contexts()
            else:
                raise NoOpenCLDeviceDetectedException("No OpenCL device was detected.")
        else:
            raise NoOpenCLPlatformDetectedException("No OpenCL platform was detected.")

    @staticmethod
    def convert_events_runtime(events):
        """
        Convert OpenCL events runtime to seconds.

        :param events: List of OpenCL events.
        :returns: Cumulated runtime in seconds.
        :rtype: Float value.
        """
        runtime = .0
        for event in events:
            tmp = event.get_profiling_info(cl.profiling_info.END) - event.get_profiling_info(cl.profiling_info.QUEUED)
            if tmp > 0:
                runtime += tmp

        return np.float64(runtime) * math.pow(10, -9)

    @staticmethod
    def set_kernel_args(kernel, args):
        """
        Set OpenCL kernel arguments.

        :param kernel: OpenCL kernel.
        :param args: Kernel arguments.
        """
        for idx in np.arange(len(args)):
            kernel.set_arg(int(idx), args[idx])

    @staticmethod
    def get_platform(platform_id=None):
        """
        Get OpenCL platform.

        :param platform_id: ID of OpenCL platform.
        :returns: OpenCL platform.
        :rtype: OpenCL platform object.
        """
        try:
            platforms = cl.get_platforms()
        except cl.RuntimeError as error:
            print "Could not find any platform: ", error

        if platform_id is not None:
            if platform_id in np.arange(len(platforms)):
                platform = platforms[platform_id]
            else:
                raise OpenCLPlatformIndexOutOfBoundsException("Platform with index '%d' could not be found." % platform_id)
        else:
            platform = platforms[0]

        return platform

    @staticmethod
    def get_platform_info(platform):
        """
        Get OpenCL platform info.

        :param platform: OpenCL platform.
        :returns: Platform info.
        :rtype: String.
        """
        return "[Platform '%s']\n" % platform.name.strip() + \
               "Vendor: %s\n" % platform.vendor.strip() + \
               "Version: %s\n" % platform.version + \
               "Profile: %s\n" % platform.profile + \
               "Extensions: %s\n" % platform.extensions + \
               "\n"

    @staticmethod
    def get_devices(platform, device_ids):
        """
        Get OpenCL devices.

        :param platform: OpenCL platform.
        :param device_ids: IDs of OpenCL devices.
        :returns: Array of OpenCL devices.
        :rtype: 1D array.
        """
        try:
            devices = platform.get_devices(cl.device_type.GPU)

            if not devices:
                raise NoOpenCLDeviceDetectedException("No OpenCL device could be detected.")

            devices_selected = []
            for device_id in device_ids:
                if device_id not in np.arange(len(devices)):
                    OpenCLDeviceIndexOutOfBoundsException("Device with index '%d' could not be found." % device_id)

                devices_selected.append(devices[device_id])

            return devices_selected

        except (cl.RuntimeError, NoOpenCLDeviceDetectedException):
            try:
                devices = platform.get_devices(cl.device_type.DEFAULT)

                if not devices:
                    raise NoOpenCLDeviceDetectedException("No OpenCL device could be detected.")

                devices_selected = []
                for device_id in device_ids:
                    if device_id not in np.arange(len(devices)):
                        OpenCLDeviceIndexOutOfBoundsException("Device with index '%d' could not be found." % device_id)

                    devices_selected.append(devices[device_id])

                return devices_selected

            except (cl.RuntimeError, NoOpenCLDeviceDetectedException):
                raise NoOpenCLDeviceDetectedException("No OpenCL device could be detected.")

    @staticmethod
    def get_device_info(device):
        """
        Get OpenCL device info.

        :param device: OpenCL device.
        :returns: Device info.
        :rtype: String.
        """
        return "[Device '%s']\n" % device.name.strip() + \
               "Vendor: %s\n" % device.vendor.strip() + \
               "Type: %s\n" % device.type + \
               "Version: %s\n" % device.version + \
               "Profile: %s\n" % device.profile + \
               "Max Clock Frequency: %s\n" % device.max_clock_frequency + \
               "Global Mem Size: %s\n" % device.global_mem_size + \
               "Address Bits: %s\n" % device.address_bits + \
               "Max Compute Units: %s\n" % device.max_compute_units + \
               "Max Work Group Size: %s\n" % device.max_work_group_size + \
               "Max Work Item Dimensions: %s\n" % device.max_work_item_dimensions + \
               "Max Work Item Sizes: %s\n" % device.max_work_item_sizes + \
               "Local Mem Size: %s\n" % device.local_mem_size + \
               "Max Mem Alloc Size: %s\n" % device.max_mem_alloc_size + \
               "Extensions: %s\n" % device.extensions + \
               "\n"
