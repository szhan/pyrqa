General Information
-------------------

PyRQA is a tool to conduct recurrence quantification analysis (RQA) and to create recurrence plots in a massively parallel manner using the OpenCL framework.
It is designed to process very long time series consisting of hundreds of thousands of data points efficiently.

PyRQA supports the computation of the following RQA measures:

    - Recurrence Rate (RR)
    - Determinism (DET)
    - Average diagonal line length (L)
    - Longest diagonal line (L_max)
    - Divergence (DIV)
    - Entropy diagonal lines (L_entr)
    - Laminarity (LAM)
    - Trapping time (TT)
    - Longest vertical line (V_max)
    - Entropy vertical lines (V_entr)
    - Average white vertical line length (W)
    - Longest white vertical line (W_max)
    - Entropy white vertical lines (W_entr)

In addition, PyRQA allows to compute the thresholded recurrence matrix as well as to save the corresponding recurrence plot as an image file.

Install
-------

PyRQA can be installed via::

    $ pip install PyRQA

Requirements
------------

To run PyRQA on OpenCL devices (such as GPUs and CPUs), it may be required to install hardware vendor specific software, e.g., device drivers.
Regarding the latest versions of MacOS X, OpenCL support should be available by default.

Software and information regarding selected vendors can be found at:

    - AMD: http://developer.amd.com/tools-and-sdks/opencl-zone/
    - Intel: https://software.intel.com/en-us/articles/opencl-drivers
    - NVIDIA: https://developer.nvidia.com/opencl

Usage
-----

RQA computations can be conducted using::

    >>> from pyrqa.settings import Settings
    >>> from pyrqa.neighbourhood import FixedRadius
    >>> from pyrqa.metric import EuclideanMetric
    >>> from pyrqa.computation import RQAComputation
    >>> time_series = [0.1, 0.5, 0.3, 1.7, 0.8, 2.4, 0.6, 1.2, 1.4, 2.1, 0.8]
    >>> settings = Settings(time_series,
                            embedding_dimension=3,
                            time_delay=1,
                            neighbourhood=FixedRadius(1.0),
                            similarity_measure=EuclideanMetric,
                            theiler_corrector=1,
                            min_diagonal_line_length=2,
                            min_vertical_line_length=2,
                            min_white_vertical_line_length=2)
    >>> computation = RQAComputation.create(settings, verbose=True)
    >>> result = computation.run()
    >>> print result

Recurrence plot computations can be conducted likewise. Building on the previous example::

    >>> from pyrqa.computation import RecurrencePlotComputation
    >>> from pyrqa.image_generator import ImageGenerator
    >>> computation = RecurrencePlotComputation.create(settings)
    >>> result = computation.run()
    >>> ImageGenerator.save_recurrence_plot(result.recurrence_matrix_reverse, 'recurrence_plot.png')

Moreover, it is possible to read time series data that is stored column-wise from a file, e.g.,::

    >>> from pyrqa.file_reader import FileReader
    >>> time_series = FileReader.file_as_float_array('data.csv',
                                                     delimiter=';',
                                                     column=0)

The previous examples use the default OpenCL environment. To create a custom environment using command line input, use::

    >>> from pyrqa.opencl import OpenCL
    >>> opencl = OpenCL(command_line=True)

In addition, the OpenCL platform as well as the computing devices can be selected using their IDs, e.g.::

    >>> opencl = OpenCL(platform_id=0, device_ids=(0,))

To use the custom environment, replace the default OpenCL object, e.g.,::

    >>> computation.opencl = opencl

or create a new computation object, e.g.,::

    >>> computation = RQAComputation.create(settings, opencl=opencl, verbose=True)

Environment
-----------

PyRQA has been tested under Python 2.7 on Mac OS X (Mavericks, Yosemite) as well as openSUSE (12.2, 13.2).


Contribution
------------

PyRQA is the result of the work of computer scientists at the Humboldt University of Berlin and the GFZ German Research Centre for Geosciences.

Acknowledgements
----------------

We would like to thank Norbert Marwan from the Potsdam Institute for Climate Impact Research for his continuous support of the project.

Publications
------------

Please acknowledge the use of the PyRQA software by citing the following publication:

    Rawald, T., Sips, M., Marwan, N., Dransch, D. (2014): Fast Computation of Recurrences in Long Time Series. - In: Marwan, N., Riley, M., Guiliani, A., Webber, C. (Eds.), Translational Recurrences. From Mathematical Theory to Real-World Applications, (Springer Proceedings in Mathematics and Statistics ; 103), p. 17-29.

For further information on the evaluation of different implementations, see:

    Rawald, T., Sips, M., Marwan, N., Leser, U. (2015): Massively Parallel Analysis of Similarity Matrices on Heterogeneous Hardware. - In: Fischer, P. M., Alonso, G., Arenas, M., Geerts, F. (Eds.), Proceedings of the Workshops of the EDBT/ICDT 2015 Joint Conference (EDBT/ICDT), (CEUR Workshop Proceedings ; 1330), p. 56-62.
# pyrqa
