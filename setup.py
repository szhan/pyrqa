import os

from distutils.core import setup


def read(file_name):
    return open(os.path.join(os.path.dirname(__file__), file_name)).read()

setup(
    name="PyRQA",
    packages=[
        "pyrqa",
        "pyrqa.recurrence_plot",
        "pyrqa.recurrence_plot.fixed_radius",
        "pyrqa.recurrence_plot.fixed_radius.opencl",
        "pyrqa.recurrence_plot.fixed_radius.plain",
        "pyrqa.recurrence_plot.fixed_radius.tests",
        "pyrqa.rqa",
        "pyrqa.rqa.fixed_radius",
        "pyrqa.rqa.fixed_radius.opencl",
        "pyrqa.rqa.fixed_radius.plain",
        "pyrqa.rqa.fixed_radius.tests",
        ],
    package_data={
        "pyrqa": [
            "config.json",
            "kernels/*.cl",
            "kernels/euclidean_metric/*.cl",
            "kernels/maximum_metric/*.cl",
            "kernels/taxicab_metric/*.cl"
            ],
        "pyrqa.rqa": [
            "kernels/*.cl"
            ],
        "pyrqa.rqa.fixed_radius": [
            "kernels/euclidean_metric/*.cl",
            "kernels/maximum_metric/*.cl",
            "kernels/taxicab_metric/*.cl"
            ],
        },
    version="0.1.0",
    description="A tool to conduct recurrence quantification analysis and tocreate recurrence plots in a massively parallel manner using the OpenCL framework.",
    author="Tobias Rawald",
    author_email="tobias.rawald@gfz-potsdam.de",
    keywords=["time series analysis", "recurrence quantification analysis", "RQA", "recurrence plot"],
    classifiers=[
        "Programming Language :: Python :: 2.7",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
        ],
    long_description=read('README.md'),
    install_requires=['numpy', 'pyopencl', 'Pillow'],
)