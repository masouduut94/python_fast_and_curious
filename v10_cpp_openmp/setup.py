from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as build_ext_orig
import pybind11
import os

class build_ext(build_ext_orig):
    def build_extensions(self):
        # Set OpenMP flags for the compiler
        if 'CXX' in os.environ:
            self.compiler.compiler_so.append('-fopenmp')
            self.compiler.linker_so.append('-fopenmp')
        super().build_extensions()

# Path to pybind11
pybind11_include = pybind11.get_include()

# Define the C++ extension
ext_modules = [
    Extension(
        'openmp_evaluator',
        ['openmp_evaluator.cpp', 'openmp_evaluator_wrapper.cpp'],
        include_dirs=[pybind11_include],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
        language='c++'
    ),
]

setup(
    name='openmp_evaluator',
    version='0.1',
    description='OpenMP evaluator with pybind11 bindings',
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext},
)
