# setup.py
from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize("evaluator.pyx",
                          compiler_directives={'language_level' : "3", 'boundscheck': False, 'wraparound': False}),
    include_dirs=[numpy.get_include()],
    extra_compile_args=['-fopenmp'],  # OpenMP flags for compilation
    extra_link_args=['-fopenmp']
)
