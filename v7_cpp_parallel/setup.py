from setuptools import setup, Extension
import pybind11

# Get the include path for pybind11
pybind11_include = pybind11.get_include()

# Define the C++ extension module
parallel_cpp_evaluator = Extension(
    'parallel_cpp_evaluator',  # The name of the module that will be imported in Python
    sources=['parallel_cpp_evaluator.cpp', 'parallel_cpp_evaluator_wrapper.cpp'],  # Source files
    include_dirs=[pybind11_include],  # Include directories (pybind11 and your headers)
    language='c++',  # Specify that we are using C++
    extra_compile_args=['-std=c++11', '-fopenmp'],  # Use C++11 standard and OpenMP
    extra_link_args=['-fopenmp'],  # Link against OpenMP
)

# Set up the module
setup(
    name='parallel_cpp_evaluator',
    version='0.1',
    ext_modules=[parallel_cpp_evaluator],  # Add the extension module
    zip_safe=False,
)
