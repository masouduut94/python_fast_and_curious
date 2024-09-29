from setuptools import setup, Extension
import pybind11

# Get the include path for pybind11
pybind11_include = pybind11.get_include()

# Define the C++ extension module
cpp_evaluator = Extension(
    'cpp_evaluator',  # The name of the module that will be imported in Python
    sources=['cpp_evaluator.cpp', 'cpp_evaluator_wrapper.cpp'],  # Source files
    include_dirs=[pybind11_include],  # Include directories (pybind11 and your headers)
    language='c++',  # Specify that we are using C++
    extra_compile_args=['-std=c++11'],  # Use C++11 standard
)

# Set up the module
setup(
    name='cpp_evaluator',
    version='0.1',
    ext_modules=[cpp_evaluator],  # Add the extension module
    zip_safe=False,
)