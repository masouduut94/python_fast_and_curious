from setuptools import setup, Extension
import pybind11

# Get the include path for pybind11
pybind11_include = pybind11.get_include()

# Define the C++ extension module
shared_mutex_parallel_evaluator = Extension(
    'shared_mutex_parallel_evaluator',  # The name of the module that will be imported in Python
    sources=['shared_mutex_parallel_evaluator.cpp', 'shared_mutex_parallel_evaluator_wrapper.cpp'],  # Source files
    include_dirs=[pybind11_include],  # Include directories (pybind11 and your headers)
    language='c++',  # Specify that we are using C++
    extra_compile_args=['-std=c++17', '-stdlib=libc++', '-I/usr/include/c++/9/'],  # Adjust this path
    extra_link_args=['-stdlib=libc++', '-lpthread'],  # Ensure libc++ is used for linking
)

# Set up the module
setup(
    name='shared_mutex_parallel_evaluator',
    version='0.1',
    ext_modules=[shared_mutex_parallel_evaluator],  # Add the extension module
    zip_safe=False,
)
