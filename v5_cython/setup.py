from setuptools import setup
from Cython.Build import cythonize
from setuptools.extension import Extension
import numpy as np

extensions = [
    Extension(
        "evaluator",
        sources=["evaluator.pyx"],
        # language="c++",
        include_dirs=[np.get_include()],
        # extra_compile_args=["-std=c++11"],
    )
]

setup(
    name="Evaluator",
    ext_modules=cythonize(extensions, language_level="3"),
)
