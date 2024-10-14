from setuptools import setup
from Cython.Build import cythonize
from setuptools.extension import Extension
import numpy as np

extensions = [
    Extension(
        "evaluator",
        sources=["evaluator.pyx"],
        include_dirs=[np.get_include()],
    )
]

setup(
    name="Evaluator",
    ext_modules=cythonize(extensions, language_level="3"),
)
