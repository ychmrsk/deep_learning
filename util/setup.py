from distutils.core import setup
from Cython.Build import cythonize

setup(name="neural_network",
      ext_modules=cythonize("neural_network.pyx"))
setup(name="layers",
      ext_modules=cythonize("layers.pyx"))
