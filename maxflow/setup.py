from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext_modules = [Extension("pygraphcut", ["maxflow.pyx"], include_dirs=[numpy.get_include()], 
	language='c++',
	extra_compile_args=['-std=c++11']
	)]

setup(cmdclass = {'build_ext': build_ext}, ext_modules = ext_modules)