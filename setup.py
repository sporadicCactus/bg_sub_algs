from distutils.core import setup, Extension
import numpy

# define the extension module
vibe_step = Extension('vibe_step', sources=['vibe_step.c'], include_dirs=[numpy.get_include()])
vibe_plus_step = Extension('vibe_plus_step', sources=['vibe_plus_step.c'], include_dirs=[numpy.get_include()])

# run the setup
setup(ext_modules=[vibe_step, vibe_plus_step])
