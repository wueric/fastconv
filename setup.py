from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools

__version__ = '0.0.2'


class get_pybind_include(object):
    """Helper class to determine the pybind11 include path
    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)


# As of Python 3.6, CCompiler has a `has_flag` method.
# cf http://bugs.python.org/issue26689
def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True


def cpp_flag(compiler):
    flags = ['-std=c++11']

    for flag in flags:
        if has_flag(compiler, flag): return flag

    raise RuntimeError('Unsupported compiler -- at least C++11 support '
                       'is needed!')


def make_pybind11_extension_with_flags (module_name, dependencies):

    c_opts = ['-O3', '-ffast-math', '-march=native', '-fopenmp']
    l_opts = ['-fopenmp']

    if sys.platform == 'darwin':
        darwin_opts = ['-stdlib=libc++', '-mmacosx-version-min=10.7', '-std=c++11']

        c_opts = c_opts + darwin_opts
        l_opts = l_opts + darwin_opts

    return Extension(
        module_name,
        dependencies,
        include_dirs=[
            # Path to pybind11 headers
            get_pybind_include(),
            get_pybind_include(user=True)
        ],
        language='c++',
        extra_compile_args = c_opts,
        extra_link_args= l_opts
    )


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        'msvc': ['/EHsc'],
        'unix': ['-march=native', '-fopenmp', '-O3', '-ffast-math'],
    }
    l_opts = {
        'msvc': [],
        'unix': ['-fopenmp'],
    }

    if sys.platform == 'darwin':
        darwin_opts = ['-stdlib=libc++', '-mmacosx-version-min=10.7']
        c_opts['unix'] += darwin_opts
        l_opts['unix'] += darwin_opts

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        link_opts = self.l_opts.get(ct, [])
        if ct == 'unix':
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')
        elif ct == 'msvc':
            opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())
        for ext in self.extensions:
            ext.extra_compile_args = opts
            ext.extra_link_args = link_opts
        print(opts)

        build_ext.build_extensions(self)

ext_modules = [
    Extension(
        'fastcorr.fastcorr_cpp',
        ['fastcorr/fastcorr_cpp/fastcorr_pbind.cpp'],
        include_dirs=[
            # Path to pybind11 headers
            get_pybind_include(),
            get_pybind_include(user=True)
        ],
        language='c++'
    ),
    Extension(
        'conv2d.imageconv_cpp',
        ['conv2d/imageconv_cpp/imageconv_pbind.cpp'],
        include_dirs=[
            # Path to pybind11 headers
            get_pybind_include(),
            get_pybind_include(user=True)
        ],
        language='c++'
    ),
]

py_modules = [
        'conv2d',
        'corr1d',
]

pybind11_ext_modules = [
    make_pybind11_extension_with_flags('corr1d.fastcorr_cpp', ['corr1d/fastcorr_cpp/fastcorr_pbind.cpp']),
    make_pybind11_extension_with_flags('conv2d.imageconv_cpp',
                                       ['conv2d/imageconv_cpp/imageconv_pbind.cpp']),
]

setup(
    name='pfastconv',
    version=__version__,
    author='Eric Gene Wu',
    install_requires=['pybind11>=2.3'],
    setup_requires=['pybind11>=2.3'],
    packages=py_modules,
    ext_modules=pybind11_ext_modules
)
