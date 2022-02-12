from pathlib import Path
from distutils.core import setup, Extension


cpp_dir = Path(__file__).resolve().parents[0] / 'bitboard'
cpp_files = ['board_wrap.cxx', 'board.cpp', 'board_bits.cpp',
             'init.cpp', 'lineshapes.cpp', 'pns.cpp', 'shapes.cpp']
sources = [str(cpp_dir / f) for f in cpp_files]

# optimization flag for msvc compiler in windows
extra_compile_args = ['/O2']
# optimization flag for gcc compiler in linux
# extra_compile_args = ['-O3']

module = Extension('_board', sources=sources,
                   extra_compile_args=extra_compile_args, language='c++')


setup(name='board', ext_modules=[module], py_modules=['board'])