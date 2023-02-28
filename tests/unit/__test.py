import os
import sys

SRC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
SRC = os.path.join(SRC, 'bin')
print(f'{os.path.isdir(SRC) = }')
