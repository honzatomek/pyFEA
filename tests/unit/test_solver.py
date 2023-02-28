import os
import sys
import pytest
import numpy as np
from random import randint

SRC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
SRC = os.path.join(SRC, 'bin')
sys.path.append(SRC)

from pyFEA.solver import *

_INT = int
_FLOAT = float

class TestNode:
    def test__init__(self):
        id = randint(1, 999999999)
        x = float(randint(0, 2000) - 1000) / 10.
        y = float(randint(0, 2000) - 1000) / 10.
        z = float(randint(0, 2000) - 1000) / 10.

        n = Node(id, x, y, z)
        assert n.id == id
        assert n.x == x
        assert n.y == y
        assert n.z == z
        assert np.array_equal(n.coors, np.array([x, y, z], dtype=float))


class TestMaterial:
    def test__init__(self):
        name = 'alu'
        E = 70000.
        nu = 0.3
        rho = 2.7e-9
        alpha = 1.5e-5

        m = Material(name, E, nu, rho, alpha)

        assert m.name == name
        assert m.E == E
        assert m.nu == nu
        assert m.rho == rho
        assert m.alpha == alpha

        C = np.array([[1.0 - nu, nu, nu, 0.0, 0.0, 0.0],
                      [nu, 1.0 - nu, nu, 0.0, 0.0, 0.0],
                      [nu, nu, 1.0 - nu, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, (1.0 - 2.0 * nu) / 2.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, (1.0 - 2.0 * nu) / 2.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, (1.0 - 2.0 * nu) / 2.0]], dtype=_FLOAT)
        C *= E / ((1.0 + nu) * (1.0 - 2.0 * nu))

        assert np.array_equal(m.C, C)

        m.E = 210000.
        with pytest.raises(AssertionError):
            assert np.array_equal(m.C, C)

        E = m.E
        C = np.array([[1.0 - nu, nu, nu, 0.0, 0.0, 0.0],
                      [nu, 1.0 - nu, nu, 0.0, 0.0, 0.0],
                      [nu, nu, 1.0 - nu, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, (1.0 - 2.0 * nu) / 2.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, (1.0 - 2.0 * nu) / 2.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, (1.0 - 2.0 * nu) / 2.0]], dtype=_FLOAT)
        C *= E / ((1.0 + nu) * (1.0 - 2.0 * nu))
        assert np.array_equal(m.C, C)


class TestHEX8:
    def test__init__(self):
        eid = randint(1, 999999999)

        nid = randint(1, 999999999)
        x = float(randint(0, 2000) - 1000) / 10.
        y = float(randint(0, 2000) - 1000) / 10.
        z = float(randint(0, 2000) - 1000) / 10.

        n = Node(id, x, y, z)



