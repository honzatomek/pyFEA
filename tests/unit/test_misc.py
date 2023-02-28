import os
import sys
import pytest
import numpy as np
from random import randint

SRC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
SRC = os.path.join(SRC, 'bin')
sys.path.append(SRC)

from pyFEA.misc import ID
from pyFEA.misc import Label
from pyFEA.misc import Coordinate


class TestID:
    # passing tests
    def test__init__pass(self):
        id = 1001
        obj = ID(id)
        assert obj.id == id
        assert obj.label is None

    def test__init__label_str(self):
        id = 1001
        label = 'test'
        obj = ID(id, label)
        assert obj.label == label

    def test__init__label_npstr(self):
        id = 1001
        label = np.str_('test')
        obj = ID(id, label)
        assert obj.label == label

    # failing tests
    def test__init__id_0(self):
        with pytest.raises(ValueError):
            obj = ID(0)

    def test__init__id_m1(self):
        with pytest.raises(ValueError):
            obj = ID(-1)

    def test__init__id_str(self):
        with pytest.raises(ValueError):
            obj = ID('1')

    def test__init__label_int(self):
        id = 1001
        with pytest.raises(ValueError):
            obj = ID(id, label=1)

    def test__init__label_empty(self):
        id = 1001
        with pytest.raises(ValueError):
            obj = ID(id, label='')

class TestLabel:
    # passing tests
    def test__init__pass(self):
        label = 'test'
        l1 = Label(label)
        assert l1.label == label

    # failing tests
    def test__init__zero(self):
        with pytest.raises(ValueError):
            l2 = Label(0)

    def test__init__empty(self):
        with pytest.raises(ValueError):
            l3 = Label('')

    def test__init__None(self):
        with pytest.raises(ValueError):
            l4 = Label(None)


class TestCoordinate:
    # passing tests
    def test__init__pass(self):
        x = float(randint(-255, 255))
        y = float(randint(-255, 255))
        z = float(randint(-255, 255))

        c1 = Coordinate(x, y, z)
        c2 = Coordinate((x, y, z))
        c3 = Coordinate([x, y, z])
        c4 = Coordinate(np.array([x, y, z], dtype=float))
        c5 = Coordinate(np.float_(x), y, z)
        c6 = Coordinate(np.float64(x), y, z)

        for c in (c1, c2, c3, c4, c5, c6):
            assert c.x == x
            assert c.y == y
            assert c.z == z
            assert np.array_equal(c.X, np.array([x, y, z], dtype=float))

    def test__add__pass_coor(self):
        x1 = float(randint(-255, 255))
        y1 = float(randint(-255, 255))
        z1 = float(randint(-255, 255))
        c1 = Coordinate(x1, y1, z1)

        y2 = float(randint(-255, 255))
        x2 = float(randint(-255, 255))
        z2 = float(randint(-255, 255))
        c2 = Coordinate(x2, y2, z2)

        c3 = c1 + c2
        assert c3.x == x1 + x2
        assert c3.y == y1 + y2
        assert c3.z == z1 + z2

    def test__add__pass_float(self):
        x1 = float(randint(-255, 255))
        y1 = float(randint(-255, 255))
        z1 = float(randint(-255, 255))
        c1 = Coordinate(x1, y1, z1)

        c2 = c1 + 5.
        assert c2.x == x1 + 5.
        assert c2.y == y1 + 5.
        assert c2.z == z1 + 5.

    def test__sub__pass_coor(self):
        x1 = float(randint(-255, 255))
        y1 = float(randint(-255, 255))
        z1 = float(randint(-255, 255))
        c1 = Coordinate(x1, y1, z1)

        y2 = float(randint(-255, 255))
        x2 = float(randint(-255, 255))
        z2 = float(randint(-255, 255))
        c2 = Coordinate(x2, y2, z2)

        c3 = c1 - c2
        assert c3.x == x1 - x2
        assert c3.y == y1 - y2
        assert c3.z == z1 - z2

    def test__sub__pass_float(self):
        x1 = float(randint(-255, 255))
        y1 = float(randint(-255, 255))
        z1 = float(randint(-255, 255))
        c1 = Coordinate(x1, y1, z1)

        c2 = c1 - 5.
        assert c2.x == x1 - 5.
        assert c2.y == y1 - 5.
        assert c2.z == z1 - 5.

    def test__iadd__pass_coor(self):
        x1 = float(randint(-255, 255))
        y1 = float(randint(-255, 255))
        z1 = float(randint(-255, 255))
        c1 = Coordinate(x1, y1, z1)

        y2 = float(randint(-255, 255))
        x2 = float(randint(-255, 255))
        z2 = float(randint(-255, 255))
        c2 = Coordinate(x2, y2, z2)

        c1 += c2
        assert c1.x == x1 + x2
        assert c1.y == y1 + y2
        assert c1.z == z1 + z2

    def test__iadd__pass_float(self):
        x1 = float(randint(-255, 255))
        y1 = float(randint(-255, 255))
        z1 = float(randint(-255, 255))
        c1 = Coordinate(x1, y1, z1)

        c1 += 5.
        assert c1.x == x1 + 5.
        assert c1.y == y1 + 5.
        assert c1.z == z1 + 5.

    def test__isub__pass_coor(self):
        x1 = float(randint(-255, 255))
        y1 = float(randint(-255, 255))
        z1 = float(randint(-255, 255))
        c1 = Coordinate(x1, y1, z1)

        y2 = float(randint(-255, 255))
        x2 = float(randint(-255, 255))
        z2 = float(randint(-255, 255))
        c2 = Coordinate(x2, y2, z2)

        c1 -= c2
        assert c1.x == x1 - x2
        assert c1.y == y1 - y2
        assert c1.z == z1 - z2

    def test__isub__pass_float(self):
        x1 = float(randint(-255, 255))
        y1 = float(randint(-255, 255))
        z1 = float(randint(-255, 255))
        c1 = Coordinate(x1, y1, z1)

        c1 -= 5.
        assert c1.x == x1 - 5.
        assert c1.y == y1 - 5.
        assert c1.z == z1 - 5.

    def test__mul__pass_float(self):
        x1 = float(randint(-255, 255))
        y1 = float(randint(-255, 255))
        z1 = float(randint(-255, 255))
        c1 = Coordinate(x1, y1, z1)

        c2 = c1 * 5.
        assert c2.x == x1 * 5.
        assert c2.y == y1 * 5.
        assert c2.z == z1 * 5.

    def test__imul__pass_float(self):
        x1 = float(randint(-255, 255))
        y1 = float(randint(-255, 255))
        z1 = float(randint(-255, 255))
        c1 = Coordinate(x1, y1, z1)

        c1 *= 5.
        assert c1.x == x1 * 5.
        assert c1.y == y1 * 5.
        assert c1.z == z1 * 5.

    def test__truediv__pass_float(self):
        x1 = float(randint(-255, 255))
        y1 = float(randint(-255, 255))
        z1 = float(randint(-255, 255))
        c1 = Coordinate(x1, y1, z1)

        c2 = c1 / 5.
        assert c2.x == x1 / 5.
        assert c2.y == y1 / 5.
        assert c2.z == z1 / 5.

    def test__itruediv__pass_float(self):
        x1 = float(randint(-255, 255))
        y1 = float(randint(-255, 255))
        z1 = float(randint(-255, 255))
        c1 = Coordinate(x1, y1, z1)

        c1 /= 5.
        assert c1.x == x1 / 5.
        assert c1.y == y1 / 5.
        assert c1.z == z1 / 5.

    # failing tests
    def test__init__x_int(self):
        with pytest.raises(ValueError):
            c8 = Coordinate(int(0), 0., 0.)

    def test__init__y_int(self):
        with pytest.raises(ValueError):
            c9 = Coordinate(0., int(0), 0.)

    def test__init__z_int(self):
        with pytest.raises(ValueError):
            c10 = Coordinate(0., 0., int(0))

    def test__init__x_str(self):
        with pytest.raises(ValueError):
            c11 = Coordinate('0.', 0., 0.)

    def test__init__y_str(self):
        with pytest.raises(ValueError):
            c12 = Coordinate(0., '0.', 0.)

    def test__init__z_str(self):
        with pytest.raises(ValueError):
            c13 = Coordinate(0., 0., '0.')

    def test__init__2_coordinates(self):
        with pytest.raises(ValueError):
            c14 = Coordinate(0., 0.)

    def test__init__5_coordinates(self):
        with pytest.raises(ValueError):
            c15 = Coordinate([0., 0., 0.], 0., 0.)


