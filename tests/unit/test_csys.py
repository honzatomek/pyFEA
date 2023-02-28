import os
import sys
import pytest
import numpy as np
from random import randint

SRC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
SRC = os.path.join(SRC, 'bin')
sys.path.append(SRC)

from pyFEA.csys import Cartesian, Cylindrical

class TestCartesian:
    def test__init__(self):
        id = 1001
        o = [10., 0., 0.]
        x = [10., 10., 0.]
        xy = [0., 10., 0.]
        # form = 'xy'
        label = 'test csys'
        for form in ('xz', 'zx', 'xy'):
            csys = Cartesian(id, o, x, xy, form=form, label=label)
            assert csys.id == id
            assert np.array_equal(csys.O, np.array(o))
            assert np.array_equal(csys.P1, np.array(x))
            assert np.array_equal(csys.P2, np.array(xy))
            assert csys.form == form
            assert csys.label == label

        T = np.array([[ 0., 1., 0.],
                      [-1., 0., 0.],
                      [ 0., 0., 1.]], dtype=float)

        assert np.array_equal(csys.T, T)

        x1 = np.array([10., 10., 10.], dtype=float)
        x2 = np.array([10.,  0., 10.], dtype=float)
        assert np.array_equal(csys.to_local(x1), x2)
        assert np.array_equal(csys.to_global(x2), x1)

        with pytest.raises(ValueError):
            csys = Cartesian(id, o, x, xy, form='x', label=label)

        with pytest.raises(ValueError):
            csys = Cartesian(id, o, x, xy, form=1, label=label)

        id = 1001
        o =  np.array([0., 0., 0.], dtype=float)
        x =  np.array([1., 0., 0.], dtype=float)
        xy = np.array([0., 1., 0.], dtype=float)
        form = 'xy'
        label = 'test csys'

        csys = Cartesian(id, o, x, xy, form=form, label=label)
        assert csys.id == id
        assert np.array_equal(csys.O, np.array(o))
        assert np.array_equal(csys.P1, np.array(x))
        assert np.array_equal(csys.P2, np.array(xy))
        assert csys.form == form
        assert csys.label == label

        assert np.array_equal(csys.to_local(o), o)
        assert np.array_equal(csys.to_global(o), o)

    def test_consolidation(self):
        # TODO:
        # global csys = 0
        csys0 = Cartesian.gcsys()
        # csys 1 defined in csys 0, at point [10., 10., 0.], rotated counterclockwise by 90°
        csys1 = Cartesian(1, [10., 10., 0.], [10., 20., 0.], [10., 10., 10.], form='xz', csys=0)
        # csys 2 defined in csys 0, at point [-10., 10., 0.], rotated clockwise by 90°
        csys2 = Cartesian(2, [-10., 10., 0.], [-10., 0., 0.], [-10, 10., 10.], form='xz', csys=1)

        csys = dict()
        for c in (csys0, csys1, csys2):
            csys[c.id] = c

        for c in csys.keys():
            csys[c].consolidate(csys[csys[c].csys])

        coor = np.array([float(randint(-255, 255)),
                         float(randint(-255, 255)),
                         float(randint(-255, 255))], dtype=float)

        assert np.array_equal(coor, csys2.to_global2(coor))



class TestCylindrical:
    def test__init__(self):
        id = 1001
        o = [10., 0., 0.]
        r = [10., 10., 0.]
        rphi = [0., 10., 0.]
        form = 'rphi'
        label = 'test csys'
        for form in ('rz', 'zr', 'rphi'):
            csys = Cylindrical(id, o, r, rphi, form=form, label=label)
            assert csys.id == id
            assert np.array_equal(csys.O, np.array(o))
            assert np.array_equal(csys.P1, np.array(r))
            assert np.array_equal(csys.P2, np.array(rphi))
            assert csys.form == form
            assert csys.label == label

        T = np.array([[ 0., 1., 0.],
                      [-1., 0., 0.],
                      [ 0., 0., 1.]], dtype=float)

        assert np.array_equal(csys.T, T)

        x1 = np.array([10., 10., 10.], dtype=float)
        x2 = np.array([10.,  0., 10.], dtype=float)
        assert np.array_equal(csys.to_local(x1), x2)
        assert np.array_equal(csys.to_global(x2), x1)

        with pytest.raises(ValueError):
            csys = Cylindrical(id, o, r, rphi, form='r', label=label)

        id = 1001
        o = [0., 0., 0.]
        r = [1., 0., 0.]
        rphi = [0., 1., 0.]
        form = 'rphi'
        label = 'test csys'
        csys = Cylindrical(id, o, r, rphi, form=form, label=label)
        assert csys.id == id
        assert np.array_equal(csys.O, np.array(o))
        assert np.array_equal(csys.P1, np.array(r))
        assert np.array_equal(csys.P2, np.array(rphi))
        assert csys.form == form
        assert csys.label == label

        assert np.array_equal(csys.to_local(r), np.array([1., 0., 0.], dtype=float))
        assert np.array_equal(csys.to_local(np.array([0., 1., 0.], dtype=float)), np.array([1., np.pi / 2., 0.], dtype=float))



