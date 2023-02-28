import os
import sys
import pytest
import numpy as np
from random import randint

SRC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
SRC = os.path.join(SRC, 'bin')
sys.path.append(SRC)

from pyFEA.node import Node, Nodes

class TestNode:
    def test__init__(self):
        id = randint(1, 255)
        x = float(randint(-255, 255))
        y = float(randint(-255, 255))
        z = float(randint(-255, 255))
        csys = randint(1, 255)
        label = ''.join([chr(randint(33, 127)) for i in range(10)])
        nd = Node(id, x, y, z, csys=csys, label=label)
        assert nd.id == id
        assert nd.x == x
        assert nd.y == y
        assert nd.z == z
        assert nd.csys == csys
        assert nd.label == label

        coors = np.array([x, y, z], dtype=float)
        nd = Node(id, x=coors)
        assert nd.x == x
        assert nd.y == y
        assert nd.z == z
        assert nd.csys == 0
        assert nd.label is None

        with pytest.raises(ValueError):
            nd = Node('1', x, y)

        with pytest.raises(ValueError):
            nd = Node('1', x, y, z)

        with pytest.raises(ValueError):
            nd = Node(0, x, y, z)

        with pytest.raises(TypeError):
            nd = Node(id)

        with pytest.raises(ValueError):
            nd = Node(id, 1.)

        with pytest.raises(ValueError):
            nd = Node(id, 1., 1.)

        with pytest.raises(ValueError):
            nd = Node(id, x=coors, csys='1')

        with pytest.raises(ValueError):
            nd = Node(id, x=coors, csys=-1)

    def test_X_g(self):
        id = 1
        x = 1000.
        y = 2000.
        z = 3000.
        nd = Node(id, x, y, z)
        with pytest.raises(Exception):
            nd.X_g

        offset = np.array([0., 0., 0.], dtype=float)
        T = np.array([[1., 0., 0.],
                      [0., 1., 0.],
                      [0., 0., 1.]], dtype=float)
        nd.set_X_g(offset, T)
        assert np.array_equal(nd.X_l, nd.X_g)

        nd.set_X_g()
        assert np.array_equal(nd.X_l, nd.X_g)

        T = np.array([[ 0., 1., 0.],
                      [-1., 0., 0.],
                      [ 0., 0., 1.]], dtype=float)
        nd.set_X_g(offset, T)
        assert np.array_equal(np.array([y, -x, z], dtype=float), nd.X_g)

        offset = np.array([2000., 0., 0.], dtype=float)
        T = np.array([[1., 0., 0.],
                      [0., 1., 0.],
                      [0., 0., 1.]], dtype=float)
        nd.set_X_g(offset, T)
        assert np.array_equal(np.array([x - 2000., y, z], dtype=float), nd.X_g)

        offset = np.array([2000., 0., 0.], dtype=float)
        T = np.array([[ 0., 1., 0.],
                      [-1., 0., 0.],
                      [ 0., 0., 1.]], dtype=float)
        nd.set_X_g(offset, T)
        assert np.array_equal(np.array([y, -(x - 2000.), z], dtype=float), nd.X_g)


class TestNodes:
    def test__init__(self):
        id = 1
        x = 1000.
        y = 2000.
        z = 3000.
        offset = np.array([0., 0., 0.], dtype=float)
        T = np.array([[1., 0., 0.],
                      [0., 1., 0.],
                      [0., 0., 1.]], dtype=float)

        nds = Nodes()

        nd1 = Node(id, x, y, z)
        nd2 = Node(id + 1, y, z, x)
        nds.add(nd1)
        nds.add(nd2)
        assert len(nds.nodes.keys()) == 2

        nds = Nodes()
        nds.add([nd1, nd2])
        assert len(nds.nodes.keys()) == 2

        with pytest.raises(ValueError):
            nds = Nodes([1, 2])

        with pytest.raises(ValueError):
            nds = Nodes([nd1, nd1])

    def test__iadd__(self):
        id = 1
        x = 1000.
        y = 2000.
        z = 3000.
        nds = Nodes()
        nd1 = Node(id, x, y, z)
        nd2 = Node(id + 1, y, z, x)
        nds += nd1
        nds += nd2
        assert len(nds.nodes.keys()) == 2

    def test__getitem__(self):
        id = 1
        x = 1000.
        y = 2000.
        z = 3000.
        nds = Nodes()
        nd1 = Node(id, x, y, z)
        nd2 = Node(id + 1, y, z, x)
        nds += nd1
        nds += nd2
        assert nds[nd1.id] == nd1
        assert nds[nd2.id] == nd2
        with pytest.raises(ValueError):
            nds['1']
        with pytest.raises(ValueError):
            nds[3]

