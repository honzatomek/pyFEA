"""
Node and Nodes objects.

Node  - A Node object
Nodes - A colleciton of Nodes
"""

import os
import sys
from typing import Union
import numpy as np

SRC = os.path.realpath(__file__)
sys.path.append(os.path.dirname(os.path.dirname(SRC)))

from pyFEA.misc import Coordinate
from pyFEA.misc import ID

def id_decorator(func):
    def decorate(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            raise Exception(f'{str(func.__class__.__name__):s} ID ' + str(e)) from e
    return decorate


class Node(ID):
    """
    A Node object
    """
    def __init__(self, id, x: Union[float, np.float_, tuple, list, np.ndarray],
                 y: Union[float, np.float_]=None, z: Union[float, np.float_]=None,
                 csys: Union[int, np.int_]=0, label: Union[str, np.str_]=None):
        """
        Node constructor

        In:
            id    - node id, int
            x     - x coordinate if one float value, if tuple, list of np.ndarray
                    of 3 values, then all 3 coordinates
            y     - y coordinate
            z     - z coordinate
            csys  - definition coordinate system ID, int
            label - optional node label
        """
        super().__init__(id=id, label=label)
        try:
            self.__coors = Coordinate(x=x, y=y, z=z)
        except Exception as e:
            raise type(e)(f'Node ID {self.id:n} ' + str(e)) from e
        self.csys = csys

        self.__X_g = None

    @property
    def x(self) -> float:
        """
        Returns Node x coordinate (definition)
        """
        return self.__coors.x

    @x.setter
    def x(self, x: Union[float, np.float_, tuple, list, np.ndarray]):
        """
        Sets the Node x coordinate (or all 3, if a tuple/list/np.ndarray is fed in)

        In:
            x - Node x coordinate as a float or a tuple/list/np.ndarray of size 3 of all coordinates
        """
        try:
            self.__coors.x = x
        except Exception as e:
            raise type(e)(f'Node ID {self.id:n} ' + str(e)) from e

    @property
    def y(self) -> float:
        """
        Returns the Node y coordinate (definition).
        """
        return self.__coors.y

    @y.setter
    def y(self, y: Union[float, np.float_]):
        """
        Sets the Node y coordinate (definition).

        In:
            y - Node y coordinate
        """
        try:
            self.__coors.y = y
        except Exception as e:
            raise type(e)(f'Node ID {self.id:n} ' + str(e)) from e

    @property
    def z(self) -> float:
        """
        Returns the Node y coordinate (definition).
        """
        return self.__coors.z

    @z.setter
    def z(self, z: Union[float, np.float_]):
        """
        Sets the Node z coordinate (definition).

        In:
            z - Node z coordinate
        """
        try:
            self.__coors.z = z
        except Exception as e:
            raise type(e)(f'Node ID {self.id:n} ' + str(e)) from e

    @property
    def csys(self) -> int:
        """
        Returns the definition coordinate ID.
        """
        return self.__csys

    @csys.setter
    def csys(self, csys: Union[int, np.int_]):
        """
        Sets the definition coordinate ID.

        In:
            csys - Coordinate system ID as int, if 0 then global
        """
        if type(csys) not in (int, np.dtype(int)):
            raise ValueError(f'Node ID {str(self.id):s} coordinate system must be an int, not {str(type(csys)):s}.')
        elif csys < 0:
            raise ValueError(f'Node ID {self.id:n} coordinate system id {csys:n} must be >= 0.')
        else:
            self.__csys = csys

    @property
    def X_l(self) -> np.ndarray:
        """
        Returns the definition cooordinates as a numpy.ndarray
        """
        return np.array([self.x, self.y, self.z], dtype=float)

    def set_X_g(self, offset: Union[tuple, list, np.ndarray]=None,
                T: Union[tuple, list, np.ndarray]=None):
        """
        Computes the node coordinates in global CSYS.

        In:
            offset - coordinate system origin [x, y, z]
            T      - transformation matrix 3x3 of direction sines and cosines
        """
        if offset is None:
            offset = np.array([0., 0., 0.], dtype=float)
        if T is None:
            T = np.eye(3 , dtype=float)

        try:
            offset = Coordinate(offset).X
        except Exception as e:
            raise type(e)(f'Node ID {self.id:n} offset ' + str(e)) from e

        try:
            T = np.array(T, dtype=float)
        except Exception as e:
            raise ValueError(f'Node ID {str(self.id):s} global coordinates transformation matrix must be of type list or array, not {str(type(T)):s}.')

        if T.shape != (3, 3):
            raise ValueError(f'Node ID {str(self.id):s} global coordinates transformation matrix must be a matrix of shape (3, 3), not {str(T.shape):s}.')

        self.__X_g = T @ (self.X_l - offset)

    @property
    def X_g(self) -> np.ndarray:
        """
        Returns the node coordinates in global CSYS, set_X_g must be called first.
        """
        if self.__X_g is not None:
            return self.__X_g
        else:
            raise Exception(f'Node ID {self.id:n} has not yet been initialised.')



class Nodes:
    def __init__(self, nodes=None):
        self.__nodes = {}
        if nodes is not None:
            self.add(nodes)

    @property
    def nodes(self):
        return self.__nodes

    @nodes.setter
    def nodes(self, nodes):
        if type(nodes) not in (tuple, list):
            raise ValueError(f'Nodes must be a list, not {str(type(nodes)):s}.')
        else:
            for nd in nodes:
                if type(nd) is not Node:
                    raise ValueError(f'Node must be of type Node, not {str(type(nd)):s}.')
                else:
                    self.__nodes.setdefault(nd.id, nd)

    def distance(self, nd1, nd2):
        if type(nd1) not in (int, np.dtype(int)):
            raise ValueError(f'Node 1 must be of type int, not {str(type(nd1)):s}.')
        if type(nd2) not in (int, np.dtype(int)):
            raise ValueError(f'Node 2 must be of type int, not {str(type(nd2)):s}.')

        if nd1 not in self.__nodes.keys():
            raise ValueError(f'Node ID {nd1:n} not found in Nodes collection.')
        if nd2 not in self.__nodes.keys():
            raise ValueError(f'Node ID {nd2:n} not found in Nodes collection.')

        nd1 = self.__nodes[nd1].X_g
        nd2 = self.__nodes[nd2].X_g

        return np.linalg.norm(nd2 - nd1)

    def add(self, node):
        if type(node) is list:
            for nd in node:
                self.add(nd)
        else:
            if type(node) is not Node:
                raise ValueError(f'Node must be of type Node or a list of Nodes, not {str(type(node)):s}.')
            elif node.id in self.__nodes.keys():
                raise ValueError(f'Duplicite Node ID found {node.id:n}.')

            self.__nodes.setdefault(node.id, node)

    def __iadd__(self, node):
        self.add(node)
        return self

    def __getitem__(self, id):
        if type(id) not in (int, np.dtype(int)):
            raise ValueError(f'Node ID {str(id):s} must be an int, not {str(type(id)):s}.')
        elif id < 1:
            raise ValueError(f'Node ID {id:n} must be > 0.')
        elif id not in self.nodes.keys():
            raise ValueError(f'Node ID {id:n} not found.')
        else:
            return self.nodes[id]

    def init(self, csys):
        for nid in self.nodes.keys():
            cs = csys[self.nodes[nid].csys]
            self.nodes[nid].set_X_g(cs.origin, cs.T)

    @property
    def table(self):
        nids = []
        coors = []
        for nid, nd in self.__nodes.items():
            nids.append(nid)
            coors.append(nd.X_g)
        nids = np.array(nids, dtype=int)
        coors = np.array(coors, dtype=float)
        return nids, coors


if __name__ == '__main__':
    nd = Node(1, 1, 0., 0.)
    print(f'{nd.id = :n}')

