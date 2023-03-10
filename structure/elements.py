import math
import logging
import numpy as np

from misc.misc import eng, Data
from misc.errors import *
from structure.nodes import *
from structure.materials import *
from structure.properties import *


logger = logging.getLogger(__name__)


class Element(Data):
    _ids = set()
    _instances = set()
    _counter = 0
    _command = 'ELEMENT'
    _last_label_id = 0

    def __init__(self, id: int, mat: str, prop: str, nodes: [int], label: str = None):
        if not (isinstance(nodes, tuple) or isinstance(nodes, list)):
            raise AttributeError(f'{type(self).__name__:s} ID {id:n} attribute nodes ({str(nodes)} '
                                 f'must be either list or tuple of integers.')
        elif False in [type(node) == int for node in nodes]:
            raise ValueError(f'{type(self).__name__:s} ID {id:n} attribute nodes ({str(nodes)} '
                             f'must be either list or tuple of integers, not {type(nodes).__name__:s}.')

        if isinstance(mat, int):
            self._mat = Material.getID(mat).label
        else:
            self._mat = mat

        if isinstance(prop, int):
            self._prop = Property.getID(prop).label
        else:
            self._prop = prop

        self._nodes = nodes
        super(Element, self).__init__(id, label)

    def __del__(self):
        super(Element, self).__del__()

    def __str__(self):
        message = f'  {self.id:9n} {self._mat:16s} {self._prop:16s}'

        if len(self._nodes) > 0:
            message += '  : '

        for node in self._nodes:
            message += f' {node:9n}'

        if self.label is not None:
            if ' ' in self.label:
                message += f" :  '{self.label:s}'"
            else:
                message += f' :  {self.label:s}'

        return message

    def __repr__(self):
        message = f"{type(self).__name__:s}(id={self.id:n}, mat={self._mat:s}, " \
                  f"prop={self._prop:s}, nodes={str(self._nodes):s}"
        if self.label is not None:
            message += f", label='{self.label:s}')"
        else:
            message += f")"
        return message

    @property
    def prop(self):
        return Property.getLabel(self._prop)

    @property
    def mat(self):
        return Material.getLabel(self._mat)

    @property
    def nodes(self):
        return [Node.getID(node) for node in self._nodes]

    @property
    def node_ids(self):
        return list(self._nodes)

    def node(self, num: int):
        """

        :param num: Zero based index of Element node
        :return:
        """
        if num < 0 or num >= len(self._nodes):
            raise ValueError(f'{type(self).__name__:s} Node index {num:n} must be in range '
                             f'0 - {len(self._nodes):n}.')
        return Node.getID(self._nodes[num])


class Rod2D(Element):
    """
    Linear Rod Element in 2D
    :param id:       Unique Element ID
    :param mat:      Material Object label
    :param prop:     Property Object label
    :param endA:     Node object of element start
    :param endB:     Node object of element end
    """
    _counter = 0

    def __init__(self, id: int, mat: str, prop: str, nodes: [int], label: str = None):
        super(Rod2D, self).__init__(id, mat, prop, nodes, label)

    def __del__(self):
        super(Rod2D, self).__del__()

    def length(self):
        nds = self.nodes
        x1 = nds[0].coor
        x2 = nds[1].coor
        length = math.sqrt((x2[0] - x1[0]) ** 2. + (x2[1] - x1[1]) ** 2.)
        return length

    def transmatrix(self):
        """
        Function computes transformation matrix (GCS -> LCS) of beam in 2d
        (Rl = T * Rg)
        :return:   t  - beam transformation matrix in 2D (6, 6)
        """
        nds = self.nodes
        x1 = nds[0].coor
        x2 = nds[1].coor
        length = math.sqrt((x2[0] - x1[0]) ** 2. + (x2[1] - x1[1]) ** 2.)
        c = (x2[0] - x1[0]) / length
        s = (x2[1] - x1[1]) / length

        tt = np.array([[c, s, 0.],
                       [-s, c, 0.],
                       [0., 0., 1.]], dtype=float)

        t = np.zeros((6, 6), dtype=float)
        t[:3, :3] = tt
        t[3:, 3:] = tt

        return t

    def stiffness_lcs(self):
        """
        Returns Rod2D element stiffness matrix in LCS (2D)
        :return: kl - element stiffness matrix in LCS
        """
        l = self.length()
        EA = self.mat.E() * self.prop.A
        kuu = EA / l * np.array([[1., -1.],
                                 [-1., 1.]], dtype=float)

        kl = np.zeros((6, 6), dtype=float)
        kl[[0, 3], [[0], [3]]] = kuu

        return kl

    def stiffness_gcs(self):
        """
        Returns Element stiffness matrix in GCS (2D)
        :return: ke - element stiffness matrix in GCS
        """
        # element stiffness matrix in LCS
        kl = self.stiffness_lcs()
        # element transformation matrix LCS -> GCS
        t = self.transmatrix()
        # transformation LCS -> GCS
        ke = t.T @ kl @ t

        return ke

    def mass_lumped_lcs(self):
        """
        Function to compute lumped mass matrix of Rod element in LCS (2D)
        :return: mll - Rod element lumped mass matrix in LCS 2D (6, 6)
        """
        # structural mass
        sm = self.mat.ro() * self.prop.A * self.length()
        # nonstructural mass
        nm = self.length() * self.prop.nsm
        # local lumped element mass matrix
        mll = np.zeros((6, 6), dtype=float)
        mll[0][0] = (sm + nm) / 2.
        mll[3][3] = (sm + nm) / 2.
        return mll

    def mass_consistent_lcs(self):
        """
        Function to compute consistent mass matrix of Rod2D element in LCS (2D)
        :return: mcl - Rod2D element lumped mass matrix in LCS 2D (6, 6)
        """
        # structural mass
        sm = self.mat.ro() * self.prop.A * self.length()
        # nonstructural mass
        nm = self.length() * self.prop.nsm
        # local consistent element mass matrix
        m11 = (sm + nm) * np.array([[1. / 3., 1. / 6.],
                                    [1. / 6., 1. / 3.]], dtype=float)

        mcl = np.zeros((6, 6), dtype=float)
        mcl[[0, 3], [[0], [3]]] = m11

        return mcl

    def mass_gcs(self, mi: float = 0.5):
        """
        Returns Element Mass matrix in GCS (2D)
        :param mi:  weighted average coefficient for lumped and consistent mass matrix (1.0 = lumped, 0.0 = consistent)
        :return: ke - element mass matrix in GCS
        """
        ml = (1. - mi) * self.mass_consistent_lcs() + mi * self.mass_lumped_lcs()
        t = self.transmatrix()
        # transformation LCS -> GCS
        me = t.T @ ml @ t

        return me


class Bar2D(Rod2D):
    """
    Bar Element, linear interpolation (Kirchhoff, 2D)
    """
    _counter = 0

    def __init__(self, id: int, mat: str, prop: str, nodes: [int],
                 releaseA: tuple = (0, 0, 0), releaseB: tuple = (0, 0, 0),
                 label: str = None):
        super(Bar2D, self).__init__(id, mat, prop, nodes, label)
        self.releases = [[bool(r) for r in releaseA], [bool(r) for r in releaseB]]

    def __del__(self):
        super(Bar2D, self).__del__()

    def __str__(self):
        message = f'  {self.id:9n} {self._mat:16s} {self._prop:16s}'

        if len(self.nodes) > 0:
            message += '  : '
        for node in self.nodes:
            message += f' {node:9n}'

        dofs = ''
        for release in self.releases:
            dofs += ''.join([str(i + 1) for i in range(len(release)) if release[i]])
            dofs += ','
        if len(dofs) > 0:
            message += f'  :  {dofs[:-1]:7s}'

        if self.label is not None:
            if ' ' in self.label:
                message += f" :  '{self.label:s}'"
            else:
                message += f' :  {self.label:s}'
        return message

    def __repr__(self):
        message = f"{type(self).__name__:s}(id={self.id:n}, mat={self._mat:s}, " \
                  f"prop={self._prop:s}, nodes={str(self.nodes):s}, " \
                  f"releaseA={str(self.releases[0])}, releaseB={str(self.releases[1])}"
        if self.label is not None:
            message += f", label='{self.label:s}')"
        else:
            message += f")"
        return message

    def stiffness_lcs(self):
        """
        Returns Bar2D element stiffness matrix in LCS (Kirchhoff, 2D)
        :return: kl - element stiffness matrix in LCS
        """
        l = self.length()
        l2 = l * l
        l3 = l2 * l
        EA = self.mat.E() * self.prop.A
        EI = self.mat.E() * self.prop.I_11
        kuu = EA / l * np.array([[1., -1.],
                                 [-1., 1.]], dtype=float)
        kww = EI / l3 * np.array([[12., -12.],
                                  [-12., 12.]], dtype=float)
        kwf = EI / l2 * np.array([[-6., -6.],
                                  [6., 6.]], dtype=float)
        kff = EI / l * np.array([[4., 2.],
                                 [2., 4.]], dtype=float)

        kl = np.zeros((6, 6), dtype=float)
        kl[[0, 3], [[0], [3]]] = kuu
        kl[[1, 4], [[1], [4]]] = kww
        kl[[1, 4], [[2], [5]]] = kwf
        kl[[2, 5], [[1], [4]]] = kwf.T
        kl[[2, 5], [[2], [5]]] = kff

        return kl

    def mass_lumped_lcs(self):
        """
        Function to compute lumped mass matrix of bar element in LCS (2D)
        :return: mll - bar element lumped mass matrix in LCS 2D (6, 6)
        """
        # structural mass
        sm = self.prop.A * self.length() * self.mat.ro()
        # nonstructural mass
        nm = self.length() * self.prop.nsm
        # local lumped element mass matrix
        mll = (sm + nm) / 2. * np.eye(6, dtype=float)
        mll[2][2] = 0.
        mll[5][5] = 0.
        return mll

    def mass_consistent_lcs(self):
        """
        Function to compute consistent mass matrix of bar element in LCS (2D)
        :return: mcl - bar element lumped mass matrix in LCS 2D (6, 6)
        """
        # structural mass
        sm = self.prop.A * self.length() * self.mat.ro()
        # nonstructural mass
        nm = self.length() * self.prop.nsm
        # local consistent element mass matrix
        m11 = (sm + nm) * np.array([[1. / 3., 1. / 6.],
                                    [1. / 6., 1. / 3.]], dtype=float)

        mcl = np.zeros((6, 6), dtype=float)
        mcl[[0, 3], [[0], [3]]] = m11
        mcl[[1, 4], [[1], [4]]] = m11

        return mcl

    def beam_mass_consistent_lcs(self):
        """
        Function to compute consistent mass matrix of bar element in LCS (2D)
        :return: mcl - beam element lumped mass matrix in LCS 2D (6, 6)
        """
        # structural mass
        sm = self.prop.A * self.length() * self.mat.ro()
        # nonstructural mass
        nm = self.length() * self.prop.nsm
        # local consistent element mass matrix
        muu = (sm + nm) * np.array([[1. / 3., 1. / 6.],
                                    [1. / 6., 1. / 3.]], dtype=float)
        mww = (sm + nm) / 420. * np.array([[156., 54.],
                                           [54., 156.]], dtype=float)
        mwf = (sm + nm) * self.length() / 420. * np.array([[22., -13.],
                                                           [-13., -22.]], dtype=float)
        mff = (sm + nm) * (self.length() ** 2) * np.array([[4., -3.],
                                                           [-3., 4.]], dtype=float)

        mcl = np.zeros((6, 6), dtype=float)
        mcl[[0, 3], [[0], [3]]] = muu
        mcl[[1, 4], [[1], [4]]] = mww
        mcl[[1, 4], [[2], [5]]] = mwf
        mcl[[2, 5], [[1], [4]]] = mwf.T
        mcl[[2, 5], [[2], [5]]] = mff

        return mcl


class Elements(DataSet):
    """
    Class for collection of Element objects
    """
    _type = Element
    _ids = set()
    _instances = set()
    _counter = 0
    _command = 'ELEMENT'
    _last_label_id = 0

    def __init__(self, id: int = None, label: str = None):
        super(Elements, self).__init__(Element, id, label)

    def add_Bar2D(self, id: int, mat: str, prop: str,
                  nodes: [int], releaseA: [int], releaseB: [int],
                  label: str = None):
        self._add_object(Bar2D(id, mat, prop, nodes, releaseA, releaseB, label))

    def add_Rod2D(self, id: int, mat: str, prop: str,
                  nodes: [int], label: str = None):
        self._add_object(Rod2D(id, mat, prop, nodes, label))


if __name__ == '__main__':
    nds = Nodes()
    for i in range(3):
        nds.add(i + 1, [i * 500., 0.])
    print(str(nds))
    m = LinearISOElastic('steel', 7.85E-9, 210.0E6, 0.3, 1.2E-5)
    print(str(m))
    p = CrossSectionBeam2D(1, 'beam', 2124.0, 3492243.0, 72755.0, 756.0, 0.0)
    print(str(p))
    els = Elements()
    for i in range(2):
        els.add_Rod2D(m.label, p.label, [i + 1, i + 2])
        els.add_Bar2D(m.label, p.label, [i + 1, i + 2],
                      [0, 0, 0], [0, 0, 0])
    print(str(els))
    pass
