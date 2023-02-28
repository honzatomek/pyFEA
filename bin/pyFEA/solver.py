#!/usr/bin/python3

import os
import sys
import numpy as np
from typing import Union
from math import floor

_INT = np.int32
_FLOAT = np.float64


def hex_beam(width, depth, height, m, n, o, offset=1000001):
  dx = width / m
  dy = depth / n
  dz = height / o

  # nodes and coors
  coors = []
  displacement = []
  # nodes = []
  for k in range(o+1):
    for j in range(n+1):
      for i in range(m+1):
        # nodes.append(i + (m+1)*j + (m+1)*(n+1)*k)
        coors.append([i*dx, j*dy, k*dz])
        displacement.append([i*dx, j*dy, np.sin(k * dz / height * np.pi)])

  # offset indexes to start numbering from 1 000 000
  coors = {i + offset: np.array(coors[i], dtype=_FLOAT) for i in range(len(coors))}
  displacement = {i + offset: np.array(displacement[i], dtype=_FLOAT) for i in range(len(displacement))}

  # connectivity hex8
  lme = []
  for k in range(o):
    for j in range(n):
      for i in range(m):
        n1 = i + (m+1)*j + (m+1)*(n+1)*k
        n2 = i + (m+1)*j + (m+1)*(n+1)*k + 1
        n3 = i + (m+1)*(j+1) + (m+1)*(n+1)*k + 1
        n4 = i + (m+1)*(j+1) + (m+1)*(n+1)*k
        n5 = i + (m+1)*j + (m+1)*(n+1)*(k+1)
        n6 = i + (m+1)*j + (m+1)*(n+1)*(k+1) + 1
        n7 = i + (m+1)*(j+1) + (m+1)*(n+1)*(k+1) + 1
        n8 = i + (m+1)*(j+1) + (m+1)*(n+1)*(k+1)
        # lme.extend([[n1,n2,n3,n4],[n5,n6,n7,n8],[n1,n2,n6,n5],[n2,n3,n7,n6],[n3,n4,n8,n7],[n4,n1,n5,n8]])
        lme.append([n1, n2, n3, n4, n5, n6, n7, n8])

  # offset indexes to start numbering from 1 000 000
  # lme = {i + offset: {'nodes': np.array(lme[i], dtype=_INT) + offset, 'material': 'steel'} for i in range(len(lme))}
  lme = {i + offset: {'nodes': np.array(lme[i], dtype=_INT) + offset, 'material': 1001} for i in range(len(lme))}

  return coors, {'HEX8': lme}, displacement


def prepare_model(x: float=10., y: float=10., z: float=500.,
                  nx: int=1, ny: int=1, nz: int=5, offset: int=1000001):
    # x, y, z = 10., 10., 500.
    # nx, ny, nz = 1, 1, 5
    # offset = 1000001

    nodes, elements, data = hex_beam(x, y, z, nx, ny, nz, offset)

    # get node ids with 0. z coordinates
    constraints = {nid: np.array([1, 1, 1], dtype=_INT) for nid, coors in nodes.items() if coors[2] == 0.}
    constraints = {i + offset: {'node': nid, 'constrained': constraints[nid]} for i, nid in enumerate(constraints.keys())}

    # get node ids with max z coordinates
    loads = {nid: np.array([10., 0., 0.], dtype=_FLOAT) for nid, coors in nodes.items() if coors[2] == z}
    loads = {i + offset: {'node': lid, 'load': loads[lid]} for i, lid in enumerate(loads.keys())}

    materials = {1001: {'name': 'steel',
                           'E': _FLOAT(210000.),
                           'nu': _FLOAT(0.3),
                           'rho': _FLOAT(7.85e-9),
                           'alpha': _FLOAT(1.2e-5)},
                 2001: {'name': 'aluminum',
                              'E': _FLOAT(70000.),
                              'nu': _FLOAT(0.3),
                              'rho': _FLOAT(2.70e-9),
                              'alpha': _FLOAT(1.5e-5)}}

    # print(f'{nodes = }')
    # print(f'{elements[1000001] = }')
    # print(f'{constraints = }')
    # print(f'{loads = }')
    # print(f'{materials = }')

    data = {'Normal Mode': {'NODAL': {101: {'id': 101,
                                            'model type': 'Structural',
                                            'analysis type': 'Normal Mode',
                                            'data characteristic': '3 DOF Global Translation Vector',
                                            'specific data type': 'Displacement',
                                            'data type': 'Real',
                                            'values per node': 3,
                                            'mode': 1,
                                            'frequency': 10.0,
                                            'values': data}}}}

    return nodes, elements, constraints, loads, materials, data


class Node:
    def __init__(self, nid: _INT, x: _FLOAT, y: _FLOAT, z: _FLOAT):
        self.id = nid
        self.x = x
        self.y = y
        self.z = z

    def __getitem__(self, coor: int):
        return self.coors[coor]

    @property
    def id(self) -> _INT:
        return self.__id

    @id.setter
    def id(self, id: _INT):
        self.__id = _INT(id)

    @property
    def x(self) -> _FLOAT:
        return self.__x

    @x.setter
    def x(self, x: _FLOAT):
        self.__x = _FLOAT(x)

    @property
    def y(self) -> _FLOAT:
        return self.__y

    @y.setter
    def y(self, y: _FLOAT):
        self.__y = _FLOAT(y)

    @property
    def z(self) -> _FLOAT:
        return self.__z

    @z.setter
    def z(self, z: _FLOAT):
        self.__z = _FLOAT(z)

    @property
    def coors(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=_FLOAT)


class Material:
    def _create_C(self):
        E = self.E
        nu = self.nu
        C = np.array([[1.0 - nu, nu, nu, 0.0, 0.0, 0.0],
                      [nu, 1.0 - nu, nu, 0.0, 0.0, 0.0],
                      [nu, nu, 1.0 - nu, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, (1.0 - 2.0 * nu) / 2.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, (1.0 - 2.0 * nu) / 2.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, (1.0 - 2.0 * nu) / 2.0]], dtype=_FLOAT)
        C *= E / ((1.0 + nu) * (1.0 - 2.0 * nu))
        self.__C = C
        return self.__C

    def __init__(self, mid: _INT, name: str, E: _FLOAT, nu: _FLOAT, rho: _FLOAT, alpha: _FLOAT):
        self.__C = None
        self.id = mid
        self.name = name
        self.E = E
        self.nu = nu
        self.rho = rho
        self.alpha = alpha

    @property
    def id(self) -> _INT:
        return self.__id

    @id.setter
    def id(self, id: _INT):
        self.__id = _INT(id)

    @property
    def name(self) -> str:
        return self.__name

    @name.setter
    def name(self, name: str):
        self.__name = str(name)

    @property
    def E(self) -> _FLOAT:
        return self.__E

    @E.setter
    def E(self, E: _FLOAT):
        self.__E = _FLOAT(E)
        if self.__C is not None:
            self._create_C()

    @property
    def nu(self) -> _FLOAT:
        return self.__nu

    @nu.setter
    def nu(self, nu: _FLOAT):
        self.__nu = _FLOAT(nu)
        if self.__C is not None:
            self._create_C()

    @property
    def rho(self) -> _FLOAT:
        return self.__rho

    @rho.setter
    def rho(self, rho: _FLOAT):
        self.__rho = _FLOAT(rho)

    @property
    def alpha(self) -> _FLOAT:
        return self.__alpha

    @alpha.setter
    def alpha(self, alpha: _FLOAT):
        self.__alpha = _FLOAT(alpha)

    @property
    def C(self) -> np.ndarray:
        if self.__C is not None:
            return self.__C
        else:
            return self._create_C()


class HEX8:
    _GAUSS = 3
    _DOFS = 3
    _NODES = 8

    def __init__(self, eid: _INT, nodes: Union[tuple, list], material: Material):
        self.id = eid
        self.nodes = nodes
        self.material = material

    @property
    def id(self) -> _INT:
        return self.__id

    @id.setter
    def id(self, id: _INT):
        self.__id = _INT(id)

    @property
    def nodes(self):
        return self.__nodes

    @nodes.setter
    def nodes(self, nodes: Union[tuple, list]):
        self.__nodes = list()
        if type(nodes) is dict:
            if len(nodes.keys()) != self._NODES:
                raise ValueError(f'Wrong number of nodes for HEX8 element ({len(nodes.keys()):n}).')
            for nid, coors in nodes.items():
                self.__nodes.append(Node, nid, *coors)
        else:
            if len(nodes) != self._NODES:
                raise ValueError(f'Wrong number of nodes for HEX8 element ({len(nodes):n}).')
            for node in nodes:
                self.__nodes.append(node)

    @property
    def coors(self) -> np.ndarray:
        self.__coors = []
        for node in self.nodes:
            self.__coors.append(node.coors)
        self.__coors = np.array(self.__coors)
        return self.__coors

    @property
    def material(self) -> _INT:
        return self.__material

    @material.setter
    def material(self, material: _INT):
        if type(material) is Material:
            self.__material = material
        else:
            self.__material = Material(material['id'],
                                       material['name'], material['E'],
                                       material['nu'], material['rho'],
                                       material['alpha'])

    @property
    def domain(self) -> np.ndarray:
        domain = np.array([[-1., -1., -1.],
                           [ 1., -1., -1.],
                           [ 1.,  1., -1.],
                           [-1.,  1., -1.],
                           [-1., -1.,  1.],
                           [ 1., -1.,  1.],
                           [ 1.,  1.,  1.],
                           [-1.,  1.,  1.]], dtype=_FLOAT)
        return domain

    @property
    def gauss(self) -> tuple:
        g_points, g_weights = np.polynomial.legendre.leggauss(self._GAUSS)
        integration_points = []
        interpolation_weights = []
        for i, xi in enumerate(g_points):
            for j, eta in enumerate(g_points):
                for k, mu in enumerate(g_points):
                    integration_points.append([xi, eta, mu])
                    interpolation_weights.append(g_weights[i] * g_weights[j] * g_weights[k])
        integration_points = np.array(integration_points, dtype=_FLOAT)
        interpolation_weights = np.array(interpolation_weights, dtype=_FLOAT)
        return integration_points.T, interpolation_weights

    def psi(self, integration_points: np.ndarray=None) -> np.ndarray:
        """
        Shape functions in natural coordinates

        In:
            integration_points - np.ndarray of natural coordinates from -1 to 1
                                 [[ xi_1,  xi_2, ... ,  xi_n],
                                  [eta_1, eta_2, ... , eta_n],
                                  [ mu_1,  mu_2, ... ,  mu_n]]
        """
        xi = intgration_points[0]
        eta = intgration_points[0]
        mu = intgration_points[0]
        psi = np.zeros((self._NODES, integration_points.shape[1]), dtype=_FLOAT)
        for i in range(8):
            psi[i] = 1/8 * (1 + xi * self.domain[i, 0])  \
                         * (1 + eta * self.domain[i, 1]) \
                         * (1 + mu * self.domain[i, 2])
        return psi.T

    def psi_g(self, psi: np.ndarray):
        """
        Shape functions in Global coordinates
        """
        return psi @ self.coors

    def dpsi(self, integration_points: np.ndarray) -> np.ndarray:
        """
        Shape Function Derivatives
        """
        xi = intgration_points[0]
        eta = intgration_points[0]
        mu = intgration_points[0]
        dpsi = 1/8*np.array([[( eta-1)*( 1-mu), ( xi-1)*(1-mu),-(1-xi)*(1-eta)],
                             [( 1+eta)*( 1-mu), ( 1+xi)*(1-mu),-(1+xi)*(1+eta)],
                             [( 1-eta)*( 1-mu), (-1-xi)*(1-mu),-(1+xi)*(1-eta)],
                             [(-1-eta)*( 1-mu), ( 1-xi)*(1-mu),-(1-xi)*(1+eta)],
                             [( 1-eta)*(-1-mu),-( 1-xi)*(1+mu), (1-xi)*(1-eta)],
                             [( 1-eta)*( 1+mu),-( 1+xi)*(1+mu), (1+xi)*(1-eta)],
                             [( 1+eta)*( 1+mu), ( 1+xi)*(1+mu), (1+xi)*(1+eta)],
                             [-(1+eta)*( 1+mu), ( 1-xi)*(1+mu), (1-xi)*(1+eta)]])
        return dpsi.T

    def J(self, dpsi: np.ndarray) -> np.ndarray:
        """
        Jacobian Matrix
        """
        return dpsi @ self.coors

    def dJ(self, jacobi: np.ndarray) -> np.ndarray:
        """
        Jacobian Matrix Determinant
        """
        return np.linalg.det(jacobi)

    def iJ(self, jacobi: np.ndarray) -> np.ndarray:
        """
        Inverse Jacobian Matrix
        """
        return np.linalg.inv(jacobi)

    def dpsi_g(self, ijacobi: np.ndarray, dpsi: np.ndarray) -> np.ndarray:
        """
        Shape Function Derivatives in global coordinates
        """
        return ijacobi @ dpsi

    def N(self, psi: np.ndarray, integration_point: _INT) -> np.ndarray:
        p = psi[integration_point,:]
        o = np.zeros(p.shape, dtype=_FLOAT)
        N = np.array([np.column_stack((p, o, o)).flatten(),
                      np.column_stack((o, p, o)).flatten(),
                      np.column_stack((o, o, p)).flatten()], dtype=_FLOAT)
        return N

    def B(self, dpsi_g: np.ndarray, integration_point: _INT) -> np.ndarray:
        dpx = dpsi_g[integration_point,0,:]
        dpy = dpsi_g[integration_point,1,:]
        dpz = dpsi_g[integration_point,2,:]
        o = np.zeros(dpx.shape, dtype=_FLOAT)
        # component-wise dof ordering (x1, .. , y1, .. y4, z1, .. , z4)
        # B = np.array([
        #   [*dpsi_g[i, 0, :],               *o,               *o],
        #   [              *o, *dpsi_g[i, 1, :],               *o],
        #   [              *o,               *o, *dpsi_g[i, 2, :]],
        #   [*dpsi_g[i, 2, :],               *o, *dpsi_g[i, 0, :]],
        #   [              *o, *dpsi_g[i, 2, :], *dpsi_g[i, 1, :]],
        #   [*dpsi_g[i, 1, :], *dpsi_g[i, 0, :],               *o]], dtype=float)
        # node wise dof ordering (x1, y1, z1, .. , x4, y4, z4)
        B = np.array([np.column_stack((dpx,   o,   o)).flatten(),
                      np.column_stack((  o, dpy,   o)).flatten(),
                      np.column_stack((  o,   o, dpz)).flatten(),
                      np.column_stack((dpy, dpx,   o)).flatten(),
                      np.column_stack((  o, dpz, dpy)).flatten(),
                      np.column_stack((dpz,   o, dpx)).flatten()], dtype=_FLOAT)
        return B

    @property
    def Ke(self) -> np.ndarray:
        """
        Returns Element stiffness matrix in global coordinates
        """
        ip, iw = self.gauss
        psi = self.psi(ip)
        dpsi = self.dpsi(ip)
        J = self.J(dpsi)
        dJ = self.dJ(J)
        iJ = self.iJ(J)
        dpsi_g = self.dpsi_g(iJ, dpsi)

        Ke = np.zeros((self._NODES * self._DOFS, self._NODES * self._DOFS), dtype=_FLOAT)
        C = self.material.C

        for i in range(ip.shape[1]):
            B = self.B(dpsi_g, i)
            Ke += (B.T @ C @ B) * dJ[i] * iw[i]

        return Ke

    @property
    def Me(self) -> np.ndarray:
        """
        Returns Element Mass Matrix in global coordinates
        """
        ip, iw = self.gauss
        psi = self.psi(ip)
        dpsi = self.dpsi(ip)
        J = self.J(dpsi)
        dJ = self.dJ(J)

        Me = np.zeros((self._NODES * self._DOFS, self._NODES * self._DOFS), dtype=_FLOAT)

        for i in range(ip.shape[1]):
            N = self.N(psi, i)
            Me += self.material.rho * (N.T @ N) * dJ[i] * iw[i]

        return Me

    def Fe(self, F: np.ndarray) -> np.ndarray:
        """
        Returns the vector of volumetric forces in global coordinates
        """
        ip, iw = self.gauss
        psi = self.psi(ip)
        dpsi = self.dpsi(ip)
        J = self.J(dpsi)
        dJ = self.dJ(J)

        Fe = np.zeros((self._NODES * self._DOFS, 1), dtype=_FLOAT)

        for i in range(ip.shape[1]):
            N = self.N(psi, i)
            Fe += (N.T @ F) * dJ[i] * iw[i]

        return Fe


_ELEMENTS = {'HEX8': HEX8}


class Load:
    def __init__(self, lid: _INT, nid: _INT, Fx: _FLOAT, Fy: _FLOAT, Fz: _FLOAT):
        self.id = lid
        self.node = nid
        self.Fx = Fx
        self.Fy = Fy
        self.Fz = Fz

    @property
    def F(self) -> np.array:
        return np.array([self.Fx, self.Fy, self.Fz], dtype=_FLOAT)

    @property
    def Fr(self) -> _FLOAT:
        return _FLOAT(np.linalg.norm(self.F))

    @property
    def id(self) -> _INT:
        return self.__id

    @id.setter
    def id(self, id: _INT):
        self.__id = _INT(id)

    @property
    def node(self) -> _INT:
        return self.__node

    @node.setter
    def node(self, id: _INT):
        self.__node = _INT(id)

    @property
    def Fx(self) -> _FLOAT:
        return self.__Fx

    @Fx.setter
    def Fx(self, Fx: _FLOAT):
        self.__Fx = _FLOAT(Fx)

    @property
    def Fy(self) -> _FLOAT:
        return self.__Fy

    @Fy.setter
    def Fy(self, Fy: _FLOAT):
        self.__Fy = _FLOAT(Fy)

    @property
    def Fz(self) -> _FLOAT:
        return self.__Fz

    @Fz.setter
    def Fz(self, Fz: _FLOAT):
        self.__Fz = _FLOAT(Fz)


class Constraint:
    def __init__(self, cid: _INT, nid: _INT, cx: bool, cy: bool, cz: bool,
                       px: _FLOAT=0., py: _FLOAT=0., pz: _FLOAT=0.):
        self.id = cid
        self.node = nid
        self.__prescribed = False
        self.cx = cx
        self.cy = cy
        self.cz = cz

        self.px = px
        self.py = py
        self.pz = pz

    @property
    def prescribed(self) -> bool:
        return self.__prescribed

    @property
    def id(self) -> _INT:
        return self.__id

    @id.setter
    def id(self, id: _INT):
        self.__id = _INT(id)

    @property
    def node(self) -> _INT:
        return self.__node

    @node.setter
    def node(self, id: _INT):
        self.__node = _INT(id)

    @property
    def cx(self) -> bool:
        return self.__cx

    @cx.setter
    def cx(self, cx: bool):
        self.__cx = bool(cx)

    @property
    def cy(self) -> bool:
        return self.__cy

    @cy.setter
    def cy(self, cy: bool):
        self.__cy = bool(cy)

    @property
    def cz(self) -> bool:
        return self.__cz

    @cz.setter
    def cz(self, cz: bool):
        self.__cz = bool(cz)

    @property
    def px(self) -> _FLOAT:
        return self.__px

    @px.setter
    def px(self, px: _FLOAT):
        if self.cx:
            self.__px = None
        else:
            self.__px = _FLOAT(px)
            if self.__px != 0.:
                self.__prescribed = True

    @property
    def py(self) -> _FLOAT:
        return self.__py

    @py.setter
    def py(self, py: _FLOAT):
        if self.cy:
            self.__py = None
        else:
            self.__py = _FLOAT(py)
            if self.__py != 0.:
                self.__prescribed = True

    @property
    def pz(self) -> _FLOAT:
        return self.__pz

    @pz.setter
    def pz(self, pz: _FLOAT):
        if self.cz:
            self.__pz = None
        else:
            self.__pz = _FLOAT(pz)
            if self.__pz != 0.:
                self.__prescribed = True


class Model:
    def __init__(self, nodes: dict, elements: dict, constraints: dict, loads: dict, materials: dict):
        self.nodes = nodes
        self.materials = materials
        self.elements = elements
        self.constraints = constraints
        self.loads = loads

    @property
    def nodes(self) -> dict:
        return self.__nodes

    @property
    def nodeids(self) -> list:
        return list(self.nodes.keys())

    @nodes.setter
    def nodes(self, nodes):
        self.__nodes = dict()
        for nid, node in nodes.items():
            if type(node) is not Node:
                self.__nodes[nid] = Node(nid, *node)
            else:
                self.__nodes[nid] = node

    @property
    def materials(self) -> dict:
        return self.__materials

    @materials.setter
    def materials(self, materials: dict):
        self.__materials = dict()
        for mid, material in materials.items():
            if type(material) is not Material:
                self.__materials[mid] = Material(mid,
                                                 material['name'],
                                                 material['E'],
                                                 material['nu'],
                                                 material['rho'],
                                                 material['alpha'])
            else:
                self.__materials[mid] = material

    @property
    def elements(self) -> dict:
        return self.__elements

    @property
    def elements_dict(self) -> dict:
        els = dict()
        for etype, elements, in self.__elements.items():
            if etype not in els.keys():
                els.setdefault(etype, {})
            for eid, e in elements.items():
                # els[etype][eid] = [n.id for n in e.nodes]
                els[etype][eid] = dict()
                # TODO:
                # els[etype][eid]['property'] = 1
                els[etype][eid]['material'] = e.material.id
                els[etype][eid]['nodes'] = [n.id for n in e.nodes]
        return els

    @elements.setter
    def elements(self, elements: dict):
        self.__elements = dict()
        if type(elements) in (list, tuple):
            for element in elements:
                if str(type(element)) in _ELEMENTS.keys():
                    if str(type(element)) not in self.__elements.keys():
                        self.__elements[str(type(element))] = dict()
                    self.__elements[str(type(element))][element.id] = element
                else:
                    raise ValueError(f'Element must be of type {", ".join(list(_ELEMENTS.keys())):s}, not {str(type(element)):s}.')
        elif type(elements) is dict:
            for etype, els in elements.items():
                self.__elements[etype] = dict()
                for eid, element in els.items():
                    if str(type(element)) not in _ELEMENTS.keys():
                        self.__elements[etype][eid] = _ELEMENTS[etype](eid,
                                                                nodes=[self.nodes[nid] for nid in element['nodes']],
                                                                material=self.materials[element['material']])
                    else:
                        self.__elements[etype][eid] = element
        else:
            raise ValueError('Elements must be supplied either as a tuple, list or dict.')

    @property
    def constraints(self) -> dict:
        return self.__constraints

    @constraints.setter
    def constraints(self, constraints: dict):
        self.__constraints = dict()
        for cid, constraint in constraints.items():
            if type(constraint) is Constraint:
                self.__constraints[constraint.id] = constraint
            else:
                self.__constraints[cid] = Constraint(cid, constraint['node'], *constraint['constrained'])

    @property
    def loads(self) -> dict:
        return self.__loads

    @loads.setter
    def loads(self, loads: dict):
        self.__loads = dict()
        for lid, load in loads.items():
            if type(load) is Load:
                self.__loads[load.id] = load
            else:
                self.__loads[lid] = Load(lid, load['node'], *load['load'])

    @property
    def stat(self):
        message = f'Nodes          : {len(self.nodeids):10n}\n'
        for etype, elements in self.elements.items():
            message += f'{etype:5s} Elements : {len(elements):10n}\n'
        message += f'Constraints    : {len(self.constraints):10n}\n'
        message += f'Loads          : {len(self.loads):10n}\n'
        message += f'Materials      : {len(self.materials):10n}'
        return message


if __name__ == '__main__':
    # model = Model(*prepare_model())
    # model = Model(*prepare_model(100., 100., 500., 10, 10, 20, 1001))
    nodes, elements, constraints, loads, materials, data = prepare_model(100., 100., 500., 10, 10, 20, 1001)
    model = Model(nodes, elements, constraints, loads, materials)
    print(f'{model.stat:s}')
    print(f'{data.keys() = }')

    # sys.path.append('/home/honza/Programming/pyFeaModel/bin/')
    from meshIO import unv

    filename = './meshIO/res/test_hex_single.unv'
    filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), filename)

    unv.write(filename, model.nodes, model.elements_dict, data, precision='single')
    nodes, elements, results = unv.read(filename)

    filename = './meshIO/res/test_hex_double.unv'
    filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), filename)
    unv.write(filename, model.nodes, model.elements_dict, data, precision='double')
    nodes, elements, results = unv.read(filename)

    # filename = os.path.splitext(filename)[0] + '_check.unv'
    # unv.write(filename, nodes, elements)
    print('[+] Done.')

    # print(f'{model.elements.keys() = }')
    # hex8 = model.elements['HEX8']
    # material = hex8[list(hex8.keys())[0]].material.name
    # print(f'{material}')


