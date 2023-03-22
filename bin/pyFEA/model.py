
import os
import sys
import typing
import pdb
import numpy as np

__PATH__ = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(__PATH__)

import pyFEA.utils as utils

_DEFAULT_SITUATION =   "DEFSITU"
_DEFAULT_STRUCTURE =   "DEFSTRU"
_DEFAULT_SYSTEM =      "DEFSYST"
_DEFAULT_CONSTRAINTS = "DEFCNST"
_DEFAULT_LOADING =     "DEFLOAD"
_DEFAULT_RESULTS =     "DEFRESU"


class Node:
    def __init__(self, id: int, **kwargs):
        self.id = id

        if "coors" in kwargs.keys():
            self.coors = kwargs["coors"]
        elif "x" in kwargs.keys():
            self.coors = np.array([kwargs["x"], kwargs["y"], kwargs["z"]], dtype=float)

        if "defsys" in kwargs.keys():
            self.defsys = kwargs["defsys"]
        else:
            self.defsys = 0

        if "outsys" in kwargs.keys():
            self.outsys = kwargs["outsys"]
        else:
            self.outsys = 0

    def __repr__(self) -> str:
        return f"<Node {self.id:n}>"

    def __str__(self) -> str:
        return f"Node ID {self.id:n} {str(self.coors):s}"

    @property
    def id(self) -> int:
        return self._id

    @id.setter
    def id(self, id: int):
        self._id = utils.check_id(id)

    @property
    def defsys(self) -> int:
        return self._defsys

    @defsys.setter
    def defsys(self, defsys: int):
        self._defsys = utils.check_id(defsys)

    @property
    def outsys(self) -> int:
        return self._outsys

    @outsys.setter
    def outsys(self, outsys: int):
        self._outsys = utils.check_id(outsys)

    @property
    def coors(self) -> np.ndarray:
        return self._coors

    @coors.setter
    def coors(self, coors):
        if len(coors) < 3:
            coors = list(coors) + [0.] * (3 - len(coors))
        self._coors = np.array(coors, dtype=float)

    @property
    def X(self) -> float:
        return self.coors[0]

    @X.setter
    def X(self, x: float):
        self.coors[0] = x

    @property
    def Y(self) -> float:
        return self.coors[1]

    @X.setter
    def Y(self, y: float):
        self.coors[1] = y

    @property
    def Z(self) -> float:
        return self.coors[2]

    @Z.setter
    def Z(self, z: float):
        self.coors[2] = z



class Nodes:
    def __init__(self, nodes: [dict, list]):
        self._nodes = {}
        self.nodes = nodes

    def __repr__(self) -> str:
        return f"<Nodes count: {len(self):n}>"

    def __str__(self) -> str:
        ids = list(self.keys())
        return f"Elements count: {len(self):9n}, min ID: {min(ids):9n}, max ID: {max(ids):9n}"

    def __len__(self) -> int:
        return len(self.nodes)

    def __iter__(self) -> (int, Node):
        for nid, node in self.nodes.items():
            yield nid, node

    def __getitem__(self, id: int) -> Node:
        return self.nodes[id]

    def __setitem__(self, id: int, node):
        self.nodes[id] = node

    def __add__(self, nodes):
        self.nodes = nodes
        return self

    def keys(self) -> int:
        return self.nodes.keys()

    def values(self) -> Node:
        return self.nodes.values()

    def items(self) -> (int, Node):
        return self.nodes.items()

    @property
    def nodes(self) -> dict:
        return self._nodes

    @nodes.setter
    def nodes(self, nodes: [list | Node]):
        if type(nodes) is Node:
            self._nodes.setdefault(nodes.id, nodes)

        elif type(nodes) is list:
            for node in nodes:
                self.nodes = node

        else:
            raise TypeError(f"Cannot add Nodes ({str(nodes):s}).")



class CoordinateSystem:
    def __init__(self, id: int, origin: np.ndarray, point1: np.ndarray,
                 point2: np.ndarray, defsys: int = 0, form: str = "xz"):
        self.id = id
        self.form = form
        self.defsys = defsys
        self.O = origin
        self.P1 = point1
        self.P2 = point2

    def __repr__(self) -> str:
        return f"<{type(self).__name__:s} {self.id:n}>"

    def __str__(self) -> str:
        return (f"{type(self).__name__:s} ID {self.id:n} FORM {self.form:s} " +
                f"CSYS {self.defsys:n} [{str(self.O):s}, {str(self.P1):s}, {str(self.P2):s}]")

    @property
    def id(self) -> int:
        return self._id

    @id.setter
    def id(self, id: int):
        self._id = utils.check_id(id)

    @property
    def form(self) -> str:
        return self._form

    @form.setter
    def form(self, form: str):
        self._form = form

    @property
    def defsys(self) -> int:
        return self._defsys

    @defsys.setter
    def defsys(self, defsys: int):
        self._defsys = utils.check_id(defsys)

    @property
    def O(self) -> np.ndarray:
        return self._origin

    @O.setter
    def O(self, point: [list | np.ndarray]):
        self._origin = utils.check_vector(point, 3)

    @property
    def P1(self) -> np.ndarray:
        return self._point1

    @P1.setter
    def P1(self, point: [list | np.ndarray]):
        self._point1 = utils.check_vector(point, 3)

    @property
    def P2(self) -> np.ndarray:
        return self._point2

    @P2.setter
    def P2(self, point: [list | np.ndarray]):
        self._point2 = utils.check_vector(point, 3)



class CART(CoordinateSystem):
    def __init__(self, id: int, origin: np.ndarray, point1: np.ndarray,
                 point2: np.ndarray, defsys: int = 0, form: str = "xz"):
        super().__init__(id, origin, point1, point2, defsys, form)

    @property
    def form(self) -> str:
        return self._form

    @form.setter
    def form(self, form: str):
        available = ("xy", "yx", "xz", "zx", "yz", "zy")
        if form not in available:
            raise ValueError(f"Coordinate System formulation ({form:s}) not in " +
                             f"({', '.join(available):s}).")
        else:
            self._form = form



class CYL(CoordinateSystem):
    def __init__(self, id: int, origin: np.ndarray, point1: np.ndarray,
                 point2: np.ndarray, defsys: int = 0, form: str = "rz"):
        super().__init__(id, origin, point1, point2, defsys, form)

    @property
    def form(self) -> str:
        return self._form

    @form.setter
    def form(self, form: str):
        available = ("rphi", "phir", "rz", "zr", "phiz", "zphi")
        if form not in available:
            raise ValueError(f"Coordinate System formulation ({form:s}) not in " +
                             f"({', '.join(available):s}).")
        else:
            self._form = form




class CoordinateSystems:
    def __init__(self, csys: [dict, list]):
        self._csys = {}
        self.csys = csys

    def __repr__(self) -> str:
        return f"<Coordinate Systems count:{len(self):n}>"

    def __str__(self) -> str:
        ids = list(self.keys())
        return f"Coordinate Systems count: {len(self):9n}, min ID: {min(ids):9n}, max ID: {max(ids):9n}"

    def __iter__(self) -> (int, CoordinateSystem):
        for csid, csys in self.csys.items():
            yield csid, csys

    def __len__(self) -> int:
        return len(self.csys)

    def __getitem__(self, id: int) -> CoordinateSystem:
        return self.csys[id]

    def __setitem__(self, id: int, csys):
        self.csys[id] = csys

    def __add__(self, csys):
        self.csys = csys
        return self

    def keys(self) -> int:
        return self.csys.keys()

    def values(self) -> CoordinateSystem:
        return self.csys.values()

    def items(self) -> (int, CoordinateSystem):
        return self.csys.items()

    @property
    def csys(self) -> dict:
        return self._csys

    @csys.setter
    def csys(self, csys: [list | CoordinateSystem]):
        if isinstance(csys, CoordinateSystem):
            self._csys.setdefault(csys.id, csys)

        elif type(csys) is list:
            for csys in csys:
                self.csys = csys

        else:
            raise TypeError(f"Cannot add Coordinate System ({str(csys):s}).")



class Element:
    _numnodes = 0

    @classmethod
    def new(cls, etype: str, *args, **kwargs):
        if etype == "mass3":
            return MASS3(*args, **kwargs)

        elif etype == "mass6":
            return MASS6(*args, **kwargs)

        elif etype == "spring1":
            return SPRING1(*args, **kwargs)

        elif etype == "spring3":
            return SPRING3(*args, **kwargs)

        elif etype == "spring6":
            return SPRING6(*args, **kwargs)

        elif etype == "rod":
            return ROD(*args, **kwargs)

        elif etype == "bar":
            return BAR(*args, **kwargs)

        elif etype == "beam":
            return BEAM(*args, **kwargs)

        elif etype == "tria3":
            return TRIA3(*args, **kwargs)

        elif etype == "tria6":
            return TRIA6(*args, **kwargs)

        elif etype == "quad4":
            return QUAD4(*args, **kwargs)

        elif etype == "quad8":
            return QUAD8(*args, **kwargs)

        elif etype == "tet4":
            return TET4(*args, **kwargs)

        elif etype == "tet10":
            return TET10(*args, **kwargs)

        elif etype == "hex8":
            return HEX8(*args, **kwargs)

        elif etype == "hex20":
            return HEX20(*args, **kwargs)

        elif etype == "wedge6":
            return WEDGE6(*args, **kwargs)

        elif etype == "wedge15":
            return WEDGE15(*args, **kwargs)

        elif etype == "pyra5":
            return PYRA5(*args, **kwargs)

        elif etype == "pyra12":
            return PYRA12(*args, **kwargs)

        else:
            raise TypeError(f"Unknown Element type {str(etype):s}.")


    def __init__(self, id: int, nodes: [list | np.ndarray]):
        self.id = id
        self.nodes = nodes

    def __repr__(self) -> str:
        return f"<{type(self).__name__:s} {self.id:n}>"

    def __str__(self) -> str:
        return f"{type(self).__name__:s} ID {self.id:n} {str(self.nodes):s}"

    @property
    def id(self) -> int:
        return self._id

    @id.setter
    def id(self, id: int):
        self._id = utils.check_id(id)

    @property
    def nodes(self) -> np.ndarray:
        return self._nodes

    @nodes.setter
    def nodes(self, nodes: list):
        if len(nodes) != type(self)._numnodes:
            raise ValueError(f"Wrong number of nodes ({len(nodes):n} != {type(self)._numnodes:n}) " +
                             f"for element type {type(self).__name__:s}.")
        else:
            self._nodes = np.array([utils.check_id(n) for n in nodes], dtype=int)



class MASS3(Element):
    _numnodes = 1


class MASS6(Element):
    _numnodes = 1



class SPRING1(Element):
    _numnodes = 1



class SPRING3(Element):
    _numnodes = 2



class SPRING6(Element):
    _numnodes = 2



class ROD(Element):
    _numnodes = 2



class BAR(Element):
    _numnodes = 2



class BEAM(Element):
    _numnodes = 2



class TRIA3(Element):
    _numnodes = 3



class TRIA6(Element):
    _numnodes = 6



class QUAD4(Element):
    _numnodes = 4



class QUAD8(Element):
    _numnodes = 8



class TET4(Element):
    _numnodes = 4



class TET10(Element):
    _numnodes = 10



class HEX8(Element):
    _numnodes = 8



class HEX20(Element):
    _numnodes = 20



class WEDGE6(Element):
    _numnodes = 6



class WEDGE15(Element):
    _numnodes = 15



class PYRA5(Element):
    _numnodes = 5



class PYRA12(Element):
    _numnodes = 12



class Elements:
    def __init__(self, elements: [dict, list]):
        self._elements = {}
        self.elements = elements

    def __repr__(self) -> str:
        return f"<Elements count:{len(self):n}>"

    def __str__(self) -> str:
        ids = list(self.keys())
        return f"Elements count: {len(self):9n}, min ID: {min(ids):9n}, max ID: {max(ids):9n}"

    def __iter__(self) -> (int, Element):
        for eid, element in self.elements.items():
            yield eid, element

    def __len__(self) -> int:
        return len(self.elements.keys())

    def __getitem__(self, id: int) -> Element:
        return self.elements[id]

    def __setitem__(self, id: int, element):
        self.elements[id] = element

    def __add__(self, elements):
        self.elements = elements
        return self

    def keys(self) -> int:
        return self.elements.keys()

    def values(self) -> Element:
        return self.elements.values()

    def items(self) -> (int, Element):
        return self.elements.items()

    @property
    def elements(self) -> dict:
        return self._elements

    @elements.setter
    def elements(self, elements: [list | Element]):
        if isinstance(elements, Element):
            self._elements.setdefault(elements.id, elements)

        elif type(elements) is list:
            for element in elements:
                self.elements = element

        else:
            raise TypeError(f"Cannot add Elements ({str(elements):s}).")



class Structure:
    def __init__(self, name: str = _DEFAULT_STRUCTURE, **kwargs):
        self.name = name
        self._nodes = Nodes()
        self._elements = Elements()
        self._csys = CoordinateSystems()

        for key, value in kwargs.items():
            if key == "nodes":
                self._nodes += value

            elif key == "elements":
                self._elements += value

            elif key == "csys":
                self._csys += value

            else:
                raise KeyError(f"{type(self).__name__:s} Unknown Key {str(key):s}.")

    def __repr__(self) -> str:
        return f"<Structure {self.name:s}>"

    def __str__(self) -> str:
        return f"Structure {self.name:s}"

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name: str):
        self._name = utils.check_name(name)

    @property
    def nodes(self) -> Nodes:
        return self._nodes

    @nodes.setter
    def nodes(self, nodes):
        self._nodes += nodes

    @property
    def elements(self) -> Elements:
        return self._elements

    @elements.setter
    def elements(self, elements):
        self._elements += elements

    @property
    def csys(self) -> CoordinateSystems:
        return self._csys

    @csys.setter
    def csys(self, csys):
        self._csys += csys



class Property:
    _props = {"nsm": 1}

    @classmethod
    def new(self, ptype: str, *args, **kwargs):
        ptype = ptype.lower()

        if ptype == "pmass3":
            return PMASS3(*args, **kwargs)

        elif ptype == "pmass6":
            return PMASS6(*args, **kwargs)

        elif ptype == "pspring1":
            return PSPRING1(*args, **kwargs)

        elif ptype == "pspring3":
            return PSPRING3(*args, **kwargs)

        elif ptype == "pspring6":
            return PSPRING6(*args, **kwargs)

        elif ptype == "prod":
            return PROD(*args, **kwargs)

        elif ptype == "pbar":
            return PBAR(*args, **kwargs)

        elif ptype == "pbeam":
            return PBEAM(*args, **kwargs)

        elif ptype == "pshell":
            return PSHELL(*args, **kwargs)

        elif ptype == "psolid":
            return PSOLID(*args, **kwargs)

        else:
            raise TypeError(f"Unknown Property type {str(ptype):s}.")

    def _init(self):
        pass

    def __init__(self, name: str, **kwargs):
        self._properties = {}

        self.name = name
        self.properties = kwargs

        self._init()

    def __repr__(self) -> str:
        return f"<{type(self).__name__:s} Property {self.name:s}>"

    def __str__(self) -> str:
        return f"{type(self).__name__:s} Property {self.name:s}"

    def __getitem__(self, key: str) -> list:
        return self._values[key]

    def __setitem__(self, key: str, values: list) -> list:
        self._values[key] = values

    def __iter__(self) -> (str, list):
        for key, value in self._values.items():
            yield key, value

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name: str):
        self._name = utils.check_name(name)

    @property
    def properties(self) -> dict:
        return self._properties

    @properties.setter
    def properties(self, properties: dict):
        for key in type(self)._props.keys():
            if key in properties.keys():
                self._properties[key] = utils.check_vector(properties[key], type(self)._props[key], True)
            else:
                raise ValueError(f"{type(self).__name__:s} missing property {key:s}.")

    def keys(self) -> str:
        return self._properties.keys()

    def values(self) -> list:
        return self._properties.values()

    def items(self) -> (str, list):
        return self._properties.items()



class PMASS3(Property):
    _props = {"mass": 3}



class PMASS6(Property):
    _props = {"mass": 6}



class PSPRING1(Property):
    _props = {"dof": 2, "stiff": 1}



class PSPRING3(Property):
    _props = {"dof": 3, "stiff": 3}



class PSPRING6(Property):
    _props = {"dof": 3, "stiff": 3}



class PROD(Property):
    _props = {"area": 1, "nsm": 1}



class PBAR(Property):
    _props = {"area": 1, "inertia": 3, "nsm": 1}



class PBEAM(Property):
    _props = {"area": 1, "inertia": 3, "nsm": 1}



class PSHELL(Property):
    _props = {"thick": (1,-1), "nsm": 1}



class PSOLID(Property):
    _props = {"nsm": 1}



class RECTANGLE(Property):
    _props = {"b": 1, "h": 1, "nsm": 1}

    def _init(self):
        b = self._properties["b"]
        h = self._properties["h"]
        A = b * h
        I1 = 1 / 12 * b * h ** 3
        I2 = 1 / 12 * h * b ** 3
        J = I1 + I2
        self._properties["area"] = [A]
        self._properties["interia"] = [I1, I2, J]



class Assign:
    def __init__(self, name: str, material: str = None, prop: str = None):
        self.name = name
        self.material = material
        self.property = prop

    def __repr__(self) -> str:
        return f"<{type(self).__name__:s} {self.name:s}>"

    def __str__(self) -> str:
        msg = f"{type(self).__name__:s} {self.name:s}"
        if self.material is not None:
            msg += f" MATERIAL = {self.material:s}"
        if self.property is not None:
            msg += f" PROPERTY = {self.property:s}"
        return msg

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name: [int | str]):
        if type(name) is int:
            self._name = utils.check_id(name)
        elif type(name) is str:
            self._name = utils.check_name(name)
        else:
            raise TypeError(f"Material and Property must be assigned to either " +
                            f"Element ID (int) or Element Set (str) not {type(name).__name__:s}.")

    @property
    def material(self) -> str:
        return self._material

    @material.setter
    def material(self, name: str):
        if name is None:
            self._material = None
        else:
            self._material = utils.check_name(name)

    @property
    def property(self) -> str:
        return self._property

    @property.setter
    def property(self, name: str):
        if name is None:
            self._property = None
        else:
            self._property = utils.check_name(name)



class System:
    def __init__(self, name: str = _DEFAULT_SYSTEM, **kwargs):
        self._properties = {}
        self._assignments = {}

        self.name = name

        for key, value in kwargs.items():
            if key == "property":
                self.properties = value

            elif key == "assign":
                self.assign = value

            else:
                raise KeyError(f"{type(self).__name__:s} Unknown Key {str(key):s}.")

    def __repr__(self) -> str:
        return f"<System {self.name:s}>"

    def __str__(self) -> str:
        return f"System {self.name:s}"

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name: str):
        self._name = utils.check_name(name)

    @property
    def properties(self) -> Property:
        return self._properties

    @properties.setter
    def properties(self, prop):
        if isinstance(prop, Property):
            self._properties.setdefault(prop.id, prop)

        elif type(prop) is list:
            for p in prop:
                self.properties = p

        else:
            raise TypeError(f"Cannot add Property ({str(property):s}).")

    @property
    def assignments(self) -> Assign:
        return self._assignments

    @assignments.setter
    def assignments(self, assign):
        if isinstance(assign, Assign):
            self._assignments.setdefault(assign.name, assign)

        elif type(assign) is list:
            for a in assign:
                self.assignments = a

        else:
            raise TypeError(f"Cannot add Property and Material assignment ({str(assign):s}).")



class Constraint:
    def __init__(self, id: int, nid: int, dofs: [list], csys: int = 0):
        self.id = id
        self.node = nid
        self.dofs = dofs
        self.csys = csys

    def __repr__(self) -> str:
        return f"<Constraint {self.id:n}>"

    def __str__(self) -> str:
        return f"Constraint {self.id:n} Node {self.node:n} {str(self.dofs):s}"

    @property
    def id(self) -> int:
        return self._id

    @id.setter
    def id(self, id: int):
        self._id = utils.check_id(id)

    @property
    def node(self) -> int:
        return self._node

    @node.setter
    def node(self, node: int):
        self._node = utils.check_node(node)

    @property
    def csys(self) -> int:
        return self._csys

    @csys.setter
    def csys(self, csys: int):
        self._csys = utils.check_id(csys)

    @property
    def dofs(self) -> np.ndarray:
        return self._dofs

    @dofs.setter
    def dofs(self, dofs: [list | np.ndarray]):
        dofs = utils.check_vector(dofs, 6)
        self._dofs = dofs.astype(bool)



class Constraints:
    def __init__(self, name: str = _DEFAULT_CONSTRAINTS, **kwargs):
        self._nodal = {}

        for key, value in kwargs.items():
            if key == "nodal":
                self.nodal = value

    def __repr__(self) -> str:
        return f"<Constraints {self.name:s}>"

    def __str__(self) -> str:
        return f"Constraints {self.name:s}"

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name: str):
        self._name = utils.check_name(name)

    @property
    def nodal(self) -> Constraint:
        return self.nodal

    @nodal.setter
    def nodal(self, nodal):
        if type(nodal) is Constraint:
            self._nodal.setdefault(nodal.id, nodal)

        elif type(nodal) is list:
            for constraint in nodal:
                self.nodal = constraint

        else:
            raise TypeError(f"Cannot add Nodal Constraint ({str(nodal):s}).")



class NodalLoad:
    def __init__(self, id: int, nid: int, values: [list], csys: int = 0):
        self.id = id
        self.node = nid
        self.values = values
        self.csys = csys

    def __repr__(self) -> str:
        return f"<Nodal Load {self.id:n}>"

    def __str__(self) -> str:
        return f"Nodal Load {self.id:n} Node {self.node:n} {str(self.values):s}"

    @property
    def id(self) -> int:
        return self._id

    @id.setter
    def id(self, id: int):
        self._id = utils.check_id(id)

    @property
    def node(self) -> int:
        return self._node

    @node.setter
    def node(self, node: int):
        self._node = utils.check_id(node)

    @property
    def csys(self) -> int:
        return self._csys

    @csys.setter
    def csys(self, csys: int):
        self._csys = utils.check_id(csys)

    @property
    def values(self) -> np.ndarray:
        return self._values

    @values.setter
    def values(self, values: [list | np.ndarray]):
        values = utils.check_vector(values, 6)
        self._values = values.astype(bool)



class Prescribed:
    def __init__(self, id: int, constraint: int, values: [list]):
        self.id = id
        self.constraint = constraint
        self.values = values

    def __repr__(self) -> str:
        return f"<Prescribed Constraint {self.id:n}>"

    def __str__(self) -> str:
        return f"Prescribed Constraint {self.id:n} Constraint {self.constraint:n} {str(self.values):s}"

    @property
    def id(self) -> int:
        return self._id

    @id.setter
    def id(self, id: int):
        self._id = utils.check_id(id)

    @property
    def constraint(self) -> int:
        return self._constraint

    @constraint.setter
    def constraint(self, constraint: int):
        self._constraint = utils.check_id(constraint)

    @property
    def values(self) -> np.ndarray:
        return self._values

    @values.setter
    def values(self, values: [list | np.ndarray]):
        values = utils.check_vector(values, 6)
        self._values = values.astype(float)



class Loading:
    def __init__(self, name: str = _DEFAULT_LOADING, **kwargs):
        self._nodal = {}
        self._prescribed = {}

        self.name = name

        if "nodal" in kwargs.keys():
            self.nodal = kwargs["nodal"]

        if "prescribed" in kwargs.keys():
            self.prescribed = kwargs["prescribed"]

    def __repr__(self) -> str:
        return f"<Loading {self.name:s}>"

    def __str__(self) -> str:
        return f"Loading {self.name:s}"

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name: str):
        self._name = utils.check_name(name)

    @property
    def nodal(self) -> NodalLoad:
        return self._nodal

    @nodal.setter
    def nodal(self, nodal):
        if type(nodal) is NodalLoad:
            self._nodal.setdefault(nodal.id, nodal)

        elif type(nodal) is list:
            for load in nodal:
                self.nodal = load

        else:
            raise TypeError(f"Cannot add Nodal Load ({str(nodal):s}).")

    @property
    def prescribed(self) -> Prescribed:
        return self._prescribed

    @prescribed.setter
    def prescribed(self, prescribed):
        if type(prescribed) is Prescribed:
            self._prescribed.setdefault(prescribed.id, prescribed)

        elif type(prescribed) is list:
            for load in prescribed:
                self.prescribed = load

        else:
            raise TypeError(f"Cannot add Prescribed Constraint Displacement ({str(prescribed):s}).")



class Results:
    def __init__(self, name: str = _DEFAULT_RESULTS, **kwargs):
        self._nodal = {}
        self._elemental = {}

        for key, value in kwargs.items():
            if key == "nodal":
                self.nodal = value

    def __repr__(self) -> str:
        return f"<Results {self.name:s}>"

    def __str__(self) -> str:
        return f"Results {self.name:s}"

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name: str):
        self._name = utils.check_name(name)



class Situation:
    def __init__(self, name: str, structure: str = None, system: str = None,
                 constraints: str = None, loading: str = None, results: str = None):
        self.name = name
        self.structure = structure
        self.system = system
        self.constraints = constraints
        self.loading = loading
        self.results = results

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name: str):
        self._name = utils.check_name(name)

    @property
    def structure(self) -> str:
        return self._structure

    @structure.setter
    def structure(self, name: str):
        if name is None:
            self._structure = _DEFAULT_STRUCTURE
        else:
            self._structure = utils.check_name(name)

    @property
    def system(self) -> str:
        return self._system

    @system.setter
    def system(self, name: str):
        if name is None:
            self._system = _DEFAULT_SYSTEM
        else:
            self._system = utils.check_name(name)

    @property
    def constraints(self) -> str:
        return self._constraints

    @constraints.setter
    def constraints(self, name: str):
        if name is None:
            self._constraints = _DEFAULT_CONSTRAINTS
        else:
            self._constraints = utils.check_name(name)

    @property
    def loading(self) -> str:
        return self._loading

    @loading.setter
    def loading(self, name: str):
        if name is None:
            self._loading = _DEFAULT_LOADING
        else:
            self._loading = utils.check_name(name)

    @property
    def results(self) -> str:
        return self._results

    @results.setter
    def results(self, name: str):
        if name is None:
            self._results = _DEFAULT_RESULTS
        else:
            self._results = utils.check_name(name)



class Component:
    def __init__(self, name: str, **kwargs):
        self.name = name
        self._situation = {}
        self._structure = {}
        self._system = {}
        self._constraints = {}
        self._loading = {}
        self._results = {}

        for key in kwargs.keys():
            if key not in ["situation", "structure", "system", "constraints", "Loading", "results"]:
                raise KeyError(f"{type(self).__name__:s} Unknown Key {str(key):s}.")

        if "situation" in kwargs.keys():
            self.situation = Situation(kwargs["situation"])
        else:
            self.situation = Situation()

        if "structure" in kwargs.keys():
            self.structure = Structure(kwargs["structure"])
        else:
            self.structure = Structure()

        if "system" in kwargs.keys():
            self.system = System(kwargs["system"])
        else:
            self.system = System()

        if "constraints" in kwargs.keys():
            self.constraints = Constraints(kwargs["constraints"])
        else:
            self.constraints = Constraints()

        if "loading" in kwargs.keys():
            self.loading = Loading(kwargs["loading"])
        else:
            self.loading = Loading()

        if "results" in kwargs.keys():
            self.results = Results(kwargs["results"])
        else:
            self.results = Results()


    def __repr__(self) -> str:
        return f"<Component {self.name:s}>"

    def __str__(self) -> str:
        return f"Component {self.name:s}"

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name: str):
        self._name = utils.check_name(name)

    @property
    def situation(self) -> Situation:
        return self._situation

    @situation.setter
    def situation(self, situation):
        if type(situation) not in (dict, Situation):
            raise TypeError(f"Situation Variant has to be of type dict or Situation, " +
                            f"not {type(situation).__name__:s}.")
        elif type(situation) is dict:
            situation = Situation(**situation)

        self._situation.setdefault(situation.name, situation)

    def get_situation(self, name: str = None):
        if name is None:
            return self._situation[_DEFAULT_SITUATION]
        else:
            return self._situation[name]

    @property
    def structure(self) -> Structure:
        return self._structure

    @structure.setter
    def structure(self, structure):
        if type(structure) not in (dict, Structure):
            raise TypeError(f"Structure Variant has to be of type dict or Structure, " +
                            f"not {type(structure).__name__:s}.")
        elif type(structure) is dict:
            structure = Structure(**structure)

        self._structure.setdefault(structure.name, structure)

    def get_structure(self, name: str = None):
        if name is None:
            return self._structure[_DEFAULT_STRUCTURE]
        else:
            return self._structure[name]

    @property
    def system(self) -> System:
        return self._system

    @system.setter
    def system(self, system):
        if type(system) not in (dict, System):
            raise TypeError(f"system Variant has to be of type dict or System, " +
                            f"not {type(system).__name__:s}.")
        elif type(system) is dict:
            system = System(**system)

        self._system.setdefault(system.name, system)

    def get_system(self, name: str = None):
        if name is None:
            return self._system[_DEFAULT_SYSTEM]
        else:
            return self._system[name]

    @property
    def constraints(self) -> Constraints:
        return self._constraints

    @constraints.setter
    def constraints(self, constraints):
        if type(constraints) not in (dict, Constraints):
            raise TypeError(f"constraints Variant has to be of type dict or Constraints, " +
                            f"not {type(constraints).__name__:s}.")
        elif type(constraints) is dict:
            constraints = Constraints(**constraints)

        self._constraints.setdefault(constraints.name, constraints)

    def get_constraints(self, name: str = None):
        if name is None:
            return self._constraints[_DEFAULT_CONSTRAINTS]
        else:
            return self._constraints[name]

    @property
    def loading(self) -> Loading:
        return self._loading

    @loading.setter
    def loading(self, loading):
        if type(loading) not in (dict, Loading):
            raise TypeError(f"loading Variant has to be of type dict or Loading, " +
                            f"not {type(loading).__name__:s}.")
        elif type(loading) is dict:
            loading = Loading(**loading)

        self._loading.setdefault(loading.name, loading)

    def get_loading(self, name: str = None):
        if name is None:
            return self._loading[_DEFAULT_LOADING]
        else:
            return self._loading[name]

    @property
    def results(self) -> Results:
        return self._results

    @results.setter
    def results(self, results):
        if type(results) not in (dict, Results):
            raise TypeError(f"results Variant has to be of type dict or Results, " +
                            f"not {type(results).__name__:s}.")
        elif type(results) is dict:
            results = Results(**results)

        self._results.setdefault(results.name, results)

    def get_results(self, name: str = None):
        if name is None:
            return self._results[_DEFAULT_RESULTS]
        else:
            return self._results[name]



class Material:
    def __init__(self, name: str, **kwargs):
        self.name = name

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name: str):
        self._name = utils.check_name(name)



class MatISO(Material):
    pass



class Materials:
    def __init__(self, materials: [list | Material] = None):
        self._materials = {}

    def __getitem__(self, name: str) -> Material:
        name = utils.check_name(name)
        return self._materials[name]

    def __setitem__(self, name: str, material: Material):
        name = utils.check_name(name)
        if isinstance(material, Material):
            if name != material.name:
                raise ValueError(f"Name {name:s} does not match material name {material.name:s}")
            self._materials[name] = material
        else:
            self._materials[name] = Material.new(**material)

    def __iter__(self):
        for key, value in self._material.items():
            yield key, value

    def __len__(self) -> int:
        return len(self._materials)

    def __add__(self, material):
        self[material.name] = material
        return self

    def keys(self) -> str:
        return self._materials.keys()

    def values(self) -> Material:
        return self._materials.values()

    def items(self) -> [str, Material]:
        return self._materials.items()

    def add(self, materials: [list | dict | Material]):
        if materials is None:
            pass

        elif isinstance(materials, Material):
            self[materials.name] = materials

        elif type(materials) is list:
            for material in materials:
                self.add(material)

        elif type(materials) is dict:
            self.add(Material(**materials)

        else:
            raise TypeError(f"Wrong type of materials {type(materials).__name__:s}.")




if __name__ == "__main__":
    pdb.set_trace()
    tria3 = Element.new("tria3", 1, [1, 2, 3])
    print(f"{tria3 = }")

    tria6 = Element.new("tria6", 2, [1, 2, 3, 4, 5, 6])
    print(f"{tria6 = }")

    quad4 = Element.new("quad4", 3, [1, 2, 3, 4])
    print(f"{str(quad4) = }")

    els = Elements([tria3, tria6])

    els[quad4.id] = quad4

    print(f"{str(els) = }")

    a = Assign("ET_ALLE", "STEEL", "SOLID")
    print(str(a))





