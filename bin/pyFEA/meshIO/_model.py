try:
    from typing import Self
except ImportError:
    from typing_extensions import Self
import numpy as np
from numpy.typing import ArrayLike
import scipy.interpolate
import scipy.spatial
import scipy.interpolate.interpnd

from rich.console import Console


element_type = {
    "MASS":     1,
    "ROD":      2,
    "BAR":      2,
    "BEAM":     2,
    "TRIA3":    3,
    "TRIA6":    6,
    "QUAD4":    4,
    "QUAD8":    8,
    "TET4":     4,
    "TET10":   10,
    "HEX8":     8,
    "HEX20":   20,
    # "HEX27":   27,
    "WEDGE6":   6,
    # "WEDGE15": 15,
    "PYRA5":    5,
}

element_topology = {
    "MASS":     0,
    "ROD":      1,
    "BAR":      1,
    "BEAM":     1,
    "TRIA3":    2,
    "TRIA6":    2,
    "QUAD4":    2,
    "QUAD8":    2,
    "TET4":     3,
    "TET10":    3,
    "HEX8":     3,
    "HEX20":    3,
    # "HEX27":   3,
    "WEDGE6":   3,
    # "WEDGE15": 3,
    "PYRA5":    3,
}


def Info(string: str, indent: int = 0, highlight: bool = True) -> None:
        # f"[bold]Info:[/bold] {str(string):s}", highlight=highlight)
    Console(stderr=True).print(
        f"[[bold]+[/bold]] {str(string):s}", highlight=highlight)


def Warn(string: str, indent: int = 0, highlight: bool = True) -> None:
        # f"[yellow][bold]Warning:[/bold] {str(string):s}[/yellow]", highlight=highlight)
    Console(stderr=True).print(
        f"[[yellow][bold]-[/bold][/yellow]] [yellow]{str(string):s}[/yellow]",
        highlight=highlight)


def Error(string: str, indent: int = 0, highlight: bool = True) -> None:
        # f"[red][bold]Error:[/bold] {str(string):s}[/red]",
    Console(stderr=True).print(
        f"[[red][bold]![/bold][/red]] [red]{str(string):s}[/red]",
        highlight=highlight)


class Id:
    def __init__(self, id: int):
        self.id = id

    def __repr__(self) -> str:
        return f"<{type(self).__name__:s} object>\n  ID: {self.id:n}"

    @property
    def id(self) -> int:
        return self._id

    @id.setter
    def id(self, id: int):
        if type(id) is not int:
            message = (f"{str(type(self).__name__):s} ID type must be an int, " +
                       f"not {str(type(id).__name__):s}.")
            Error(message)
            raise TypeError(message)
        elif id < 1:
            message = f"{str(type(self).__name__):s} ID {id:n} < 1."
            Error(message)
            raise ValueError(message)
        else:
            self._id = id


class Name:
    def __init__(self, name: str):
        self.name = name

    def __repr__(self) -> str:
        return f"<{type(self).__name__:s} object>\n  Name: {self.name:n}"

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name: str):
        if type(name) is not str:
            message = (f"{str(type(self).__name__):s} name type must be a str, " +
                       f"not {str(type(name).__name__):s}.")
            Error(message)
            raise TypeError(message)
        # ascii A-Z = 65 - 90
        # ascii a-z = 97 - 122
        elif name[0].upper() not in [chr(i) for i in range(65, 91)]:
            message = f"{str(type(self).__name__):s} name {name:s} must start with a-z or A-Z."
            Error(message)
            raise ValueError(message)
        else:
            self._name = name


class ContainerDict:
    def __init__(self, key_name: str, key_type: [tuple[type] | list[type] | type],
                 value_name: str, value_type: [tuple[type] | list[type] | type]):
        self.__key_name = str(key_name)
        if key_type not in (tuple, list):
            self.__key_type = tuple([key_type])
        else:
            self.__key_type = tuple(key_type)
        self.__val_name = str(value_name)
        if value_type not in (tuple, list):
            self.__val_type = tuple([value_type])
        else:
            self.__val_type = tuple(value_type)
        self.__dict = {}
        self.__min = None
        self.__max = None

    def __len__(self) -> int:
        return len(self.__dict)

    def __iter__(self):
        return self.__dict.__iter__()

    def __getitem__(self, key):
        if key not in self.__dict.keys():
            message = f"{str(type(self).__name__):s} {self.__key_name:s} {str(key):s} found."
            Error(message)
            raise KeyError(message)
        else:
            return self.__dict[key]

    def __setitem__(self, key, value):
        if key is None or value is None:
            pass
        elif not isinstance(key, self.__key_type):
            message = (f"{str(type(self).__name__):s} {self.__key_name:s} {str(key):s} " +
                        "must be one of types " +
                       f"({', '.join([str(t.__name__) for t in self.__key_type]):s}), not " +
                       f"{str(type(key).__name__):s}.")
            Error(message)
            raise TypeError(message)
        elif not isinstance(value, self.__val_type):
            message = (f"{str(type(self).__name__):s} {self.__val_name:s} {str(value):s} " +
                        "must be one of types " +
                       f"({', '.join([str(t.__name__) for t in self.__val_type]):s}), not " +
                       f"{str(type(value).__name__):s}.")
            Error(message)
            raise TypeError(message)
        elif key in self.__dict.keys():
            message = (f"{str(type(self).__name__):s} duplicate {self.__key_name:s} {str(key):s}.")
            Error(message)
            raise ValueError(message)
            # self.__dict[key] = value
        else:
            self.__dict.setdefault(key, value)
            key_sort = sorted(list(self.__dict.keys()))
            self.__dict = {key: self.__dict[key] for key in key_sort}
            self.__min = key_sort[ 0]
            self.__max = key_sort[-1]

    def __repr__(self) -> str:
        lines = [f"<{str(type(self).__name__):s} container object>",
                 f"  Count: {len(self.__dict):n}",
                 f"  Min {self.__key_name:s}: {str(self.__min)}",
                 f"  Max {self.__key_name:s}: {str(self.__max)}"]
        return "\n".join(lines)

    def __str__(self) -> str:
        return (f"{str(type(self).__name__):s} count: {self.count:n}, " +
                f"min {self.__key_name:s}: {str(self.__min):s}, " +
                f"max {self.__key_name:s}: {str(self.__max):s}")

    def __add__(self, container):
        for key, value in container.items():
            self[key] = value
        return self

    def __iadd__(self, container):
        for key, value in container.items():
            self[key] = value
        return self

    def keys(self):
        return self.__dict.keys()

    def items(self):
        return self.__dict.items()

    def values(self):
        return self.__dict.values()

    @property
    def count(self):
        return len(self.__dict)

    @property
    def min(self):
        return self.__min

    @property
    def max(self):
        return self.__max

    def asdict(self):
        return self.__dict

    def aslist(self):
        return list(self.__dict.values())



class Node(Id):
    def __init__(self, nid: int, coors: ArrayLike):
        super().__init__(nid)
        self.coors = coors

    def __str__(self) -> str:
        return f"Node ID {self.id:n}"

    def __repr__(self) -> str:
        lines = ["<Node object>",
                 f"  ID: {self.id:n}",
                 f"  Coordinates: {str(self.coors):s}"]
        return "\n".join(lines)

    def __eq__(self, node) -> bool:
        if type(node) is not type(self):
            return False
        else:
            return self.id == node.id and np.array_equal(self.coors, node.coors)

    def __neq__(self, node) -> bool:
        return not self.__eq__(node)

    def __add__(self, value: [int, float, tuple, list, ArrayLike]):
        self._coors += value
        return self

    def __iadd__(self, value: [int, float, tuple, list, ArrayLike]):
        self._coors += value
        return self

    def __sub__(self, value: [int, float, tuple, list, ArrayLike]):
        self._coors -= value
        return self

    def __isub__(self, value: [int, float, tuple, list, ArrayLike]):
        self._coors -= value
        return self

    def __mul__(self, value: [int, float, tuple, list, ArrayLike]):
        self._coors *= value
        return self

    def __imul__(self, value: [int, float, tuple, list, ArrayLike]):
        self._coors *= value
        return self

    def __truediv__(self, value: [int, float, tuple, list, ArrayLike]):
        self._coors /= value
        return self

    def __itruediv__(self, value: [int, float, tuple, list, ArrayLike]):
        self._coors /= value
        return self

    def __pow__(self, value: [int, float]):
        self._coors = self._coors ** value
        return self

    @property
    def coors(self) -> ArrayLike:
        return self._coors

    @coors.setter
    def coors(self, coors: ArrayLike):
        if coors is None:
            coors = [0., 0., 0.]
        elif type(coors) not in (tuple, list, np.ndarray):
            message = (f"Node {self.id:n} coordinates must be an array or list, " +
                       f"not {str(type(coors).__name__):s}.")
            Error(message)
            raise TypeError(message)
        coors = np.array(coors, dtype=float).flatten()
        if len(coors) < 3:
            message = (f"Node {self.id:n} coordinates length < 3, " +
                       f"appending zeros.")
            Warn(message)
            coors = np.hstack((coors, [0.] * (3 - len(coors))))
        self._coors = np.array(coors, dtype=float).flatten()

    def transform(self, T: np.ndarray):
        """
        Multiply coordinates by a matrix of transformation sines and cosines:
        x' = T @ x

        rotation matrix:
        T  = [x_x x_y x_z]
             [y_x y_y y_z]
             [z_x z_y z_z]

        reduced rotation and translation matrix:
        T  = [x_x x_y x_z t_x]
             [y_x y_y y_z t_y]
             [z_x z_y z_z t_z]

        full rotation and translation matrix:
        T  = [x_x x_y x_z t_x]
             [y_x y_y y_z t_y]
             [z_x z_y z_z t_z]
             [ 0.  0.  0.  1.]

        x' = [x_x x_y x_z t_x]   [x]   [x_x * x + x_y * y + x_z * z + t_x * 1]
             [y_x y_y y_z t_y] @ [y] = [y_x * x + y_y * y + y_z * z + t_y * 1]
             [z_x z_y z_z t_z]   [z]   [z_x * x + z_y * y + z_z * z + t_z * 1]
             [ 0   0   0   1 ]   [1]   [                 1                   ]

        """
        if type(T) is not np.ndarray:
            message = (f"Transformation matrix {str(T):s} must be of type numpy.ndarray, " +
                       f"node {str(type(T).__name__):s}.")
            Error(message)
            raise TypeError(message)
        elif T.shape == (3, 3):
            self._coors = T @ self._coors
        elif T.shape == (3, 4):
            self._coors = T @ np.hstack((self._coors, [1.]))
        elif T.shape == (4, 4):
            self._coors = (T @ np.hstack((self._coors, [1.])))[:3]
        else:
            message = (f"Transformation matrix {str(T):s} must be of shape (3, 3) or (3, 4) " +
                       f"or (4, 4), not {str(T.shape):s}.")
            Error(message)
            raise ValueError(message)


class Nodes(ContainerDict):
    @classmethod
    def __cuboid(cls, min: np.ndarray, max: np.ndarray, scale: float = 1.) -> np.ndarray:
        """
        Creates a cuboid from min coordinates and max coordinates, scales it by
        a scale factor. min and max ara opposite corners of the final cuboid

        In:
            min  - min coordinates [x_min, y_min, z_min]
            max  - max coordinates [x_max, y_max, z_max]
        """
        max = max.flatten()
        min = min.flatten()
        avg = (max + min) / 2
        cube = np.zeros((8, 3), dtype=float)
        cube[0] = scale * (np.array([max[0], max[1], max[2]], dtype=float) - avg) + avg
        cube[1] = scale * (np.array([min[0], max[1], max[2]], dtype=float) - avg) + avg
        cube[2] = scale * (np.array([min[0], min[1], max[2]], dtype=float) - avg) + avg
        cube[3] = scale * (np.array([max[0], min[1], max[2]], dtype=float) - avg) + avg
        cube[4] = scale * (np.array([max[0], min[1], min[2]], dtype=float) - avg) + avg
        cube[5] = scale * (np.array([min[0], min[1], min[2]], dtype=float) - avg) + avg
        cube[6] = scale * (np.array([max[0], max[1], min[2]], dtype=float) - avg) + avg
        cube[7] = scale * (np.array([min[0], max[1], min[2]], dtype=float) - avg) + avg

        return avg, cube

    def __init__(self, nodes: [list[Node] | dict[int, ArrayLike] | ArrayLike]):
        super().__init__("ID", int, "Node", Node)
        self.nodes = nodes

    @property
    def nodes(self) -> dict[int, Node]:
        return super().asdict()

    @nodes.setter
    def nodes(self, nodes: [Node | list[Node] | dict[int, ArrayLike] | ArrayLike]):
        if type(nodes) is Node:
            self[nodes.id] = nodes

        elif type(nodes) in (tuple, list):
            if type(nodes[0]) is float:
                self.nodes = Node(self.max + 1 if self.max else 1, nodes)

            else:
                for node in nodes:
                    if type(node) is Node:
                        self.nodes = node

                    elif type(node) in (tuple, list, np.ndarray):
                        self.nodes = Node(self.max + 1 if self.max else 1, node)
                    else:
                        message = (f"Node {str(node):s} must be of type Node, not " +
                                   f"{str(type(node).__name__):s}.")
                        Error(message)
                        raise TypeError(message)

        elif type(nodes) is dict:
            for nid, node in nodes.items():
                if type(node) is Node:
                    self.nodes = node

                elif type(node) in (list, tuple, np.ndarray):
                    self.nodes = Node(nid, node)

        elif type(nodes) is np.ndarray:
            for node in nodes:
                self.nodes = Node(self.max + 1 if self.max else 1, node)

        else:
            message = (f"Nodes {str(nodes):s} must be either a list or tuple of Node types, " +
                       f"not {str(type(nodes).__name__):s}.")
            Error(message)
            raise TypeError(message)

    @property
    def coors(self) -> np.ndarray:
        return np.array([node.coors for node in self.values()], dtype=float)

    @property
    def extents(self) -> [np.ndarray, np.ndarray]:
        return np.min(self.coors, axis=0), np.max(self.coors, axis=0)


    def aslist(self) -> list:
        coors = []
        for nid in self.nodes.keys():
            coors.append(list(self.nodes[nid].coors))
        return coors

    def asdict(self) -> dict:
        coors = {}
        for nid in self.nodes.keys():
            coors.setdefault(nid, self.nodes[nid].coors)
        return coors

    def asarray(self) -> np.ndarray:
        return np.array(self.aslist(), dtype=float)

    def transform(self, T: np.ndarray):
        for nid in self.nodes.keys():
            self.nodes[nid].transform(T)

    def match(self, nodes) -> dict:
        """
        Match two sets of Nodes based on their coordinates (distance)
        """
        pass

    def map_scalar(self, onodes: Self, scalar: dict,
                   cube_scale: float = 20., distances: bool = False,
                   max_distance: float = None) -> np.ndarray:
        """
        Map scalar from one set of nodes to these nodes

        In:
            onodes - Nodes object with original nodes
            scalar - dict with values on original nodes {nid, value}
        """
        # create a cube for extrapolation
        omin, omax = onodes.extents
        nmin, nmax = self.extents
        min = np.minimum(omin, nmin).flatten()
        max = np.maximum(omax, nmax).flatten()

        avg, cube = self.__cuboid(max, min, scale=cube_scale)

        # pair original coordinates to scalar values and add the cuboid
        odata = [(nid, onodes[nid].coors, scalar[nid]) for nid in onodes.keys()]
        opoints = np.array([x[1] for x in odata], dtype=float)
        ovalues = np.array([x[2] for x in odata], dtype=float)
        mean = np.mean(ovalues, axis=0)
        cube_values = np.array([mean] * cube.shape[0], dtype=float)
        opoints = np.concatenate((opoints, cube), axis=0)
        ovalues = np.concatenate((ovalues, cube_values), axis=0)

        # prepare new nodes and their coordinates
        ndata = [(nid, self.nodes[nid].coors) for nid in self.nodes.keys()]
        nids = [x[0] for x in ndata]
        npoints = np.array([x[1] for x in ndata], dtype=float)

        # map values to new nodes
        grid = scipy.interpolate.griddata(opoints, ovalues, npoints, method="linear")


        # reformat mapped values to a dict {nid, value}
        results = dict(list(zip(nids, grid)))

        # if closest distances are reuqested
        if distances:
            tree = scipy.spatial.cKDTree(opoints)
            xi = scipy.interpolate.interpnd._ndim_coords_from_arrays(npoints,
                                                                     ndim=npoints.shape[1])
            distances, indexes = tree.query(xi)

            # Copy original result but mask missing values with NaNs
            if max_distance:
                grid2 = grid[:]
                grid2[distances > max_distance] = np.nan
                grid = grid2
            distances = dict(list(zip(nids, distances)))

        else:
            distances = None

        return results, distances


    def map_vector(self, onodes: Self, vector: np.ndarray,
                   cube_scale: float = 20., distances: bool = False,
                   max_distance: float = None) -> np.ndarray:
        """
        Map vectors from one set of nodes to these nodes

        In:
            onodes - Nodes object with original nodes
            vector - dict with values on original nodes {nid, vector[n]}
        """
        # create a cube for extrapolation
        omin, omax = onodes.extents
        nmin, nmax = self.extents
        min = np.minimum(omin, nmin).flatten()
        max = np.maximum(omax, nmax).flatten()

        avg, cube = self.__cuboid(max, min, scale=cube_scale)

        # pair original coordinates to scalar values and add the cuboid
        odata = [(nid, onodes[nid].coors, vector[nid]) for nid in onodes.keys()]
        opoints = np.array([x[1] for x in odata], dtype=float)
        ovalues = np.array([x[2] for x in odata], dtype=float)
        mean = np.mean(ovalues, axis=0)
        cube_values = np.array([mean] * 8, dtype=float).reshape(-1, ovalues.shape[1])
        opoints = np.concatenate((opoints, cube), axis=0)
        ovalues = np.concatenate((ovalues, cube_values), axis=0)

        # prepare new nodes and their coordinates
        ndata = [(nid, self.nodes[nid].coors) for nid in self.nodes.keys()]
        nids = [x[0] for x in ndata]
        npoints = np.array([x[1] for x in ndata], dtype=float)

        # map values to new nodes
        grid = []
        for n in range(ovalues.shape[1]):
            grid.append(scipy.interpolate.griddata(opoints, ovalues[:,n], npoints, method="linear"))


        # reformat mapped values to a dict {nid, value}
        grid = np.array(list(zip(*grid)), dtype=float)
        results = dict(list(zip(nids, grid)))

        # if closest distances are reuqested
        if distances:
            tree = scipy.spatial.cKDTree(opoints)
            xi = scipy.interpolate.interpnd._ndim_coords_from_arrays(npoints,
                                                                     ndim=npoints.shape[1])
            distances, indexes = tree.query(xi)

            # Copy original result but mask missing values with NaNs
            if max_distance:
                grid2 = grid[:]
                grid2[distances > max_distance] = np.nan
                grid = grid2
            distances = dict(list(zip(nids, distances)))

        else:
            distances = None

        return results, distances

    def map_tensor(self, onodes: Self, tensor: np.ndarray,
                   cube_scale: float = 20., distances: bool = False,
                   max_distance: float = None) -> np.ndarray:
        """
        Map tensors from one set of nodes to these nodes

        In:
            onodes - Nodes object with original nodes
            tensor - dict with values on original nodes {nid, tensor[m x n]}
        """
        # create a cube for extrapolation
        omin, omax = onodes.extents
        nmin, nmax = self.extents
        min = np.minimum(omin, nmin).flatten()
        max = np.maximum(omax, nmax).flatten()

        avg, cube = self.__cuboid(max, min, scale=cube_scale)

        # pair original coordinates to scalar values and add the cuboid
        odata = [(nid, onodes[nid].coors, tensor[nid]) for nid in onodes.keys()]
        opoints = np.array([x[1] for x in odata], dtype=float)
        ovalues = np.array([x[2] for x in odata], dtype=float)

        oshape = ovalues.shape

        if len(ovalues.shape) == 1:
            ovalues = ovalues.reshape(ovalues.shape[0], 1, 1)
            value_type = "scalar"
        elif len(ovalues.shape) == 2:
            ovalues = ovalues.reshape(ovalues.shape[0], 1, ovalues.shape[1])
            value_type = "vector"
        else:
            value_type = "tensor"

        mean = np.mean(ovalues, axis=0)
        cube_values = np.array([mean] * cube.shape[0], dtype=float).reshape(-1, ovalues.shape[1], ovalues.shape[2])
        opoints = np.concatenate((opoints, cube), axis=0)
        ovalues = np.concatenate((ovalues, cube_values), axis=0)

        # prepare new nodes and their coordinates
        ndata = [(nid, self.nodes[nid].coors) for nid in self.nodes.keys()]
        nids = [x[0] for x in ndata]
        npoints = np.array([x[1] for x in ndata], dtype=float)

        # map values to new nodes
        grid = np.empty((npoints.shape[0], ovalues.shape[1], ovalues.shape[2]), dtype=float)
        for m in range(ovalues.shape[1]):
            for n in range(ovalues.shape[2]):
                grid[:,m,n] = scipy.interpolate.griddata(opoints, ovalues[:,m,n], npoints, method="linear")

        if value_type == "scalar":
            grid = grid.reshape(grid.shape[0])
        elif value_type == "vector":
            grid = grid.reshape(grid.shape[0], -1)

        # reformat mapped values to a dict {nid, value}
        results = dict(list(zip(nids, grid)))

        # if closest distances are reuqested
        if distances:
            tree = scipy.spatial.cKDTree(opoints)
            xi = scipy.interpolate.interpnd._ndim_coords_from_arrays(npoints,
                                                                     ndim=npoints.shape[1])
            distances, indexes = tree.query(xi)

            # Copy original result but mask missing values with NaNs
            if max_distance:
                grid2 = grid[:]
                grid2[distances > max_distance] = np.nan
                grid = grid2
            distances = dict(list(zip(nids, distances)))

        else:
            distances = None

        return results, distances



class Element(Id):
    def __init__(self, eid: int, etype: str, nodes: [list | tuple | ArrayLike]):
        super().__init__(eid)
        self.type = etype
        self.nodes = nodes

    def __repr__(self) -> str:
        lines = [f"<Element object>",
                 f"  ID: {self.id:n}",
                 f"  Type: {self.type:s}",
                 f"  Nodes: {self.count:n}"]
        return "\n".join(lines)

    def __str__(self) -> str:
        return f"{self.type:s} Element ID {self.id:n}"

    def __len__(self) -> int:
        return self.count

    @property
    def type(self) -> str:
        return self._type

    @type.setter
    def type(self, etype: str):
        if type(etype) is not str:
            message = (f"Element {self.id:n} type {str(etype):s} must be a str, " +
                       f"not {str(type(etype).__name__):s}.")
            Error(message)
            raise TypeError(message)
        elif etype.upper() not in element_type.keys():
            message = (f"Element {self.id:n} type {str(etype):s} not found in " +
                       f"not ({', '.join(list(element_type.keys())):s}).")
            Warn(message)
            self._len = None
        else:
            self._len = int(element_type[etype])
        self._type = etype.upper()

    @property
    def count(self) -> int:
        if self._len:
            return self._len
        else:
            return len(self._nodes)

    @property
    def nodes(self) -> list:
        return self._nodes

    @nodes.setter
    def nodes(self, nodes: [tuple | list | ArrayLike]):
        if type(nodes) not in (tuple, list, ArrayLike):
            message = (f"Element {self.id:n} nodes {str(nodes):s} must be " +
                        "either tuple, list or a vector, not " +
                       f"{str(type(nodes).__name__):s}.")
            Error(message)
            raise TypeError(message)
        nodes = list(nodes)
        if self._len is None:
            message = (f"Assigning {len(nodes):n} Nodes to Element {self.id:n} " +
                       f"of unknown type {self.type:s}.")
            Info(message)
            self._len = len(nodes)
        elif self._len is not None and self._len != len(nodes):
            message = (f"Element {self.id:n} type {self.type:s} must have " +
                       f"{self._len:n} nodes, not {len(nodes):n}.")
            Error(message)
            raise ValueError(message)
        self._nodes = nodes


class Elements(ContainerDict):
    def __init__(self, elements: [Element | tuple | list | dict]):
        super().__init__("ID", int, "Element", Element)
        self.elements = elements

    @property
    def elements(self) -> dict:
        return super().asdict()

    @elements.setter
    def elements(self, elements: [Element | tuple | list | dict]):
        if elements is None:
            pass
        elif type(elements) is Element:
            self[elements.id] = elements

        elif type(elements) in (tuple, list):
            for element in elements:
                if type(element) is Element:
                    self.elements = element
                elif type(element) in (tuple, list):
                    self.elements = Element(element[0], element[1], element[2])
                else:
                    message(f"Wrong definition of Element {str(element):s}")
                    Error(message)
                    raise ValueError(message)

        elif type(elements) is dict:
            for eid, element in elements.items():
                if type(element) is Element:
                    self.elements = element
                elif type(element) in (tuple, list):
                    self.elements = Element(eid, element[0], element[1])
                else:
                    message(f"Wrong definition of Element {eid:n} ({str(element):s}).")
                    Error(message)
                    raise ValueError(message)
        else:
            message(f"Elements must be of type tuple, list, dict or Element, " +
                    f"not {str(type(elements).__name__):s}.")
            Error(message)
            raise ValueError(message)

    def aslist(self) -> list:
        elements = []
        for eid, element in self.elements.items():
            # elements.append([eid, element.type, *element.nodes])
            elements.append([eid, element.type, element.nodes])
        return elements

    def asdict(self) -> dict:
        elements = {}
        for eid, element in self.elements.items():
            # elements.setdefault(eid, [element.type, *element.nodes])
            elements.setdefault(eid, [element.type, element.nodes])
        return elements


class Material(Name):
    __types = ("ISO")

    def _check_values(self,
                      name: str,
                      values: [float | ArrayLike | list[list[float, float]] | dict[float, float]]):
        # no temperature dependance
        if type(values) in (float, np.float64, np.float128):
            if values <= 0.:
                message = (f"{str(type(self).__name__):s} {name:s} must be " +
                           f"> 0.0, not {str(values):s}.")
                Error(message)
                raise ValueError(message)
            return values

        # temperature dependent values
        elif type(values) is np.ndarray:
            if values.dtype not in (float, np.float64, np.float128):
                message = (f"{str(type(self).__name__):s} {name:s} must be of types " +
                           f"float, not {str(values.dtype.__name__):s}.")
                Error(message)
                raise TypeError(message)
            elif values.shape[1] != 2:
                message = (f"{str(type(self).__name__):s} {name:s} must be an array " +
                           f"of 2 columns [Temperature, {name:s}], not {str(values.shape):s}.")
                Error(message)
                raise ValueError(message)
            elif any([val <= 0.0 for val in values[:, 1]]):
                message = (f"{str(type(self).__name__):s} {name:s} must be " +
                           f"> 0.0, not {str(values):s}.")
                Error(message)
                raise ValueError(message)
            idx = np.argsort(values[:,0])
            return values[idx, :]

        # temperature dependent values as dict [T: value]
        elif type(values) is dict:
            for t, val in values.items():
                if type(t) not in (float, np.float64, np.float128):
                    message = (f"{str(type(self).__name__):s} {name:s} Temperature must be " +
                               f"a float, not {str(type(t).__name__):s}.")
                    Error(message)
                    raise TypeError(message)
                elif type(val) not in (float, np.float64, np.float128):
                    message = (f"{str(type(self).__name__):s} {name:s} must be " +
                               f"a float, not {str(val):s}.")
                    Error(message)
                    raise TypeError(message)
                elif val <= 0.0:
                    message = (f"{str(type(self).__name__):s} {name:s} must be " +
                               f"> 0.0, not {str(val):s}.")
                    Error(message)
                    raise ValueError(message)
            return self._check_values(name, np.array(list(values.items()), dtype=float))

        # temperature dependent values as list of lists [T, value]
        elif type(values) is list:
            for val in values:
                if type(val) is not list or len(val) != 2:
                    message = (f"{str(type(self).__name__):s} {name:s} list must be " +
                               f"a list of lists of len 2 [Temperature, {name:s}], " +
                               f"not {str(val):s}.")
                    Error(message)
                    raise ValueError(message)
                elif type(val[0]) not in (float, np.float64, np.float128):
                    message = (f"{str(type(self).__name__):s} {name:s} Temperature must be " +
                               f"a float, not {str(val[0]):s}.")
                    Error(message)
                    raise TypeError(message)
                elif type(val[1]) not in (float, np.float64, np.float128):
                    message = (f"{str(type(self).__name__):s} {name:s} must be " +
                               f"a float, not {str(val[1]):s}.")
                    Error(message)
                    raise TypeError(message)
            return self._check_values(name, np.array(values, dtype=float))

        # some other type
        else:
            message = (f"{str(type(self).__name__):s} {name:s} must be " +
                       f"either a float, list of floats or a numpy array, not " +
                       f"{str(type(values).__name__):s}.")
            Error(message)
            raise TypeError(message)
        return None

    def _interpolate(self, temperature: float, values: ArrayLike) -> float:
        if type(values) is float:
            return values
        else:
            if temperature is None:
                Warn(f"Material {self.name:s} temperature nod supplied, returning value " +
                     f"for 0. degrees.")
                temperature = 0.
            return np.interp(temperature, values[:,0], values[:,1])

    @classmethod
    def New(cls, name: str, mtype: str, *args, **kwargs):
        if type(mtype) is not str:
            message = f"Material type must be a str, not {str(type(mtype).__name__):s}."
            Error(message)
            raise TypeError(message)
        elif mtype not in cls.__types:
            message = f"Material type {str(mtype):s} not implemented."
            Error(message)
            raise ValueError(message)

        if len(args) > 0 or len(kwargs) > 0:
            if mtype == "ISO":
                if len(args) > 1:
                    return MaterialISO(name, *args)
                elif len(args) == 1:
                    if type(args[0]) is dict:
                        E     = args[0]["E"]
                        nu    = args[0]["nu"]
                        rho   = args[0]["rho"]
                        alpha = args[0]["alpha"]
                        G     = args[0]["G"]
                    elif type(args[0]) is list:
                        E     = args[0][0]
                        nu    = args[0][1]
                        rho   = args[0][2]
                        alpha = args[0][3]
                        G     = args[0][4]
                elif len(kwargs) > 0:
                    E     = kwargs["E"]
                    nu    = kwargs["nu"]
                    rho   = kwargs["rho"]
                    alpha = kwargs["alpha"]
                    if "G" in kwargs.keys():
                        G = kwargs["G"]
                    else:
                        G = None
                return MaterialISO(name, E, nu, rho, alpha, G)
        else:
            message = f"To create a new Material, it's properties must be supplied."
            Error(message)
            raise ValueError(message)

    def __init__(self, name: str, mtype: str, *args, **kwargs):
        super().__init__(name)
        self.type = mtype

    def __repr__(self) -> str:
        return f"<{str(type(self).__name__):s} object>\n  Name: {self.name:s}\n  Type: {self.type:s}"

    @property
    def type(self) -> str:
        return self._type

    @type.setter
    def type(self, mtype: str):
        if type(mtype) is not str:
            message = f"Material type must be a str, not {str(type(mtype).__name__):s}."
            Error(message)
            raise TypeError(message)
        elif mtype not in self.__types:
            message = (f"Material type must be one of ({', '.join(self.__types):s}), " +
                       f"not {str(mtype):s}.")
            Error(message)
            raise TypeError(message)
        else:
            self._type = mtype


# TODO:
# ? strain dependence ?
class MaterialISO(Material):
    def __init__(self, name: str,
                 E: [float | list[float, float] | dict[float, float] | ArrayLike],
                 nu: [float | list[float, float] | dict[float, float] | ArrayLike],
                 rho: [float | list[float, float] | dict[float, float] | ArrayLike],
                 alpha: [float | list[float, float] | dict[float, float] | ArrayLike],
                 G: [float | list[float, float] | dict[float, float] | ArrayLike] = None):
        super().__init__(name, "ISO")
        self.young = E                  # Young's Modulus
        self.poisson = nu               # Poisson's ratio
        self.density = rho              # density
        self.thermal_expansion = alpha  # coefficient of thermal expansion
        self.shear = G                  # Shear Modulus

    @property
    def young(self) -> [float | ArrayLike]:
        return self._E

    @young.setter
    def young(self, E: [float | list[list[float, float]] | dict[float, float] | ArrayLike]):
        self._E = self._check_values("Young's Modulus", E)

    def E(self, temperature: float = None) -> float:
        return self._interpolate(temperature, self.young)

    @property
    def poisson(self) -> [float | ArrayLike]:
        return self._nu

    @poisson.setter
    def poisson(self, nu: [float | list[list[float, float]] | dict[float, float] | ArrayLike]):
        self._nu = self._check_values("Poisson's ratio", nu)

    def nu(self, temperature: float = None) -> float:
        return self._interpolate(temperature, self.poisson)

    @property
    def density(self) -> [float | ArrayLike]:
        return self._rho

    @density.setter
    def density(self, rho: [float | list[list[float, float]] | dict[float, float] | ArrayLike]):
        self._rho = self._check_values("Density", rho)

    def rho(self, temperature: float = None) -> float:
        return self._interpolate(temperature, self.density)

    @property
    def thermal_expansion(self) -> [float | ArrayLike]:
        return self._alpha

    @thermal_expansion.setter
    def thermal_expansion(self, alpha: [float | list[list[float, float]] | dict[float, float] | ArrayLike]):
        self._alpha = self._check_values("Thermal expansion coefficient", alpha)

    def alpha(self, temperature: float = None) -> float:
        return self._interpolate(temperature, self.thermal_expansion)

    @property
    def shear(self) -> [float | ArrayLike]:
        return self._G

    @shear.setter
    def shear(self, G: [float | list[list[float, float]] | dict[float, float] | ArrayLike] = None):
        if G is None:
            if type(self.young) is float:
                if type(self.poisson) is float:
                    self._G = self.young / (2. * (1. + self.poisson))
                else:
                    self._G = self.poisson.copy()
                    self._G[:,1] = self.young / (2. * (1. + self.poisson[:,1]))
            else:
                if type(self.poisson) is float:
                    self._G = self.young.copy()
                    self._G[:,1] = self.young[:,1] / (2. * (1. + self.poisson))
                else:
                    self._G = self.young.copy()
                    for i, t in enumerate(self.young[:,0]):
                        self._G[i,1] = self.young[i,1] / (2. * (1. + self.nu(t)))
            message = (f"Material {self.name:s} Shear Modulus not supplied, calculated based on " +
                        "Young's Modulus and Poisson's ratio.\n")
            message += f"    G = E / (2 * (1 + nu)) = {str(self._G):s}"
            Warn(message)
        else:
            self._G = self._check_values("Shear modulus", G)

        # check material consistency
        if type(self.shear) is float:
            G = self.shear
            E = self.E()
            nu = self.nu()
            _G = E / (2. * (1. + nu))
            if (G - _G) >= 0.001 * G:
                Warn(f"Material {self.name:s} Shear Modulus not consistent with Young's Modulus " +
                     f"and Poisson's ratio: " +
                     f"G = {G:.2f} != {E:.2f} / (2. * (1. + {nu:.3f})) = {_G:.2f}.")
        else:
            for t, G in self.shear[:]:
                E = self.E(t)
                nu = self.nu(t)
                _G = E / (2. * (1. + nu))
                if (G - _G) >= 0.001 * G:
                    Warn(f"Material {self.name:s} Shear Modulus not consistent with " +
                          "Young's Modulus and Poisson's ratio: " +
                         f"G({t:.1f}) = {G:.2f} != {E:.2f} / (2. * (1. + {nu:.3f})) " +
                         f"= {_G:2.f}.")

    def G(self, temperature: float = None) -> float:
        return self._interpolate(temperature, self.shear)

    def aslist(self) -> list:
        material = [self.name, self.type]
        for prop in (self.young, self.poisson, self.density, self.thermal_expansion, self.shear):
            if type(prop) is np.ndarray:
                material.append(prop.tolist())
            else:
                material.append(prop)
        return material

    def asdict(self) -> dict:
        material = {"name": self.name, "type": self.type}
        for name, prop in zip(("E", "nu", "rho", "alpha", "G"),
                (self.young, self.poisson, self.density, self.thermal_expansion, self.shear)):
            if type(prop) is np.ndarray:
                material.setdefault(name, prop.tolist())
            else:
                material.setdefault(name, prop)
        return material

    # TODO:
    # for bar multiply by area
    # for beam multiply by inertia
    def C1D(self, area: float, temperature: float = None) -> float:
        """
        [eps_xx]
        [sigma_xx]
        """
        t = temperature
        return self.E(t)

    # TODO:
    # for in-plane keep as-is
    # for out-of-plane bending multiply by (t ** 3 / 12)
    def C2D(self, temperature: float = None) -> ArrayLike:
        """
        [eps_xx, eps_yy, gamma_xy]
        [sigma_xx, sigma_yy, gamma_xy]
        """
        t = temperature
        n1 = self.nu(t) / (1 - self.nu(t))
        n2 = (1 - 2 * self.nu(t)) / (2 * (1 - self.nu(t)))

        C = np.array([[ 1, n1,  0],
                      [n1,  1,  0],
                      [ 0,  0, n2]], dtype=float)
        C *= self.E(t) * (1 - self.nu(t)) / ((1 + self.nu(t)) * (1 - 2 * self.nu(t)))
        return C

    def C3D(self, temperature: float = None) -> ArrayLike:
        """
        [eps_xx, eps_yy, eps_zz, gamma_xy, gamma_yz, gamma_zx]
        [sigma_xx, sigma_yy, sigma_zz, gamma_xy, gamma_yz, gamma_zx]
        """
        t = temperature
        n1 = self.nu(t) / (1 - self.nu(t))
        n2 = (1 - 2 * self.nu(t)) / (2 * (1 - self.nu(t)))

        C = np.array([[ 1, n1, n1,  0,  0,  0],
                      [n1,  1, n1,  0,  0,  0],
                      [n1, n1,  1,  0,  0,  0],
                      [ 0,  0,  0, n2,  0,  0],
                      [ 0,  0,  0,  0, n2,  0],
                      [ 0,  0,  0,  0,  0, n2]], dtype=float)
        C *= self.E(t) * (1 - self.nu(t)) / ((1 + self.nu(t)) * (1 - 2 * self.nu(t)))

        return C


# TODO:
class MaterialANISO:
    pass


class Materials(ContainerDict):
    def __init__(self, materials: [Material | tuple | list | dict]):
        super().__init__("Name", str, "Material", Material)
        self.materials = materials

    @property
    def materials(self) -> dict:
        return super().asdict()

    @materials.setter
    def materials(self, materials: [Material | tuple | list | dict]):
        if materials is None:
            pass

        elif isinstance(materials, Material):
            self[materials.name] = materials

        elif type(materials) in (tuple, list):
            for material in materials:
                if isinstance(material, Material):
                    self.materials = material
                elif type(material) in (tuple, list):
                    self.materials = Material.New(*material)
                elif type(material) is dict:
                    self.materials = Material.New(**material)
                else:
                    message(f"Wrong definition of Material {str(material):s}")
                    Error(message)
                    raise ValueError(message)

        elif type(materials) is dict:
            for mname, material in materials.items():
                if isinstance(material, Material):
                    self.materials = material
                elif type(material) in (tuple, list):
                    self.materials = Material.New(*material)
                elif type(material) is dict:
                    self.materials = Material.New(**material)
                else:
                    message(f"Wrong definition of Material {mname:n} ({str(material):s}).")
                    Error(message)
                    raise ValueError(message)
        else:
            message(f"Materials must be of type tuple, list, dict or Material, " +
                    f"not {str(type(materials).__name__):s}.")
            Error(message)
            raise ValueError(message)

    def aslist(self) -> list:
        materials = []
        for mname, material in self.materials.items():
            # elements.append([eid, element.type, *element.nodes])
            materials.append([mname, material.type, material.aslist()])
        return materials

    def asdict(self) -> dict:
        materials = {}
        for mname, material in self.materials.items():
            # elements.setdefault(eid, [element.type, *element.nodes])
            materials.setdefault(mname, [material.type, material.aslist()])
        return materials


class Property(Name):
    __types = ("MASS",
               "ROD", "BEAM",
               "SHELL", "TRIA3", "TRIA6", "QUAD4", "QUAD8",
               "SOLID")
               # "TET4", "TET10", "HEX8", "HEX20",
               # "WEDGE6", "PYRA5")

    def _check_value(self, name: str, value: float, allow_zero: bool = False):
        if type(value) not in (float, np.float64, np.float128):
            message = (f"Property {self.name:s} {name:s} Value must be of type float, " +
                       f"not {str(type(value).__name__):s}.")
            Error(message)
            raise TypeError(message)

        elif value < 0. if allow_zero else value <= 0.:
            message = (f"Property {self.name:s} {name:s} Value must be " +
                       f"{'<=' if allow_zero else '<':s} 0., not {str(value):s}.")
            Error(message)
            raise ValueError(message)
        return value

    @classmethod
    def New(cls, name: str, ptype: str, *args, **kwargs):
        if type(ptype) is not str:
            message = (f"Property {name:s} type must be a str, " +
                       f"not {str(type(ptype).__name__):s}.")
            Error(message)
            raise TypeError(message)
        elif ptype not in cls.__types:
            message = (f"Property {name:s} type must be one of " +
                       f"{', '.join(cls.__types):s}, not {str(ptype):s}.")
            Error(message)
            raise ValueError(message)
        else:
            if ptype == "MASS":
                p = PMass(name, *args, **kwargs)
            elif ptype == "ROD":
                p = PRod(name, *args, **kwargs)
            elif ptype == "BEAM":
                p = PBeam(name, *args, **kwargs)
            elif ptype == "SHELL":
                p = PShell(name, *args, **kwargs)
            elif ptype == "TRIA3":
                p = PTria3(name, *args, **kwargs)
            elif ptype == "TRIA6":
                p = PTria6(name, *args, **kwargs)
            elif ptype == "QUAD4":
                p = PQuad4(name, *args, **kwargs)
            elif ptype == "QUAD8":
                p = PQuad8(name, *args, **kwargs)
            elif ptype == "SOLID":
                p = PSolid(name, *args, **kwargs)
            return p

    def __init__(self, name: str, ptype: str):
        super().__init__(name)
        self.type = ptype

    def __repr__(self) -> str:
        return f"<{str(type(self).__name__):s} object>\n  Name: {self.name:s}\n  Type: {self.type:s}"

    @property
    def type(self) -> str:
        return self._type

    @type.setter
    def type(self, ptype: str):
        if type(ptype) is not str:
            message = f"Property type must be a str, not {str(type(ptype).__name__):s}."
            Error(message)
            raise TypeError(message)
        elif ptype not in self.__types:
            message = (f"Property type must be one of ({', '.join(self.__types):s}), " +
                       f"not {str(ptype):s}.")
            Error(message)
            raise TypeError(message)
        else:
            self._type = ptype


class PMass(Property):
    def __init__(self, name: str,
                 mxx: float, myy: float, mzz: float,
                 Ixx: float = None, Iyy: float = None, Izz: float = None,
                 Ixy: float = None, Iyz: float = None, Ixz: float = None):
        super().__init__(name, "MASS")
        if Ixx is None:
            self.mass = [mxx, myy, mzz]
        else:
            if Iyy is None:
                Iyy = 0.
            if Izz is None:
                Izz = 0.
            if Ixy is None:
                Ixy = 0.
            if Iyz is None:
                Iyz = 0.
            if Ixz is None:
                Ixz = 0.
            self.mass = [mxx, myy, mzz, Ixx, Iyy, Izz, Ixy, Iyz, Ixz]

    @property
    def mass(self) -> np.ndarray:
        return self._M

    @mass.setter
    def mass(self, M: [list | tuple | ArrayLike]):
        if type(M) not in (list, tuple, np.ndarray):
            message = (f"{self.type:s} Property {self.name:s} values must be one of " +
                       f" list, tuple or np.ndarray, not {str(type(M).__name__):s}.")
            Error(message)
            raise TypeError(message)

        M = np.array(M, dtype=float).flatten()
        if len(M) not in  (3, 9):
            message = (f"{self.type:s} Property {self.name:s} values must be of length 3, or 9, " +
                       f"[m_xx, m_yy, m_zz [, I_xx, I_yy, I_zz, I_xy, I_yz, I_xz]] not {len(M):n}.")
            Error(message)
            raise ValueError(message)
        elif any([m < 0. for m in M[:6]]):
            message = (f"{self.type:s} Property {self.name:s} values must be >= 0.," +
                       f" not {str(M):n}.")
            Error(message)
            raise ValueError(message)

        else:
            self._M = M

    def M(self) -> ArrayLike:
        mxx, myy, mzz, Ixx, Iyy, Izz, Ixy, Iyz, Ixz = self.mass
        if all([m != 0 for m in self.mass[3:]]):
            M = np.array([[mxx,   0,   0],
                          [  0, myy,   0],
                          [  0,   0, mzz]], dtype = float)
        else:
            M = np.array([[mxx,   0,   0,    0,    0,    0],
                          [  0, myy,   0,    0,    0,    0],
                          [  0,   0, mzz,    0,    0,    0],
                          [  0,   0,   0,  Ixx, -Ixy, -Ixz],
                          [  0,   0,   0, -Ixy,  Iyy, -Iyz],
                          [  0,   0,   0, -Ixz, -Iyz,  Izz]], dtype = float)
        return M

    def aslist(self):
        return [self.name, self.type, *list(self.mass)]

    def asdict(self):
        property = {"name": self.name, "ptype": self.type}
        if len(self.mass) == 3:
            for i, key in enumerate(["mxx", "myy", "mzz"]):
                property.setdefault(key, self.mass[i])
        elif len(self.mass) == 9:
            for i, key in enumerate(["mxx", "myy", "mzz", "Ixx", "Iyy", "Izz", "Ixy", "Iyz", "Ixz"]):
                property.setdefault(key, self.mass[i])
        return property


class PRod(Property):
    def __init__(self, name: str, A: float, nsm: float = None):
        super().__init__(name, "ROD")
        self.A = A
        self.nsm = (nsm if nsm is not None else 0.)

    @property
    def A(self) -> float:
        return self._A

    @A.setter
    def A(self, A: float):
        self._A = self._check_value("Area", A)

    @property
    def nsm(self) -> float:
        return self._nsm

    @nsm.setter
    def nsm(self, nsm: float):
        self._nsm = self._check_value("Nonstructural Mass", nsm, True)

    # TODO:
    # not sure if it should be here
    def M(self, length: float, density: float) -> ArrayLike:
        l = self._check_value("Lenght", length)
        r = self._check_value("Density", density)

        return (self.A * r + self.nsm) * l

    def aslist(self):
        return [self.name, self.type, self.A, self.nsm]

    def asdict(self):
        property = {"name": self.name, "ptype": self.type, "A": self.A, "nsm": self.nsm}
        return property


class PBeam(Property):
    def __init__(self, name: str, A: float, Iyy: float, Izz: float, Iw: float,
                 nsm: float = None):
        super().__init__(name, "BEAM")
        self.A = A
        self.Iyy = Iyy
        self.Izz = Izz
        self.Iw = Iw
        self.nsm = (nsm if nsm is not None else 0.)

    @property
    def A(self) -> float:
        return self._A

    @A.setter
    def A(self, A: float):
        self._A = self._check_value("Area", A)

    @property
    def Iyy(self) -> float:
        return self._Iyy

    @Iyy.setter
    def Iyy(self, Iyy: float):
        self._Iyy = self._check_value("Moment of Inertia y-y", Iyy)

    @property
    def Izz(self) -> float:
        return self._Izz

    @Izz.setter
    def Izz(self, Izz: float):
        self._Izz = self._check_value("Moment of Inertia z-z", Izz)

    @property
    def Iw(self) -> float:
        return self._Iw

    @Iw.setter
    def Iw(self, Iw: float):
        self._Iw = self._check_value("Torsional Moment of Inertia", Iw)

    @property
    def nsm(self) -> float:
        return self._nsm

    @nsm.setter
    def nsm(self, nsm: float):
        self._nsm = self._check_value("Nonstructural Mass", nsm, True)

    def aslist(self):
        return [self.name, self.type, self.A, self.Iyy, self.Izz, self.Iw, self.nsm]

    def asdict(self):
        property = {"name": self.name, "ptype": self.type, "A": self.A,
                    "Iyy": self.Iyy, "Izz": self.Izz, "Iw": self.Iw,
                    "nsm": self.nsm}
        return property


class PShell(Property):
    def __init__(self, name: str, t: [float | tuple | list | ArrayLike],
                 nsm: float = None, ptype: str = None, num_nodes: int = 1):
        if ptype is None:
            super().__init__(name, "SHELL")
            self.__n = len(t) if type(t) in (tuple, list, np.ndarray) else 1
        else:
            super().__init__(name, ptype)
            self.__n = num_nodes
        self.t = t
        self.nsm = (nsm if nsm is not None else 0.)

    @property
    def t(self) -> np.ndarray:
        if self.__n == 1:
            return self._t[0]
        else:
            return self._t

    @t.setter
    def t(self, t: [float | list | tuple | ArrayLike]):
        if type(t) is float:
            t = [t] * self.__n
        elif type(t) in (tuple, list):
            t = list(t)
        elif type(t) is np.ndarray:
            t = t.tolist()
        else:
            message = (f"{self.type:s} Property {self.name:s} thickness must be one of " +
                       f"(float, tuple, list, np.ndarray), not {str(type(t).__name__):s}.")
            Error(message)
            raise TypeError(message)

        if t[0] is None:
            message = (f"{self.type:s} Property {self.name:s} t1 must be a float, " +
                       f"not {str(t[0]):s}.")
            Error(message)
            raise TypeError(message)

        if len(t) != self.__n:
            message = (f"{self.type:s} Property {self.name:s} thickness must be a list " +
                       f"of len {self.__n:n}, not {len(t):s}.")
            Error(message)
            raise TypeError(message)
        # TODO:
        t = list(map(lambda tt: t[0] if tt is None else tt, t))
        # for i, tt in enumerate(t):
        #     if tt is None:
        #         t[i] = t[0]
        #         Info(f"{type(self).__name__:s} {self.name:s} thickness {i + 1:n} set to t1 " +
        #              f"({t[0]:f}).")
        self._t = np.array([self._check_value(f"Thickness {i+1:n}", t[i]) for i in range(len(t))],
                           dtype=float)

    @property
    def nsm(self) -> float:
        return self._nsm

    @nsm.setter
    def nsm(self, nsm: float):
        self._nsm = self._check_value("Nonstructural Mass", nsm, True)

    def aslist(self):
        if type(self.t) in (float, np.float64, np.float128):
            return [self.name, self.type, self.t, self.nsm]
        else:
            return [self.name, self.type, *self.t, self.nsm]

    def asdict(self):
        property = {"name": self.name, "ptype": self.type, "nsm": self.nsm}
        # print(f"{self.t = }")
        if type(self.t) in (float, np.float64, np.float128):
            property.setdefault("t", self.t)
        else:
            for i, t in enumerate(self.t):
                property.setdefault(f"t{i+1:n}", t)
        return property


class PTria3(PShell):
    def __init__(self, name: str,
                 t1: float, t2: float = None, t3: float = None,
                 nsm: float = None):
        super().__init__(name, [t1, t2, t3], nsm, "TRIA3", 3)


class PTria6(PShell):
    def __init__(self, name: str,
                 t1: float,        t2: float = None, t3: float = None,
                 t4: float = None, t5: float = None, t6: float = None,
                 nsm: float = None):
        super().__init__(name, [t1, t2, t3, t4, t5, t6], nsm, "TRIA6", 6)


class PQuad4(PShell):
    def __init__(self, name: str,
                 t1: float, t2: float = None, t3: float = None, t4: float = None,
                 nsm: float = None):
        super().__init__(name, [t1, t2, t3, t4], nsm, "QUAD4", 4)


class PQuad8(PShell):
    def __init__(self, name: str,
                 t1: float,        t2: float = None, t3: float = None, t4: float = None,
                 t5: float = None, t6: float = None, t7: float = None, t8: float = None,
                 nsm: float = None):
        super().__init__(name, [t1, t2, t3, t4, t5, t6, t7, t8], nsm, "QUAD8", 8)


class PSolid(Property):
    def __init__(self, name: str, nsm: float = None):
        super().__init__(name, "SOLID")
        self.nsm = (nsm if nsm is not None else 0.)

    @property
    def nsm(self) -> float:
        return self._nsm

    @nsm.setter
    def nsm(self, nsm: float):
        self._nsm = self._check_value("Nonstructural Mass", nsm, True)

    def aslist(self):
        return [self.name, self.type, self.nsm]

    def asdict(self):
        property = {"name": self.name, "ptype": self.type, "nsm": self.nsm}
        return property


class Properties(ContainerDict):
    def __init__(self, properties: [Property | tuple | list | dict]):
        super().__init__("Name", str, "Property", Property)
        self.properties = properties

    @property
    def properties(self) -> dict:
        return super().asdict()

    @properties.setter
    def properties(self, properties: [Property | tuple | list | dict]):
        if properties is None:
            pass

        elif isinstance(properties, Property):
            self[properties.name] = properties

        elif type(properties) in (tuple, list):
            for property in properties:
                if isinstance(property, Property):
                    self.properties = property
                elif type(property) in (tuple, list):
                    self.properties = Property.New(*property)
                elif type(property) is dict:
                    self.properties = Property.New(**property)
                else:
                    message(f"Wrong definition of Property {str(property):s}")
                    Error(message)
                    raise ValueError(message)

        elif type(properties) is dict:
            for name, property in properties.items():
                if isinstance(property, Property):
                    self.properties = property
                elif type(property) in (tuple, list):
                    self.properties = Property.New(*property)
                elif type(property) is dict:
                    self.properties = Property.New(**property)
                else:
                    message(f"Wrong definition of Property {name:n} ({str(property):s}).")
                    Error(message)
                    raise ValueError(message)
        else:
            message(f"Properties must be of type tuple, list, dict or Property, " +
                    f"not {str(type(properties).__name__):s}.")
            Error(message)
            raise ValueError(message)

    def aslist(self) -> list:
        properties = []
        for name, property in self.properties.items():
            properties.append([name, property.type, property.aslist()])
        return properties

    def asdict(self) -> dict:
        properties = {}
        for name, property in self.properties.items():
            properties.setdefault(name, [property.type, property.aslist()])
        return properties


# TODO:
class LoadN(Id):
    def __init__(self, nid: int, lpat: int,
                 Fx: float = None, Fy: float = None, Fz: float = None,
                 Mx: float = None, My: float = None, Mz: float = None):
        super().__init__(nid)
        self.lpat = lpat
        self.F = [Fx, Fy, Fz, Mx, My, Mz]

    def __add__(self, nodal_load):
        if type(nodal_load) is type(self):
            if self.node != nodal_load.node:
                message = (f"Cannot add loads on two different nodes {self.node:n} and " +
                           f"{nodal_load.node:n}.")
                Error(message)
                raise ValueError(message)
            else:
                self._F += nodal_load.F
                return self
        elif type(nodal_load) in (tuple, list, np.ndarray) and len(nodal_load) == 6:
            self._F += nodal_load
            return self
        else:
            message = (f"Nodal Load addition can be only done between two Nodal Loads " +
                       f"or a Nodal Load and a vector of type (tuple, list, np.ndarray), " +
                       f"not {str(type(nodal_load).__name__):s}.")
            Error(message)
            raise TypeError(message)

    def __iadd__(self, nodal_load):
        return self.__add__(nodal_load)

    def __sub__(self, nodal_load):
        if type(nodal_load) is type(self):
            if self.node != nodal_load.node:
                message = (f"Cannot add loads on two different nodes {self.node:n} and " +
                           f"{nodal_load.node:n}.")
                Error(message)
                raise ValueError(message)
            else:
                self._F -= nodal_load.F
                return self
        elif type(nodal_load) in (tuple, list, np.ndarray) and len(nodal_load) == 6:
            self._F -= nodal_load
            return self
        else:
            message = (f"Nodal Load addition can be only done between two Nodal Loads " +
                       f"or a Nodal Load and a vector of type (tuple, list, np.ndarray), " +
                       f"not {str(type(nodal_load).__name__):s}.")
            Error(message)
            raise TypeError(message)

    def __isub__(self, nodal_load):
        return self.__sub__(nodal_load)

    @property
    def node(self) -> int:
        return self.id

    @node.setter
    def node(self, nid: int):
        self.id = nid

    @property
    def lpat(self) -> int:
        return self._lpat

    @lpat.setter
    def lpat(self, lpat: int):
        if type(lpat) is not int:
            message = (f"{str(type(self).__name__):s} LPAT ID must be of type int, " +
                       f"not {str(type(lpat).__name__):s}.")
            Error(message)
            raise TypeError(message)
        elif lpat < 1:
            message = f"{str(type(self).__name__):s} LPAT ID must be > 0, not {lpat:n}."
            Error(message)
            raise ValueError(message)
        else:
            self._lpat = lpat

    @property
    def F(self) -> np.ndarray:
        return self._F

    @F.setter
    def F(self, F: [tuple | list | np.ndarray]):
        if type(F) not in (tuple, list, np.ndarray):
            message = (f"Load on Node ID {self.node:n} must be of type " +
                       f"(tuple, list, np.ndarray), not {str(type(F).__name__):s}.")
            Error(message)
            raise TypeError(message)
        elif len(F) != 6:
            message = (f"Load on Node ID {self.node:n} must be of lenght 6 " +
                       f", not {len(F):n}.")
            Error(message)
            raise ValueError(message)
        elif all([f is None or f == 0. for f in F]):
            message = (f"Load on Node ID {self.node:n} must be of lenght > 0. " +
                       f", not {str(F):n}.")
            Error(message)
            raise ValueError(message)

        if type(F) in (tuple, list):
            F = list(map(lambda f: 0. if f is None else f, F))

        if any([type(f) not in (float, np.float64, np.float128) for f in F]):
            message = (f"Load on Node ID {self.node:n} must be a vector of floats " +
                       f", not ({', '.join([type(f).__name__ for f in F]):s}).")
            Error(message)
            raise TypeError(message)
        else:
            self._F = np.array(F, dtype=float)

    def aslist(self) -> list:
        return [self.node, self.lpat, *(self.F.tolist())]

    def asdict(self) -> dict:
        return {"nid": self.node,
                "lpat": self.lpat,
                "Fx": self.F[0],
                "Fy": self.F[0],
                "Fz": self.F[0],
                "Mx": self.F[0],
                "My": self.F[0],
                "Mz": self.F[0]}


# TODO:
class LoadE:
    pass


# TODO:
# ?? LPAT ??
# group loads based on lpat
class LoadsN(Id, ContainerDict):
    def __init__(self, lpat: int, loads: [LoadN | tuple | list | dict]):
        Id.__init__(self, lpat)
        ContainerDict.__init__(self, "ID", int, "Nodal Load", LoadN)
        self.loads = loads

    @property
    def lpat(self) -> int:
        return self.id

    @lpat.setter
    def lpat(self, lpat: int):
        self.id = lpat

    @property
    def loads(self) -> dict[int, LoadN]:
        return super().asdict()

    @loads.setter
    def loads(self, loads: [LoadN | list[LoadN] | dict[int, ArrayLike] | ArrayLike]):
        if type(loads) is LoadN:
            # print(f"{loads = }")
            if self.lpat != loads.lpat:
                message = (f"{type(self).__name__:s} cannot have more than one LPAT " +
                           f"({self.lpat:n}, {loads.lpat:n})")
                Error(message)
                raise ValueError(message)
            elif loads.node in self.keys():
                self[loads.node].F += loads.F
            else:
                self[loads.node] = loads

        # TODO:
        elif type(loads) in (tuple, list):
            for load in loads:
                if type(load) is LoadN:
                    self.loads = load

                elif type(load) in (tuple, list):
                    self.loads = LoadN(*load)

                elif type(load) is dict:
                    self.loads = LoadN(**load)

                else:
                    message = (f"Nodal Load {str(load):s} must be of type " +
                               f"(tuple, list, dict, LoadN), not {str(type(load).__name__):s}.")
                    Error(message)
                    raise TypeError(message)

        elif type(loads) is dict:
            for nid, load in loads.items():
                if type(load) is LoadN:
                    self.loads = load

                elif type(load) in (list, tuple):
                    self.loads = LoadN(*list(load))

                elif type(load) is dict:
                    self.loads = LoadN(**list(load))

                else:
                    message = (f"Nodal Load {str(load):s} must be of type " +
                               f"(tuple, list, dict, LoadN), not {str(type(load).__name__):s}.")
                    Error(message)
                    raise TypeError(message)

        else:
            message = (f"Nodal Load {str(loads):s} must be one of (tuple, list, dict, LoadN) " +
                       f"types, not {str(type(nodes).__name__):s}.")
            Error(message)
            raise TypeError(message)

    def aslist(self) -> list:
        loads = []
        for nid in self.loads.keys():
            loads.append(self.loads[nid].aslist())
        return loads

    def asdict(self) -> dict:
        loads = {}
        for nid in self.loads.keys():
            loads.setdefault(nid, self.loads[nid].asdict())
        return loads


# TODO:
class LoadsE:
    pass


# TODO:
class Loading:
    def __init__(self, loads: [LoadN | LoadE | tuple | list | dict] = None):
        self._nodal = ContainerDict("LPat", int, "Nodal Load", (LoadsN))
        self._elemental = ContainerDict("LPat", int, "Elemental Load", (LoadsE))
        self._loads = loads

    @property
    def nodal(self) -> ContainerDict:
        return self._nodal.asdict()

    @nodal.setter
    def nodal(self, load: [tuple | list | dict | LoadsN | LoadN]):
        if load is None:
            pass

        # LoadN
        elif type(load) is LoadN:
            nid = load.id
            lpat = load.lpat
            if lpat not in self._nodal.keys():
                self._nodal[lpat] = LoadsN(lpat, load)

            else:
                self._nodal[lpat] = load

        # LoadsN
        elif type(load) is LoadsN:
            lpat = load.lpat
            if lpat not in self._nodal.keys():
                self._nodal[lpat] = load

            else:
                self._nodal[lpat] += load

        # [load_1, load_2, ... , load_n]
        # TODO:
        elif type(load) in (tuple, list):
            # [nid, lpat, Fx .. Mz]
            if type(load[0]) is int and len(load) == 8:
                self.nodal = LoadN(*load)

            else:
                for l in load:
                    # [LoadN_1, ... , LoadN_2]
                    if type(l) is LoadN:
                        self.nodal = l

                    # [[nid, lpat, Fx .. Mz]]
                    elif type(l) in (tuple, list):
                        lpat = l[1]
                        self.nodal = LoadN(*l)

                    # [{nid, lpat, Fx .. Mz}]
                    elif type(l) is dict:
                        self.nodal = LoadN(**l)

                    else:
                        message = (f"{type(self).__name__:s} Nodal Loads type must be a list " +
                                   f"of lists not {type(load).__name__:s}({type(l).__name__:s}) " +
                                   f"- {str(l):s}.")
                        Error(message)
                        raise TypeError(message)

        elif type(load) is dict:
            for lpat, l in load.items():
                # {lpat: [[nid, lpat, Fx ... Mz]]}
                if type(l) in (tuple, list):
                    self.nodal = l

                # {lpat: {nid: {nid, lpat, Fx .. Mz}}}
                elif type(l) is dict:
                    self.nodal = LoadsN(lpat, l)

                # {lpat: LoadN | LoadsN}
                elif type(l) in (LoadN, LoadsN):
                    self.nodal = l

                else:
                    message = (f"{type(self).__name__:s} Nodal Loads type must be a tuple, " +
                               f"list, dict, LoadN or LoadsN, not " +
                               f"{type(load).__name__:s}({type(l).__name__:s}) - {str(l):s}.")
                    Error(message)
                    raise TypeError(message)

        else:
            message = (f"{type(self).__name__:s} Nodal Loads type must be one of " +
                       f"(tuple, list, dict, LoadsN, LoadN), not {type(load).__name__:s}.")
            Error(message)
            raise TypeError(message)

    @property
    def elemental(self) -> ContainerDict:
        # TODO:
        return self._elemental.asdict()

    @elemental.setter
    def elemental(self, load: (tuple, list, dict, LoadsN, LoadN)):
        # TODO:
        self._elemental = load

    @property
    def _loads(self) -> tuple[LoadsN, LoadsE]:
        return (self.nodal, self.elemental)

    @_loads.setter
    def _loads(self, load: [tuple | list | dict | LoadsN | LoadsE | LoadN | LoadE]):
        if load is None:
            pass

        elif type(load) is LoadN:
            nid = load.id
            lpat = load.lpat

            if lpat not in self.nodal.keys():
                self.nodal[lpat] = LoadsN(load)
            else:
                self.nodal[lpat].loads = load

        elif type(load) is LoadE:
            eid = load.id
            lpat = load.lpat

            if lpat not in self.elemental.keys():
                self.elemental[lpat] = LoadsE(load)
            else:
                self.elemental[lpat].loads = load

# ??????


# TODO:
class Constraint:
    pass


# TODO:
class Suppressed(Constraint):
    pass


# TODO:
class Prescribed(Constraint):
    pass


# TODO:
class Structure:
    pass


# TODO:
class Constraints:
    pass


# TODO:
class System:
    pass


# TODO:
class Situation:
    pass


# TODO:
class Modification:
    pass


# TODO:
class Model:
    pass


# TODO:
class ResultsN:
    pass


# TODO:
class ResultsE:
    pass


# TODO:
class Results:
    pass


# TODO:
class Solver:
    pass


# TODO:
class Static(Solver):
    pass


# TODO:
class Nonlinear(Solver):
    pass


# TODO:
class Eigenvalues(Solver):
    pass


# TODO:
class FrequencyResponse(Solver):
    pass


# TODO:
class SteadyState(Solver):
    pass


# TODO:
class Spectral(Solver):
    pass


# TODO:
class Thermal(Solver):
    pass


# TODO:
class Optimisation(Solver):
    pass




if __name__ == "__main__":
    # coors = np.random.rand(20, 3)
    # print(f"{coors = }")
    # nodes = Nodes({i+1: coors for i, coor in enumerate(coors)})
    # print(f"{nodes.coors = }")
    # print(f"{nodes.extents = }")
    # Info("Info")
    # Warn("Warning")
    # Error("Error")

    # m = MaterialISO("stahl", 210000., 0.3, 7.85e-9, 1.2e-5)

    # pb = PBeam("ipe", 1000., 1000000., 1000000., 1000000., 0.05)

    import os
    import sys
    sys.path.append("../../../tests/unit/")
    from _models import nodal_loads

    loads = LoadsN(nodal_loads[0]["lpat"], nodal_loads)
    print(loads.aslist())
    print(loads.asdict())

