"""
Collection of miscellaneous objects.

ID         - Base class for all ID based objects.
Label      - Base class for all Label based objects.
Coordinate - Base class for formatting all coordinates.
"""

from typing import Union
import numpy as np

import defaults


class ID:
    """
    Base class for all ID based objects.
    """
    def __init__(self, id: Union[int, np.int_], label: str=None, id_gt_0: bool=True):
        """
        Constructor

        In:
            id    - an int > 0
            label - an optional label, str
            id_gt_0 - must ID be greater than 0 (True/False)
        """
        if type(id_gt_0) is bool:
            self.__id_gt_0 = id_gt_0
        else:
            self.__id_gt_0 = True

        self.id = id
        self.label = label

    @property
    def id(self) -> int:
        """
        Returns the Object ID.
        """
        return self.__id

    @id.setter
    def id(self, id: Union[int, np.int_]):
        """
        Sets the Object ID.
        """
        if self.__id_gt_0:
            _id = 1
        else:
            _id = 0

        if type(id) not in (int, np.dtype(int)):
            raise ValueError(f'{str(type(self)):s} ID {str(id):s} must be an int, not {str(type(id)):s}.')
        elif id < _id:
            raise ValueError(f'{str(type(self)):s} ID {id:n} must be > {_id-1:n}.')
        else:
            self.__id = id

    @property
    def label(self) -> str:
        """
        Returns the object label.
        """
        return self.__label

    @label.setter
    def label(self, label: Union[str, np.str_]):
        """
        Sets the object label.
        """
        if label is None:
            self.__label = None
        elif type(label) not in (str, np.dtype(str)):
            raise ValueError(f'{str(type(self)):s} label type must be a str, not {str(type(label)):s}.')
        elif label == '':
            raise ValueError(f'{str(type(self)):s} label cannot be an empty string.')
        else:
            check = [c in defaults.CHARSET for c in label]
            if not all(check):
                raise ValueError(f'{str(type(self)):s} label characters can be only "{defaults.CHARSET:s}".')
            else:
                self.__label = label


class Label:
    """
    Base class for all label based objects.
    """
    def __init__(self, label: Union[str, np.str_]):
        """
        Constructor

        In:
            label - a str
        """
        self.label = label

    @property
    def label(self) -> str:
        """
        Returns the object label.
        """
        return self.__label

    @label.setter
    def label(self, label: Union[str, np.str_]):
        """
        Sets the object label.
        """
        if label is None:
            raise ValueError(f'{str(type(self)):s} label type must be a str, not {str(type(label)):s}.')
        elif type(label) not in (str, np.dtype(str)):
            raise ValueError(f'{str(type(self)):s} label type must be a str, not {str(type(label)):s}.')
        elif label == '':
            raise ValueError(f'{str(type(self)):s} label cannot be an empty string.')
        else:
            self.__label = label


class Coordinate:
    """
    Class for coordinate checking.
    """
    def __init__(self, x: Union[float, np.float_, tuple, list, np.ndarray],
                 y: Union[float, np.float_]=None, z: Union[float, np.float_]=None):
        """
        Coordinate constructor.

        In:
            x - x coordinate as float or a tuple/list/np.ndarray of size 3 of all coordinates
            y - y coordinate
            z - z coordinate
        """
        self.__x = None
        self.__y = None
        self.__z = None

        if y is None and z is None and type(x) in (tuple, list, np.ndarray):
            self.x = x
        elif type(x) in (tuple, list, np.ndarray):
            raise ValueError(f'Coordinates must be exactly 3, not x: {str(x):s}, y: {str(y):s}, z: {str(z):s}.')
        else:
            self.x = x
            self.y = y
            self.z = z

    def __add__(self, coor2: Union[float, np.float_, tuple, list, np.ndarray]):
        if type(coor2) in (float, np.float_):
            return type(self)(self.X + coor2)
        elif type(coor2) is not type(self):
            coor2 = type(self)(coor2)
        return type(self)(self.X + coor2.X)

    def __iadd__(self, coor2: Union[float, np.float_, tuple, list, np.ndarray]):
        if type(coor2) in (float, np.float_):
            return type(self)(self.X + coor2)
        elif type(coor2) is not type(self):
            coor2 = type(self)(coor2)
        return type(self)(self.X + coor2.X)

    def __sub__(self, coor2: Union[float, np.float_, tuple, list, np.ndarray]):
        if type(coor2) in (float, np.float_):
            return type(self)(self.X - coor2)
        elif type(coor2) is not type(self):
            coor2 = type(self)(coor2)
        return type(self)(self.X - coor2.X)

    def __isub__(self, coor2: Union[float, np.float_, tuple, list, np.ndarray]):
        if type(coor2) in (float, np.float_):
            return type(self)(self.X - coor2)
        elif type(coor2) is not type(self):
            coor2 = type(self)(coor2)
        return type(self)(self.X - coor2.X)

    def __mul__(self, number: Union[float, np.float_]):
        if type(number) in (float, np.float_):
            return type(self)(self.X * number)
        else:
            raise ValueError(f'Coordinate can be only multiplied by a float, '
                              'not a {str(type(number)):s}.')

    def __imul__(self, number: Union[float, np.float_]):
        if type(number) in (float, np.float_):
            return type(self)(self.X * number)
        else:
            raise ValueError(f'Coordinate can be only multiplied by a float, '
                              'not a {str(type(number)):s}.')

    def __truediv__(self, number: Union[float, np.float_]):
        if type(number) in (float, np.float_):
            return type(self)(self.X / number)
        else:
            raise ValueError(f'Coordinate can be only divided by a float, '
                              'not a {str(type(number)):s}.')

    def __itruediv__(self, number: Union[float, np.float_]):
        if type(number) in (float, np.float_):
            return type(self)(self.X / number)
        else:
            raise ValueError(f'Coordinate can be only divided by a float, '
                              'not a {str(type(number)):s}.')

    @property
    def x(self) -> float:
        """
        Returns the x coordinate.
        """
        return self.__x

    @x.setter
    def x(self, x: Union[float, np.float_, tuple, list, np.ndarray]):
        """
        Sets the x coordinate (or all 3, if a tuple/list/np.ndarray is fed in)

        In:
            x - x coordinate as a float or a tuple/list/np.ndarray of size 3 of all coordinates
        """
        if x is None:
            raise ValueError('X coordinate must be either float or a list of 3 coordinates.')
        elif type(x) in (tuple, list, np.ndarray):
            x = np.array(x, dtype=float)
            if x.shape != (3, ):
                raise ValueError(f'X coordinate must be either float or a list of 3 coordinates, not {x.shape:n}.')
            else:
                self.x = x[0]
                self.y = x[1]
                self.z = x[2]
        elif type(x) not in (float, np.dtype(float)):
            raise ValueError(f'X coordinate must be either float or a list of 3 coordinates, not {str(type(x)):s}.')
        else:
            self.__x = x

    @property
    def y(self) -> float:
        """
        Returns the y coordinate.
        """
        return self.__y

    @y.setter
    def y(self, y: Union[float, np.float_]):
        """
        Sets the y coordinate.

        In:
            y - y coordinate
        """
        if y is None:
            raise ValueError(f'Y coordinate must be a float, not {str(type(y)):s}.')
        elif type(y) not in (float, np.dtype(float)):
            raise ValueError(f'Y coordinate must be a float, not {str(type(y)):s}.')
        else:
            self.__y = y

    @property
    def z(self) -> float:
        """
        Returns the z coordinate.
        """
        return self.__z

    @z.setter
    def z(self, z: Union[float, np.float_]):
        """
        Sets the z coordinate.

        In:
            z - z coordinate
        """
        if z is None:
            raise ValueError(f'Z coordinate must be a float, not {str(type(z)):s}.')
        elif type(z) not in (float, np.dtype(float)):
            raise ValueError(f'Z coordinate must be a float, not {str(type(z)):s}.')
        else:
            self.__z = z

    @property
    def X(self) -> np.ndarray:
        """
        Returns all 3 coordinates as a np.ndarray.
        """
        return np.array([self.x, self.y, self.z])


