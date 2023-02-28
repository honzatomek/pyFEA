#!/home/honza/Programming/pyFEA/__venv__/bin/python

import os
import sys
import numpy as np

from typing import Union
import weakref

SRC = os.path.realpath(__file__)
sys.path.append(os.path.dirname(os.path.dirname(SRC)))

# import pyFEA.defaults as default
from pyFEA.defaults import *
from pyFEA.misc import Coordinate
from pyFEA.misc import ID


class CSys(ID):
    def _init(self):
        """
        _init function placeholder.
        """
        pass

    def __init__(self, id: INT,
                       label: Union[str, np.str_]=None):
        super().__init__(id=id, label=label, id_gt_0=False)
        self.__csys_ref = None
        self.__initialised = False

    @property
    def initialised(self) -> bool:
        return self.__initialised

    @initialised.setter
    def initialised(self, initialised: bool):
        if type(initialised) is bool:
            self.__initialised = initialised
        else:
            raise ValueError(f'CSYS ID {self.id} initialised type must be a bool, '
                             f'not {str(type(initialised)):s}.')

    def consolidate(self, csys):
        """
        Links the current CSYS To its definition CSYS.

        In:
            csys - a reference to a definition CSYS object
        """
        if not isinstance(csys, CSys):
            raise ValueError(f'CSys ID {self.id:n} consolidation error. CSys supplied must be '
                             f'of type CSys, not {str(type(csys)):s}.')
        elif csys.id != self.csys:
            raise ValueError(f'CSys ID {self.id:n} consolidation error. CSys ID should be '
                             f'{self.csys:n}, not {csys.id:n}/')
        else:
            self.__csys_ref = weakref.ref(csys)

    def transform(self):
        """
        Transforms the current CSYS defintion to be defined based on global CSYS.
        """
        if self.__csys_ref is None:
            raise ValueError(f'CSys ID {self.id:n} has not yet been consolidated.')
        else:
            self.__initialised = False
            if self.__csys_ref().csys != 0:
                if self.__csys_ref().initialised:
                    self.__csys_ref().transform()
            self.O  = self.__csys_ref().to_global(self.O)
            self.P1 = self.__csys_ref().to_global(self.P1)
            self.P2 = self.__csys_ref().to_global(self.P2)
            self._init()

    # TODO:
    def to_global2(self, coors):
        coors = self.to_global(coors)
        if self.csys != 0 and self.initialised:
            coors = self.__csys_ref().to_global2(coors)
        return coors


class Cartesian(CSys):
    @classmethod
    def gcsys(cls):
        return cls(0, [0., 0., 0.], [1., 0., 0.], [0., 1., 0.], form='xy', label='GCSYS')

    def _init(self):
        if self.form == 'xz':
            x = (self.P1 - self.O) / np.linalg.norm(self.P1 - self.O)
            z = (self.P2 - self.O) / np.linalg.norm(self.P2 - self.O)
            y = np.cross(z, x)
            z = np.cross(x, y)
        elif self.form == 'zx':
            z = (self.P1 - self.O) / np.linalg.norm(self.P1 - self.O)
            x = (self.P2 - self.O) / np.linalg.norm(self.P2 - self.O)
            y = np.cross(z, x)
            z = np.cross(x, y)
        elif self.form == 'xy':
            x = (self.P1 - self.O) / np.linalg.norm(self.P1 - self.O)
            y = (self.P2 - self.O) / np.linalg.norm(self.P2 - self.O)
            z = np.cross(x, y)
            y = np.cross(z, x)
        else:
            raise NotImplementedError(f'Cartesian CSYS formulation {form:s} is '
                                       'not implemented, use xz, zx or xy.')

        x = x / np.linalg.norm(x)
        y = y / np.linalg.norm(y)
        z = z / np.linalg.norm(z)

        self.__T = np.vstack((x, y, z))
        self.initialised = True

    def __init__(self, id: INT,
                       origin: Union[tuple, list, np.ndarray],
                       point1: Union[tuple, list, np.ndarray],
                       point2: Union[tuple, list, np.ndarray],
                       form: Union[str, np.str_]='xz',
                       csys: INT=0,
                       label: Union[str, np.str_]=None):

        super().__init__(id=id, label=label)

        self.O = origin
        self.P1 = point1
        self.P2 = point2
        self.form = form
        self.csys = csys

        self._init()

    @property
    def O(self) -> np.ndarray:
        '''
        Returns the CSYS origin in global coordinates.
        '''
        return self.__o.X

    @O.setter
    def O(self, origin: list):
        '''
        Sets the CSYS origin in global coordinates.
        '''
        if type(origin) is Coordinate:
            self.__o = origin
        else:
            self.__o = Coordinate(origin)

        if self.initialised:
            self._init()

    @property
    def P1(self) -> np.ndarray:
        '''
        Returns the first defintion point of the CSYS.
        '''
        return self.__p1.X

    @P1.setter
    def P1(self, point1: list):
        '''
        Sets the first defintion point of the CSYS.
        '''
        if type(point1) is Coordinate:
            self.__p1 = point1
        else:
            self.__p1 = Coordinate(point1)

        if self.initialised:
            self._init()

    @property
    def P2(self):
        '''
        Returns the second defintion point of the CSYS.
        '''
        return self.__p2.X

    @P2.setter
    def P2(self, point2):
        '''
        Sets the second defintion point of the CSYS.
        '''
        if type(point2) is Coordinate:
            self.__p2 = point2
        else:
            self.__p2 = Coordinate(point2)

        if self.initialised:
            self._init()

    @property
    def form(self):
        '''
        Returns the CSYS formulation.
        '''
        return self.__form

    @form.setter
    def form(self, form):
        '''
        Sets the CSYS formulation.
        '''
        if type(form) is not str:
            raise ValueError(f'{str(type(self)):s} CSYS ID {self.id:n} formulation type must be a string, not {str(type(form)):s}.')
        elif form not in ('xy', 'xz', 'zx'):
            raise ValueError(f'{str(type(self)):s} CSYS ID {self.id:n} formulation must be one of "xy", "xz" and "zx", not {form:s}.')
        else:
            self.__form = form

        if self.initialised:
            self._init()

    @property
    def T(self):
        '''
        returns the direction sine and cosine matrix transforming global
        coordinates to local ones:

        x_l = T @ (x_g - x_o),

        where x_o is the cylindrical CSYS origin
        '''
        return self.__T

    def to_local(self, coors):
        '''
        Transforms a point in global cartesian CSYS to local cartesian CSYS.

        x_l = T @ (x_g - x_o)

        where x_o is the cylindrical CSYS origin
        '''
        if type(coors) is Coordinate:
            x_g = coors.X
        else:
            x_g = Coordinate(coors).X

        return self.T @ (x_g - self.O)

    def to_global(self, coors):
        '''
        Transforms a point in local cylindrical CSYS to global cartesian CSYS.

        x_g = T^T @ x_l + x_o

        where x_o is the cylindrical CSYS origin
        '''
        if type(coors) is Coordinate:
            x_l = coors.X
        else:
            x_l = Coordinate(coors).X

        return self.T.T @ x_l + self.O


class Cylindrical(Cartesian):
    def _init(self):
        if self.form == 'rz':
            r = (self.P1 - self.O) / np.linalg.norm(self.P1 - self.O)
            z = (self.P2 - self.O) / np.linalg.norm(self.P2 - self.O)
            phi = np.cross(z, r)
            z = np.cross(r, phi)
        elif self.form == 'zr':
            z = (self.P1 - self.O) / np.linalg.norm(self.P1 - self.O)
            r = (self.P2 - self.O) / np.linalg.norm(self.P2 - self.O)
            phi = np.cross(z, r)
            z = np.cross(r, phi)
        elif self.form == 'rphi':
            r = (self.P1 - self.O) / np.linalg.norm(self.P1 - self.O)
            phi = (self.P2 - self.O) / np.linalg.norm(self.P2 - self.O)
            z = np.cross(r, phi)
            phi = np.cross(z, r)
        else:
            raise NotImplementedError(f'Cartesian CSYS formulation {form:s} is '
                                       'not implemented, use xz, zx or xy.')

        r = r / np.linalg.norm(r)
        phi = phi / np.linalg.norm(phi)
        z = z / np.linalg.norm(z)

        self.__T = np.vstack((r, phi, z))
        self.initialised = True

    def __init__(self, id, origin, point1, point2, form='rz', label=None):
        super().__init__(id=id, origin=origin, point1=point1, point2=point2, form=form, label=label)

    @property
    def form(self):
        '''
        Returns the CSYS formulation.
        '''
        return self.__form

    @form.setter
    def form(self, form):
        '''
        Sets the CSYS formulation.
        '''
        if type(form) is not str:
            raise ValueError(f'{str(type(self)):s} CSYS ID {self.id:n} formulation type must be a string, not {str(type(form)):s}.')
        elif form not in ('rphi', 'rz', 'zr'):
            raise ValueError(f'{str(type(self)):s} CSYS ID {self.id:n} formulation must be one of "rphi", "rz" and "zr", not {form:s}.')
        else:
            self.__form = form

    @property
    def T(self):
        return self.__T

    def to_local(self, coors):
        '''
        Transforms a point in global cartesian CSYS to local cartesian CSYS.

        x_l = T @ (x_g - x_o)

        where x_o is the cylindrical CSYS origin,
              r_c = (x_l^2 + y_l^2)^(1/2),
              phi_c = atan2(y_l, x_l),
              z_c = z_l
        '''
        if type(coors) is Coordinate:
            x_g = coors.X
        else:
            x_g = Coordinate(coors).X

        x_l = self.T @ (x_g - self.O)

        return np.array([(x_l[0] ** 2 + x_l[1] ** 2) ** 0.5, np.arctan2(x_l[1], x_l[0]), x_l[2]], dtype=FLOAT)

    def to_global(self, coors):
        '''
        Transforms a point in local cylindrical CSYS to global cartesian CSYS.

        x_g = T^T @ x_l + x_o

        where x_o is the cylindrical CSYS origin,
              x_l = r_c * cos(phi_c),
              y_l = r_c * sin(phi_c),
              z_l = z_c
        '''
        if type(coors) is Coordinate:
            x_c = coors.X
        else:
            x_c = Coordinate(coors).X

        x_l = np.array([x_c[0] * np.cos(x_c[1]), x_c[0] * np.sin(x_c[1]), x_c[2]], dtype=FLOAT)

        return self.T.T @ x_l + self.O



if __name__ == '__main__':
    from random import randint

    # TODO:
    # global csys = 0
    csys0 = Cartesian.gcsys()
    # csys 1 defined in csys 0, at point [0., 0., 0.], rotated counterclockwise by 90°
    csys1 = Cartesian(1, [0., 0., 0.], [0., 20., 0.], [0., 0., 10.], form='xz', csys=0)
    # csys 2 defined in csys 0, at point [0., 0., 0.], rotated clockwise by 90°
    csys2 = Cartesian(2, [0., 0., 0.], [0., -10., 0.], [0., 0., 10.], form='xz', csys=1)

    csys = dict()
    for c in (csys0, csys1, csys2):
        csys[c.id] = c

    for c in csys.keys():
        csys[c].consolidate(csys[csys[c].csys])

    coor = np.array([FLOAT(randint(-255, 255)),
                     FLOAT(randint(-255, 255)),
                     FLOAT(randint(-255, 255))], dtype=FLOAT)

    print(f'{coor = }')
    print(f'{csys1.to_global2(coor) = }')
    print(f'{csys2.to_global2(coor) = }')

    print(f'{str(INT)}')


