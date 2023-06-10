#!/usr/bin/env python3

import os
import sys
import typing
import numpy as np
import scipy.integrate


class RefFrame:
    def __init__(self, name: str, cog: np.ndarray = None, velo: np.ndarray = None,
                 form: str = "cartesian", angle: str = "radians"):
        self._name = str(name)
        if cog is None:
            cog = np.zeros(3, dtype=float)
        elif type(cog) is not np.ndarray:
            cog = np.array(cog, dtype=float)

        cog = np.hstack([cog, [0.] * (3 - cog.shape[0])]).astype(float)
        cog[2] = np.radians(cog[2]) if angle == "degrees" else cog[2]

        if form == "cartesian":
            self.cog = cog

        elif form == "cylindrical":
            x = cog[0] * np.cos(np.radians(cog[1]))
            y = cog[0] * np.sin(np.radians(cog[1]))
            alpha = cog[2]
            self.cog = np.array([x, y, alpha], dtype=float)

        if velo is None:
            velo = np.zeros(3, dtype=float)
        elif type(velo) is not np.ndarray:
            velo = np.array(velo, dtype=float)

        velo = np.hstack([velo, [0.] * (3 - velo.shape[0])]).astype(float)
        velo[2] = np.radians(velo[2]) if angle == "degrees" else velo[2]

        if form == "cartesian":
            self.v = velo

        elif form == "cylindrical":
            vx = velo[0] * np.cos(np.radians(velo[1]))
            vy = velo[0] * np.sin(np.radians(velo[1]))
            wz = velo[2]
            self.v = np.array([vx, vy, wz], dtype=float)


    def __str__(self) -> str:
        return f"{self.name:s} {type(self).__name__:s}"

    @property
    def name(self) -> str:
        return self._name

    @property
    def cog(self) -> np.ndarray:
        return self._cog

    @cog.setter
    def cog(self, cog: [np.ndarray, list, tuple]):
        if type(cog) in (tuple, list):
            cog = np.array(cog, dtype=float)
        elif type(cog) is np.ndarray:
            cog = cog.astype(float)

        self._cog = np.hstack([cog, [0.] * (3 - cog.shape[0])]).astype(float)

    @property
    def x(self) -> float:
        return self.cog[0]

    @x.setter
    def x(self, x: float):
        self._cog[0] == x

    @property
    def y(self) -> float:
        return self.cog[1]

    @x.setter
    def y(self, y: float):
        self._cog[1] == y

    @property
    def alpha(self) -> float:
        return self._cog[2]

    @alpha.setter
    def alpha(self, alpha: float):
        self._cog[2] == alpha

    @property
    def r(self) -> float:
        return (self.x ** 2. + self.y ** 2.) ** 0.5

    @property
    def phi(self) -> float:
        return np.arctan2(self.y, self.x)

    @property
    def v(self) -> np.ndarray:
        return self._velo

    @v.setter
    def v(self, velo: np.ndarray):
        if type(velo) in (tuple, list):
            velo = np.array(velo, dtype=float)
        elif type(velo) is np.ndarray:
            velo = velo.astype(float)

        self._velo = np.hstack([velo, [0.] * (3 - velo.shape[0])]).astype(float)

    @property
    def vx(self) -> float:
        return self._velo[0]

    @vx.setter
    def vx(self, vx: float):
        self._velo[0] = vx

    @property
    def vy(self) -> float:
        return self._velo[1]

    @vy.setter
    def vy(self, vy: float):
        self._velo[1] = vy

    @property
    def wz(self) -> float:
        return self._velo[2]

    @wz.setter
    def wz(self, wz: float):
        self._velo[2] = wz

    @property
    def T(self) -> np.ndarray:
        T = np.array([[ np.cos(self.alpha), np.sin(self.alpha), 0.],
                      [-np.sin(self.alpha), np.cos(self.alpha), 0.],
                      [                 0.,                 0., 1.]], dtype=float)
        return T

    def gcs2lcs(self, vector: np.ndarray) -> np.ndarray:
        if type(vector) is not np.ndarray:
            vector = np.array(vector, dtype=float)
        if vector.shape[0] == 2:
            cog = self.cog[:2]
            T = self.T[:2,:2]
        else:
            cog = self.cog
            T = self.T
        return T @ (vector - cog)

    def lcs2gcs(self, vector: np.ndarray) -> np.ndarray:
        if type(vector) is not np.ndarray:
            vector = np.array(vector, dtype=float)
        if vector.shape[0] == 2:
            cog = self.cog[:2]
            T = self.T[:2,:2]
        else:
            cog = self.cog
            T = self.T
        return T.T @ vector + cog



class Part(RefFrame):
    def __init__(self, name: str, cog: np.ndarray = None, velo: np.ndarray = None,
                 mass: float = 0., inertia: float = 0., rotation_ratio: float = 0.,
                 form: str = "cartesian", angle: str = "radians"):
        super().__init__(name, cog, velo, form, angle)

        self.M = mass
        self.J = inertia
        self.rr = rotation_ratio

    @property
    def M(self) -> float:
        return self._mass

    @M.setter
    def M(self, mass: float) -> float:
        self._mass = float(mass)

    @property
    def J(self) -> float:
        return self._inertia

    @J.setter
    def J(self, inertia: float) -> float:
        self._inertia = float(inertia)

    @property
    def Jcog(self) -> float:
        return self.J + self.M * self.r ** 2

    @property
    def w(self) -> float:
        return self._omega

    @property
    def rr(self) -> float:
        return self._rotation_ratio

    @rr.setter
    def rr(self, rotation_ratio: float = 0.):
        self._rotation_ratio = float(rotation_ratio)

    @property
    def rotating(self) -> bool:
        return self.rr != 0.



class Force(RefFrame):
    def __init__(self, name: str, cog: np.ndarray = None,
                 force: np.ndarray = None,
                 weight_function: object = lambda t: 1.,
                 form: str = "cartesian", angle: str = "radians"):

        super().__init__(name, cog, None, form, angle)

        if force is None:
            force = np.array([0., 0., 0.], dtype=float)

        if form == "cartesian":
            self.F = force

        elif form == "cylindrical":
            Fx = force[0] * np.cos(force[1])
            Fy = force[1] * np.cos(force[2])
            Mz = force[2]
            self.F = np.array([Fx, Fy, Mz], dtype=float)

        self._weight_function = weight_function

    @property
    def F(self) -> float:
        return self._vector

    @F.setter
    def F(self, force: np.ndarray):
        if type(force) is not np.ndarray:
            force = np.array(force)
        force = np.hstack([force, [0.] * (3 - force.shape[0])]).astype(float)
        self._vector = force

    @property
    def Fx(self) -> float:
        return self._vector[0]

    @property
    def Fy(self) -> float:
        return self._vector[1]

    @property
    def Fr(self) -> float:
        return (self.Fx ** 2. + self.Fy ** 2.) ** 0.5

    @property
    def Fphi(self) -> float:
        return np.arctan2(self.Fx, self.Fy)

    @property
    def Mz(self) -> float:
        return self._vector[2]

    @property
    def wt(self) -> object:
        return self._weight_function

    def Ft(self, t: float) -> float:
        return self.F * self.wt(t)

    def Ftx(self, t: float) -> float:
        return self.Fx * self.wt(t)

    def Fty(self, t: float) -> float:
        return self.Fy * self.wt(t)

    def Ftr(self, t: float) -> float:
        return self.Fr * self.wt(t)

    def Ftphi(self, t: float) -> float:
        return self.Fphi * self.wt(t)

    def Mtz(self, t: float) -> float:
        return self.Mz * self.wt(t)



class HandsArmSystem:
    def __init__(self, name: str, gro: np.ndarray = None, grs: np.ndarray = None,
                 t_init: float = 0., form: str = "cartesian", angle: str = "radians"):
        self.gro = RefFrame("GRO", gro, None, form, angle)
        self.grs = RefFrame("GRS", grs, None, form, angle)

        self._t0 = t_init
        self._Mtot = 0.
        self._Jtot = 0.
        self._Ektx = 0.
        self._Ekty = 0.
        self._Ekrz = 0.

    @property
    def t0(self) -> float:
        return self._t0

    def set_max_energy(self, Mtot: float, Jtot: float, vCOGx: float, vCOGy, wCOGz: float):
        self._Mtot = Mtot
        self._Jtot = Jtot

        self._Ektx = 0.5 * Mtot * vCOGx ** 2.
        self._Ekty = 0.5 * Mtot * vCOGy ** 2.
        self._Ekrz = 0.5 * Jtot * wCOGz ** 2.

    def init(self, Mtot: float = 0., Jtot: float = 0., Ektx: float = 0.,
                   Ekty: float = 0., Ekrz: float = 0.):
        self._Mtot = Mtot
        self._Jtot = Jtot

        self._Ektx = Ektx
        self._Ekty = Ekty
        self._Ekrz = Ekrz

    def force(self, t: float):
        # t ? shouldn't t be offset by clamping ?

        # effective radius
        rHAM = (self.gro.r + self.grs.r) / 1000. / 0.0254 # [in]

        # maxumum moment transferable by the HAM system
        Mmax = 15. * rHAM

        # recalculate energies into imperial units
        Ektx = self._Ektx / 1000. / 0.1129848290276167 # [ln * in]
        Ekty = self._Ekty / 1000. / 0.1129848290276167 # [ln * in]
        Ekrz = self._Ekrz / 1000. / 0.1129848290276167 # [ln * in]

        # transform experimental kickback energies into those suitable for HAM
        Etx = Ektx
        Ety = min(Ektx, Ekrz / 3.)
        Erz = Ekrz - Ety

        # work
        Wtot = 9.80665 * self.Mtot / 4.4482216152605 # [lbf]
        Jtot = Jtot * 1000. / 0.1130                    # [lb*in*s^2]

        J1 = 25.554 * (Erz ** 0.25) * ((rHAM * self.Jtot) ** 0.5)
        J2 = -3.614 * ((Wtot * Etx) ** 0.25) + 6.012
        J3 =  0.0713 * J1 + 3.047 * Wtot - 5.215 * (Ety ** 0.5) + 2.989

        K1 = 0.1070
        K3 = 0.1128
        O1 = 0.0000
        O3 = 0.32754 * Wtot + 2.063

        # calculation of effective force (x-component)
        if t < 0.1800:
            Fxcog = J2
        elif J2 > 0.:
            Fxcog = max(0., J2 - 200. * t + 36.)
        else:
            Fxcog = max(0., J2 + 200. * t - 36.)

        # calculation of effective force (y-component)
        if t < 0.1128:
            Fycog = -1. * (O3 - J3) * ((t - K3) ** 2. - 2. * (t / K3)) + J3
        elif J2 > 0.:
            Fycog = O3
        else:
            Fycog = max(0., O3 - 200. * t + 30.)

        # calculation of effective moment
        if t < 0.1070:
            Mzcog = -1. * (O1 - J1) * ((t / K1) ** 2. - 2. * (t / K1)) + J1
        elif J2 > 0.:
            Mzcog = O1
        else:
            Mzcog = min(Mmax, O1 - 1741. * t - 278.56)

        # recalc back to N and Nmm
        Fxcog = -4.4482216152605 * Fxcog * 1000.          # N
        Fycog =  4.4482216152605 * (Fycog - Wtot) * 1000. # N
        Mzcog = -0.1129848290276 * Mzcog * 1.E6           # Nmm

        return np.array([Fxcog, Fycog, Mzcog])

    def Ftx(self, t: float):
        return self.force(t)[0]

    def Fty(self, t: float):
        return self.force(t)[1]

    def Mtz(self, t: float):
        return self.force(t)[2]



class Machine(RefFrame):
    def __init__(self, name: str,
                 cog: np.ndarray = [0., 0., 0.],
                 velo: np.ndarray = [0., 0., 0.],
                 parts: list = [],
                 forces: list = [],
                 kuw_forces: list = [],
                 ham: HandsArmSystem = None):
        super().__init__(name, cog, velo, "cartesian", "radians")
        self._parts = {}
        self._forces = {}
        self._kuw_forces = {}
        self.parts = parts
        self.forces = forces
        self.kuw_forces = kuw_forces

        self._wr = 0. # angular velocity of KUW


    def __str__(self) -> str:
        line = f"{self.name:s} {type(self).__name__:s}:\n"

        header = ["Name", "x", "y", "alpha", "vx", "vy", "wz", "M", "J", "Jcog"]
        header_line = " | " + " | ".join([f"{h:^13s}" for h in header]) + " |"
        delimiter_line = " +-" + "-+-".join(["-" * 13 for h in header]) + "-+"
        format_line = " | {0:13s} | " + " | ".join(["{" + str(i+1) + ":13.5f}" for i in range(len(header) - 1)]) + " |"

        line += delimiter_line.replace("-", "=") + "\n"
        line += header_line + "\n"
        line += delimiter_line + "\n"

        for part in self.parts:
            line += format_line.format(part.name, part.x, part.y, part.alpha, part.vx, part.vy, part.wz, part.M, part.J, part.Jcog) + "\n"

        line += delimiter_line.replace("-", "=") + "\n"

        line += f" Mtot = {self.M:13.5f}\n"
        line += f" Jtot = {self.Jtot:13.5f}\n"
        line += f" Jrot = {self.Jrot:13.5f}\n"

        return line


    def __getitem__(self, key: str, item: str = None) -> Part:
        if key == "part":
            ret = self._parts
        elif key == "force":
            ret = self._forces
        elif key == "kuw_force":
            ret = self._kuw_forces

        if item is None:
            return list(ret.values())
        else:
            return ret[item.upper()]

    def __setitem__(self, key: str, item: str) -> Part:
        if key == "part":
            self._parts.setdefault(item.name.upper(), item)
        elif key == "force":
            self._forces.setdefault(item.name.upper(), item)
        elif key == "kuw_force":
            self._kuw_forces.setdefault(item.name.upper(), item)

    @property
    def parts(self) -> list:
        return list(self._parts.values())

    @parts.setter
    def parts(self, parts: [Part]):
        for part in parts:
            self._parts.setdefault(part.name.upper(), part)

    @property
    def forces(self) -> list:
        return list(self._forces.values())

    @forces.setter
    def forces(self, forces: [Force]):
        for force in forces:
            self._forces.setdefault(force.name.upper(), force)

    @property
    def kuw_forces(self) -> list:
        return list(self._kuw_forces.values())

    @forces.setter
    def kuw_forces(self, forces: [Force]):
        for force in forces:
            self._kuw_forces.setdefault(force.name.upper(), force)

    @property
    def M(self) -> float:
        mass = 0.
        for part in self.parts:
            mass += part.M

        return mass

    @property
    def Jtot(self) -> float:
        inertia = 0.
        for part in self.parts:
            if part.rotating:
                inertia += part.M * part.r ** 2.
            else:
                inertia += part.Jcog

        return inertia

    @property
    def Jrot(self) -> float:
        inertia = 0.
        for part in self.parts:
            if part.rotating:
                inertia += part.J

        return inertia

    @property
    def Ekx(self):
        return 0.5 * self.M * self.vx ** 2

    @property
    def Eky(self):
        return 0.5 * self.M * self.vy ** 2

    @property
    def Ekz(self):
        return 0.5 * self.Jtot * self.wz ** 2

    # TODO: transform forces to GCS
    def Ftx(self, t: float):
        Ftx = 0.
        for force in self.forces:
            Ftx += (self.T @ force.T @ force.Ft(t))[0]
        return Ftx

    # TODO: transform forces to GCS
    def Fty(self, t: float):
        Fty = 0.
        for force in self.forces:
            Fty += (self.T @ force.T @ force.Ft(t))[1]
        return Fty

    def Mtz(self, t: float):
        Mtz = 0.
        for force in self.forces:
            Mtz += force.Mtz(t)
        return Mtz

    # rotation of KUW
    def Mtr(self, t: float):
        Mtr = 0.
        for force in self.kuw_forces:
            Mtr += force.Mtz(t) + force.r * force.Ftphi(t)
        return Mtr

    def solve(self, tmax: float, t0: float = 0., tHAM: float = 0., maxiter: int = 10000,
              x0: np.ndarray = None, v0: np.ndarray = None, wr0: float = None):
        if x0 is None:
            x0 = np.array([0., 0., 0.], dtype=float)
        if v0 is None:
            v0 = np.array([0., 0., 0.], dtype=float)
        if wr0 is None:
            wr0 = 0.

        def diff(t, y):
            x, y, phi, vx, vy, wz, wr = y

            self.x = x
            self.y = y
            self.phi = phi

            self.vx = vx
            self.vy = vy
            self.wz = wz

            self.wr = wr

            if t < self.ham.t0:
                self.ham.init(self.M, self.Jtot, self.Ekx, self.Eky, self.Ekz)

            dx = vx
            dy = vy
            dphi = wz

            dvx = self.Ftx(t) / self.M
            dvy = self.Fty(t) / self.M
            dwz = self.Mtz(t) / self.Jtot

            dwr = self.Mtr(t) / self.Jrot

            return np.array([dx, dy, dphi, dvx, dvy, dwz, dwzr], dtype=float)

        self.cog = x0
        self.v = v0
        self.wr = wr0

        y = np.array([self.x, self.y, self.phi, self.vx, self.vy, self.wz, self.wr])

        breakpoint()
        solver = scipy.integrate.RK45(diff, t0, y, tmax)

        t_values = []
        y_values = []

        header = ["Time", "x_COG", "y_COG", "a_COG", "vx_COG", "vy_COG", "wz_COG", "wz_ROT"]

        header_line = " | " + " | ".join([f"{h:^13s}" for h in header]) + " |"
        delimiter_line = " +-" + "-+-".join(["-" * 13 for h in header]) + "-+"
        format_line = " | " + " | ".join(["{" + str(i) + ":13.5f}" for i in range(len(header))]) + " |"

        # while True:
        print(delimiter_line.replace("-", "="))
        print(header_line)
        print(delimiter_line)
        for i in range(10000):
            solver.step()
            t_values.append(solver.t)
            y_values.append(solver.y)
            print(format_line.format(solver.t, *solver.y.tolist()))
            if solver.status == "finished":
                break
        print(delimiter_line.replace("-", "="))



if __name__ == "__main__":

    def clamping(t: float) -> float:
        tUp = 0.055
        tContact = 0.06
        tDown = 0.05
        tClamp = tUp + tContact + tDown

        if t < tUp:
            f = np.sin((np.pi * t) / (2 * tUp)) ** 2
        elif t <= (tUp + tContact):
            f = 1.
        elif t <= tClamp:
            f = np.cos((np.pi * (t - tUp - tContact)) / (2 * tDown)) ** 2
        else:
            f = 0.

    rpm = 9300.
    tClamp = 0.055 + 0.060 + 0.050
    tBreak = 0.1
    breaking = lambda t: 1. if t >= tBreak else 0.
    mu = 0.51
    ue = 2.37
    wr0 = 2. * np.pi * rpm / 60.

    #          Name    cog(r, phi, theta)       velo    mass       inertia    uebersetzung
    kuw = Part("KUW", [ 96.0, -2.792, 0.], [0., 0., wr0], 1337.2E-6,   1180893.9E-6, 1., form="cylindrical", angle="radians")
    breakpoint()
    sch = Part("SCH", [225.0,  0.137, 0.], [0., 0., wr0 / ue], 1668.0E-6,  25634000.0E-6, ue, form="cylindrical", angle="radians")
    awl = Part("AWL", [225.0,  0.137, 0.], [0., 0., 0.],  643.8E-6,    601657.3E-6, 0., form="cylindrical", angle="radians")
    rts = Part("RTS", [ 43.7, -3.073, 0.], [0., 0., 0.], 9065.9E-6, 257954230.0E-6, 0., form="cylindrical", angle="radians")

    # define the clamping force relative to SCH as it is tangential to it
    Fcg = Force("CLAMP", [167., 1.1247, 0.], [0., 500. * mu, 0.], weight_function=clamping, form="cylindrical", angle="radians")
    # move it
    Fcg.cog = Fcg.cog + sch.cog
    Fcr = Force("CLAMP", [167., 1.1247, 0.], [0., 500. * mu * ue, 0.], weight_function=clamping, form="cylindrical", angle="radians")

    # define the breaking moment
    Fbg = Force("BREAK", [0., 0., 0.], [0., 0., -3500.], weight_function=breaking, form="cartesian", angle="radians")
    Fbr = Force("BREAK", [0., 0., 0.], [0., 0.,  3500.], weight_function=breaking, form="cartesian", angle="radians")

    # define HandsArmSystem
    ham = HandsArmSystem("HAM", gro=[174.5,  1.571, 0.], grs=[285.3,  2.711, 0.],
                         t_init=tClamp, form="cylindrical", angle="radians")

    # put it together
    tsa400 = Machine("tsa400",
                     cog=[0., 0., np.radians(-9.0)],
                     velo=[0., 0., 0.],
                     parts=[kuw, sch, awl, rts],
                     forces=[Fcg, Fbg],
                     kuw_forces=[Fcr, Fbr],
                     ham=ham)

    print(str(tsa400))
    breakpoint()
    tsa400.solve(0.3, wr0=-2 * np.pi * rpm / 60.)


