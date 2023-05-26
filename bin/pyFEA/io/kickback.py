#!/usr/bin/python3

from math import pi, sin, cos, atan2, radians, degrees
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class Part:
    def __init__(self, name, x_cog=None, y_cog=None, mass=None, moment_of_inertia=None,
                angular_velocity=None, rpm=None, frequency=None, radius=None, angle=None):
        '''
        angular velocity, rpm or frequency are in GCS (counterclockwise = +,
        clockwise = -)
        '''
        self.__name = name
        if x_cog is not None:
            self.__x_cog = x_cog
            self.__y_cog = y_cog
            self.__radius = None
            self.__angle = None
        else:
            self.__radius = radius
            self.__angle = angle
            self.__x_cog = None
            self.__y_cog = None
        self.__mass = mass
        self.__moment_of_inertia = moment_of_inertia

        if angular_velocity is not None:
            self.__w = angular_velocity
        elif rpm is not None:
            self.__w = 2.0 * pi * rpm / 60.
        elif frequency is not None:
            self.__w = 2.0 * pi * frequency
        else:
            self.__w = 0.


    def __str__(self):
        return f'{self.name:16s}{self.x:12.4E}{self.y:12.4E}{self.r:12.4E}{self.phi:12.4E}{self.M:12.4E}{self.J:12.4E}{self.w:12.4E}'

    def cog(self, x, y):
        if self.__radius is None:
            self.__radius = ((self.__x_cog - x) ** 2 + (self.__y_cog - y) ** 2) ** (1/2)
            self.phi = atan2(self.__y_cog - y, self.__x_cog - x)

    @property
    def x(self):
        if self.__x_cog is None:
            self.__x_cog = self.r * np.cos(self.phi)
        return self.__x_cog

    @property
    def y(self):
        if self.__y_cog is None:
            self.__y_cog = self.r * np.sin(self.phi)
        return self.__y_cog

    @property
    def name(self):
        return self.__name

    @property
    def r(self):
        return self.__radius

    @property
    def phi(self):
        return self.__angle

    @phi.setter
    def phi(self, angle):
        self.__angle = angle % (2 * pi)

    @property
    def M(self):
        return self.__mass

    @property
    def J(self):
        return self.__moment_of_inertia

    @property
    def w(self):
        return self.__w


class Machine:
    def __init__(self):
        self.x = 0.
        self.y = 0.
        self.mTOT = 0.
        self.JTOT = 0.

        self.parts = []

        self.t = None
        self.vx = None
        self.vy = None
        self.rx = None
        self.ry = None
        self.w = None
        self.g = None

        self.max_angle = 0.

        self.fig = None
        self.ax = None

    def print(self, type='info'):
        if type == 'info':
            header = '{0:16s}{1:^12s}{2:^12s}{3:^12s}{4:^12s}{5:^12s}{6:^12s}{7:^12s}'.format('name', 'x_cog', 'y_cog', 'r', 'phi', 'm', 'J', 'w')
            divider = '-' * len(header)
            print(divider)
            print(header)
            print(divider)
            for part in self.parts:
                print(str(part))
            print(divider)
            summary = '{0:16s}{1:12.4E}{2:12.4E}{3:12s}{4:12s}{5:12.4E}{6:12.4E}'.format('summe', self.x, self.y, '', '', self.mTOT, self.JTOT)
            print(summary)

        elif type == 'results':
            header = '{0:16s}{1:>12s}{2:>12s}{3:>12s}{4:>12s}{5:>12s}{6:>12s}'.format('time', 'vx', 'vy', 'rx', 'ry', 'w', 'gamma')
            divider = '-' * len(header)
            print(divider)
            print(header)
            print(divider)
            for i in range(len(self.t)):
                print(f'{self.t[i]:16.8f}{self.vx[i]:12.4E}{self.vy[i]:12.4E}{self.rx[i]:12.4E}{self.ry[i]:12.4E}{self.w[i]:12.4E}{self.g[i]:12.4E}')
            print(divider)

    def add_part(self, *args, **kwargs):
        part = Part(*args, **kwargs)
        self.parts.append(part)

    def parts2cog(self, x_cog=None, y_cog=None):
        if x_cog is None or y_cog is None:
            x_cog = 0.
            y_cog = 0.
            for part in self.parts:
                self.mTOT += part.M
                x_cog += part.M * part.x
                y_cog += part.M * part.y
            x_cog /= self.mTOT
            y_cog /= self.mTOT
        else:
            for part in self.parts:
                self.mTOT += part.M

        self.x = x_cog
        self.y = y_cog

        for part in self.parts:
            part.cog(x_cog, y_cog)
            self.JTOT += part.M * part.r ** 2 + part.J

    def time(self, T0, T1, NT, init=True):
        if init:
            return np.linspace(T0, T1, NT, dtype=float).T, (T1 - T0) / NT
        else:
            return np.linspace(T0, T1, NT, dtype=float).T[1:], (T1 - T0) / NT

    def impulse(self, t, T0, T1, force_amplitude):
        if type(t) in [list, np.ndarray]:
            return np.array([self.impulse(time, T0, T1, force_amplitude) for time in t], dtype=float)
        else:
            if t < T0 or t > T1:
                return 0.
            else:
                return force_amplitude * sin(pi * (t - T0) / (T1 - T0)) ** 2

    def integrate(self, values, dt):
        return np.cumsum(values) * dt

    def save_results(self, t, vx, vy, rx, ry, w, g):
        if self.t is None:
            self.t = t
            self.vx = vx
            self.vy = vy
            self.rx = rx
            self.ry = ry
            self.w = w
            self.g = g
        else:
            self.t = np.hstack([self.t, t])
            self.vx = np.hstack([self.vx, vx])
            self.vy = np.hstack([self.vy, vy])
            self.rx = np.hstack([self.rx, rx])
            self.ry = np.hstack([self.ry, ry])
            self.w = np.hstack([self.w, w])
            self.g = np.hstack([self.g, g])

        self.max_angle = degrees(np.max(self.g))

    def gravity_forces(self, ag, t, dt):
        # integral of gravity forces and moments
        # int_F = ∫ F(t) dt = F * t
        # gravity = m * a * r
        int_F_x = self.integrate(np.ones(t.size) * self.mTOT * ag[0], dt)
        int_F_y = self.integrate(np.ones(t.size) * self.mTOT * ag[1], dt)
        mom = 0.
        if self.t is None:
            g = 0.
        else:
            g = self.g[-1]

        for p in self.parts:
            # x direction
            if p.phi + g < pi:
                mom +=  -p.M * ag[0] * p.r
            elif p.phi + g > pi:
                mom +=  p.M * ag[0] * p.r
            # y direction
            if p.phi + g < pi / 2. or p.phi + g > 3 / 4 * pi:
                mom += p.M * ag[1] * p.r
            elif p.phi + g > pi / 2. and p.phi + g < 3 / 4 * pi:
                mom += -p.M * ag[1] * p.r
        int_M_z = self.integrate(np.ones(t.size) * mom , dt)

        return int_F_x, int_F_y, int_M_z

    def movement(self, T0, T1, NT, F=(0., 0., 0.), ag=(0., 0.), v0=(0., 0.), r0=(0., 0.), w0=0., gamma0=0., brake=False):
        if self.t is None:
            t, dt = self.time(T0, T1, NT)
        else:
            t, dt = self.time(T0, T1, NT, init=False)

        # initial conditions
        if self.t is None:
            t0 = t[0]
            vcx, vcy, rcx, rcy, w, gamma = 0., 0., 0., 0., 0., 0.
        else:
            t0 = self.t[-1]
            vcx, vcy, rcx, rcy, w, gamma = self.vx[-1], self.vy[-1], self.rx[-1], self.ry[-1], self.w[-1], self.g[-1]

        # integral of forces and moments
        # int_F = ∫ F(t) dt = F * t
        int_F_x = self.integrate(self.impulse(t, T0, T1, F[0]), dt)
        int_F_y = self.integrate(self.impulse(t, T0, T1, F[1]), dt)
        int_M_z = self.integrate(self.impulse(t, T0, T1, F[2]), dt)

        # integral of gravity forces and moments
        int_F_gx, int_F_gy, int_M_gz = self.gravity_forces(ag, t, dt)
        int_F_x += int_F_gx
        int_F_y += int_F_gy
        int_M_z += int_M_gz

        # velocities of COG
        # v = v0 + ∫ F(t) / m dt = v0 + F * t / m    <-- F = m * (dv / dt) = m * a
        vcx += v0[0] + int_F_x / self.mTOT
        vcy += v0[1] + int_F_y / self.mTOT
        vcz = np.zeros(vcx.size)

        # position of COG
        # r = r0 + ∫v dt = r0 + v * t
        rcx += r0[0] + self.integrate(vcx, dt)
        rcy += r0[1] + self.integrate(vcy, dt)
        rcz = np.zeros(rcx.size)

        # angular velocity = cross product of linear velocity and radius
        # w = L / J (angular velocity = angular momentum / rotational inertia)
        # w = r x v / |r|^2
        # rcv = r x v = w * |r|^2
        # L = r * m * v = m * r^2 * w = ∫ dm r^2 w = J * w
        # ∑ M_z = dL / dt = d(Jw) / dt = J * dw / dt + w * dJ / dt
        # dL = ∫ ∑ M_z dt
        rcv = np.cross(np.vstack([rcx, rcy, rcz]).T, np.vstack([vcx, vcy, vcz]).T)
        w += w0 + (self.mTOT * rcv[:, 2] + int_M_z) / self.JTOT
        if brake:
            for part in self.parts:
                w -= part.J * (np.linspace(part.w, 0., t.size).T - part.w) / self.JTOT

        # angular position
        # gamma = gamma0 + ∫ w dt
        # w = d gamma / dt
        gamma += gamma0 + self.integrate(w, dt)

        self.save_results(t, vcx, vcy, rcx, rcy, w, gamma)

        return (t, np.vstack([vcx, vcy]), np.vstack([rcx, rcy]), w, gamma)

    def force_and_moment_impulse(self, Fx, Fy, Mz, T0, T1, NT,
                                 v0=(0., 0.), r0=(0., 0.), w0=0., gamma0=0.):
        # times
        t, dt = self.time(T0, T1, NT)

        # integral of forces and moments
        # int_F = ∫ F(t) dt = F * t
        int_F_x = self.integrate(self.impulse(t, T0, T1, Fx), dt)
        int_F_y = self.integrate(self.impulse(t, T0, T1, Fy), dt)
        int_M_z = self.integrate(self.impulse(t, T0, T1, Mz), dt)

        # velocities of COG
        # v = v0 + ∫ F(t) / m dt = v0 + F * t / m    <-- F = m * (dv / dt) = m * a
        vcx = v0[0] + int_F_x / self.mTOT
        vcy = v0[1] + int_F_y / self.mTOT
        vcz = np.zeros(vcx.size)

        # position of COG
        # r = r0 + ∫v dt = r0 + v * t
        rcx = r0[0] + self.integrate(vcx, dt)
        rcy = r0[1] + self.integrate(vcy, dt)
        rcz = np.zeros(rcx.size)

        # angular velocity = cross product of linear velocity and radius
        # w = L / J (angular velocity = angular momentum / rotational inertia)
        # w = r x v / |r|^2
        # L = r * m * v = m * r^2 * w = ∫ dm r^2 w = J * w
        # ∑ M_z = dL / dt = d(Jw) / dt = J * dw / dt + w * dJ / dt
        # dL = ∫ ∑ M_z dt
        rcv = np.cross(np.vstack([rcx, rcy, rcz]).T, np.vstack([vcx, vcy, vcz]).T)
        w = w0 + (self.mTOT * rcv[:, 2] + int_M_z) / self.JTOT

        # angular position
        gamma = gamma0 + self.integrate(w, dt)

        self.save_results(t, vcx, vcy, rcx, rcy, w, gamma)

        return (t, np.vstack([vcx, vcy]), np.vstack([rcx, rcy]), w, gamma)

    def free_movement(self, T0, T1, NT, v0=(0., 0.), r0=(0., 0.), w0=0., gamma0=0.):
        # times
        t, dt = self.time(T0, T1, NT)

        # velocities of COG
        vcx = v0[0] * np.ones(t.size)
        vcy = v0[1] * np.ones(t.size)

        # position of COG
        rcx = r0[0] + self.integrate(vcx, dt)
        rcy = r0[1] + self.integrate(vcy, dt)

        # angular velocity
        w = w0 * np.ones(t.size)

        # angular position
        gamma = gamma0 + self.integrate(w, dt)

        self.save_results(t, vcx, vcy, rcx, rcy, w, gamma)

        return (t, np.vstack([vcx, vcy]), np.vstack([rcx, rcy]), w, gamma)

    def braking(self, T0, T1, NT, v0=(0., 0.), r0=(0., 0.), w0=0., gamma0=0.):
        # times
        t, dt = self.time(T0, T1, NT)

        # velocities of COG
        vcx = v0[0] * np.ones(t.size)
        vcy = v0[1] * np.ones(t.size)

        # position of COG
        rcx = r0[0] + self.integrate(vcx, dt)
        rcy = r0[1] + self.integrate(vcy, dt)

        # angular velocity
        w = w0
        for part in self.parts:
            w -= part.J * (np.linspace(part.w, 0., t.size).T - part.w) / self.JTOT

        # angular position
        gamma = gamma0 + self.integrate(w, dt)

        self.save_results(t, vcx, vcy, rcx, rcy, w, gamma)

        return (t, np.vstack([vcx, vcy]), np.vstack([rcx, rcy]), w, gamma)

    def gravity(self, T0, T1, NT, v0=(0., 0.), r0=(0., 0.), w=0., gamma0=0., ag=(0.,-9810.)):
        # times
        t, dt = self.time(T0, T1, NT)

        # integral of forces and moments
        # int_F = ∫ F(t) dt = F * t
        # F = m * a * r
        int_F_x = self.integrate(np.ones(t.size) * self.mTOT * ag[0], dt)
        int_F_y = self.integrate(np.ones(t.size) * self.mTOT * ag[1], dt)
        mom = 0.
        for p in self.parts:
            # x direction
            if p.phi < pi:
                mom +=  -p.M * ag[0] * p.r
            elif p.phi > pi:
                mom +=  p.M * ag[0] * p.r
            # y direction
            if p.phi < pi / 2. or p.phi > 3 / 4 * pi:
                mom += p.M * ag[1] * p.r
            elif p.phi > pi / 2. and p.phi < 3 / 4 * pi:
                mom += -p.M * ag[1] * p.r
        int_M_z = self.integrate(np.ones(t.size) * mom , dt)

        # velocities of COG
        # v = v0 + ∫ F(t) / m dt = v0 + F * t / m    <-- F = m * (dv / dt) = m * a
        vcx = v0[0] + int_F_x / self.mTOT
        vcy = v0[1] + int_F_y / self.mTOT
        vcz = np.zeros(vcx.size)

        # position of COG
        # r = r0 + ∫v dt = r0 + v * t
        rcx = r0[0] + self.integrate(vcx, dt)
        rcy = r0[1] + self.integrate(vcy, dt)
        rcz = np.zeros(rcx.size)

        # angular velocity = cross product of linear velocity and radius
        # w = L / I (angular velocity = angular momentun / rotational inertia)
        # w = r x v / |r|^2
        # L = r * m * v
        rcv = np.cross(np.vstack([rcx, rcy, rcz]).T, np.vstack([vcx, vcy, vcz]).T)
        w = w0 + (self.mTOT * rcv[:, 2] + int_M_z) / self.JTOT

        # angular position
        gamma = gamma0 + self.itegrate(w, dt)

        self.save_results(t, vcx, vcy, rcx, rcy, w, gamma)

        return (t, np.vstack([vcx, vcy]), np.vstack([rcx, rcy]), w, gamma)

    def plot(self, idt=0):
        idt = ((idt + 1) % len(self.t)) - 1
        dx = np.array([p.r * np.cos(p.phi + self.g[idt]) for p in self.parts])
        dy = np.array([p.r * np.sin(p.phi + self.g[idt]) for p in self.parts])
        m = np.array([p.M for p in self.parts]) / self.mTOT * 1000.

        bounds = np.max(np.maximum(np.abs(dx) + np.abs(self.rx[idt]), np.abs(dy) + np.abs(self.ry[idt]))) * 1.5

        if self.fig is None:
            self.fig, self.ax = plt.subplots(1, 1)
            self.bounds = (-bounds, bounds)
        else:
            self.ax.clear()

        self.ax.plot(self.bounds, [0., 0.], color='black', lw=0.5)
        self.ax.plot([0., 0.], self.bounds, color='black', lw=0.5)

        objs = []
        # color = '#d4rb3b'
        color = '#249abe'
        for i in range(len(self.parts)):
            objs.append(self.ax.plot([self.rx[idt], self.rx[idt] + dx[i]], [self.ry[idt], self.ry[idt] + dy[i]], zorder=90, color=color)[0])
            objs.append(self.ax.scatter(self.rx[idt] + dx[i], self.ry[idt] + dy[i], s=m[i], marker='o', zorder=100, color=color))

        self.ax.set(xlim=self.bounds, ylim=self.bounds)
        self.ax.set_title(f't = {self.t[idt]:8.4f}, α_max = {self.max_angle:6.2f}°')

        return objs

    def animate_results(self):
        self.plot()

        def update_plot(idt):
            return self.plot(idt)

        ani = animation.FuncAnimation(self.fig, update_plot, frames=len(self.t), interval=20) #, blit=True, save_count=50)

        plt.show()


def PTest():
    pTest = Machine()

    # pTest.add_part('RTS',  43.7E-3, -3.0729, 9065.9E-3, 257954230.0E-9)
    # pTest.add_part('KUW',  96.0E-3, -2.7918, 1337.2E-3,   1180893.9E-9, rpm=6000.)
    # pTest.add_part('AWL', 225.0E-3,  0.1373,  643.8E-3,    601657.3E-9)
    # pTest.add_part('SCH', 225.0E-3,  0.1373, 1668.0E-3,  25634000.0E-9, rpm=6000.)

    # pTest.add_part('RTS', -43.5969, -2.99951, 9065.9E-6, 257954230.0E-6)
    # pTest.add_part('KUW', -90.1866, -32.8995, 1337.2E-6,   1180893.9E-6, rpm=6000.)
    # pTest.add_part('AWL', 222.8830,  30.7955,  643.8E-6,    601657.3E-6)
    # pTest.add_part('SCH', 222.8830,  30.7955, 1668.0E-6,  25634000.0E-6, rpm=6000.)

    # pTest.add_part('RTS', radius= 43.7E-3, angle=-3.0729, mass=9065.9E-3, moment_of_inertia=257954230.0E-9)
    # pTest.add_part('KUW', radius= 96.0E-3, angle=-2.7918, mass=1337.2E-3, moment_of_inertia=  1180893.9E-9, rpm=-6000.)
    # pTest.add_part('AWL', radius=225.0E-3, angle= 0.1373, mass= 643.8E-3, moment_of_inertia=   601657.3E-9)
    # pTest.add_part('SCH', radius=225.0E-3, angle= 0.1373, mass=1668.0E-3, moment_of_inertia= 25634000.0E-9, rpm=-6000.)

    pTest.add_part('RTS', radius= 43.7, angle=-3.0729, mass=9065.9E-6, moment_of_inertia=257954230.0E-6)
    pTest.add_part('KUW', radius= 96.0, angle=-2.7918, mass=1337.2E-6, moment_of_inertia=  1180893.9E-6, rpm=-6000.)
    pTest.add_part('AWL', radius=225.0, angle= 0.1373, mass= 643.8E-6, moment_of_inertia=   601657.3E-6)
    pTest.add_part('SCH', radius=225.0, angle= 0.1373, mass=1668.0E-6, moment_of_inertia= 25634000.0E-6, rpm=-6000.)

    pTest.parts2cog(0., 0.)

    pTest.print('info')

    # phase I
    # t1, vc1, rc1, w1, gamma1 = pTest.force_and_moment_impulse(Fx=-100., Fy=1000., Mz=1000000., T0=0., T1=0.02, NT=50)
    t1, vc1, rc1, w1, gamma1 = pTest.movement(T0=0., T1=0.02, NT=50, F=(-100., 1000., 1000000.))

    # phase II
    # t2, vc2, rc2, w2, gamma2 = pTest.free_movement(T0=0.02, T1=0.04, NT=50, v0=vc1[:,-1], r0=rc1[:,-1], w0=w1[-1], gamma0=gamma1[-1])
    t2, vc2, rc2, w2, gamma2 = pTest.movement(T0=0.02, T1=0.04, NT=50)

    # phase III
    # t3, vc3, rc3, w3, gamma3 = pTest.braking(T0=0.04, T1=0.09, NT=125, v0=vc2[:,-1], r0=rc2[:,-1], w0=w2[-1], gamma0=gamma2[-1])
    t3, vc3, rc3, w3, gamma3 = pTest.movement(T0=0.04, T1=0.09, NT=125, brake=True)

    pTest.print('results')

    pTest.animate_results()


if __name__ == '__main__':
    PTest()

