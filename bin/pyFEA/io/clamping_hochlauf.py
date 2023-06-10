#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt


tUp = 0.055
tContact = 0.06
tDown = 0.05
tClamp = tUp + tContact + tDown

dt = min(tUp, tContact, tDown) / 10.

tFinal = 0.3

t = np.linspace(0., tFinal, int(tFinal / dt) + 1)

f = []
for i in range(t.shape[0]):
    if t[i] < tUp:
        f.append(np.sin((np.pi * t[i]) / (2 * tUp)) ** 2)
    elif t[i] <= (tUp + tContact):
        f.append(1.)
    elif t[i] <= tClamp:
        f.append(np.cos((np.pi * (t[i] - tUp - tContact)) / (2 * tDown)) ** 2)
    else:
        f.append(0.)

breakpoint()
f = np.array(f, dtype=float)

fig, ax = plt.subplots()

ax.plot(t, f)

plt.show()


