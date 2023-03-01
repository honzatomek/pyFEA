import numpy as np

mass = {
    "nodes": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
    "elements": ["MASS", [[1], [2], [3], [4]]]
}

rod = {
    "nodes": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
    "elements": ["ROD", [[1, 2], [1, 3], [1, 4], [2, 3], [3, 4]]]
}

bar = {
    "nodes": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
    "elements": ["BAR", [[1, 2], [1, 3], [1, 4], [2, 3], [3, 4]]]
}

beam = {
    "nodes": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
    "elements": ["BEAM", [[1, 2], [1, 3], [1, 4], [2, 3], [3, 4]]]
}

tria3 = {
    "nodes": [[0.0, 0.0, 0.0],
              [1.0, 0.0, 0.0],
              [1.0, 1.0, 0.0],
              [0.0, 1.0, 0.0]],
    "elements": ["TRIA3", [[1, 2, 3],
                           [1, 3, 4]]]
}

tria6 = {
    "nodes": [[0.00, 0.00, 0.00],
              [1.00, 0.00, 0.00],
              [1.00, 1.00, 0.00],
              [0.50, 0.25, 0.00],
              [1.25, 0.50, 0.00],
              [0.25, 0.75, 0.00],
              [2.00, 1.00, 0.00],
              [1.50, 1.25, 0.00],
              [1.75, 0.25, 0.00]],
    "elements": ["TRIA6", [[1, 2, 3, 4, 5, 6],
                           [2, 7, 3, 9, 8, 5]]]
}

quad4 = {
    "nodes": [[0.0, 0.0, 0.0],
              [1.0, 0.0, 0.0],
              [2.0, 0.0, 0.0],
              [2.0, 1.0, 0.0],
              [1.0, 1.0, 0.0],
              [0.0, 1.0, 0.0]],
    "elements": ["QUAD4", [[1, 2, 5, 6],
                           [2, 3, 4, 5]]]
}

d = 0.1
quad8 = {
    "nodes": [[  0.0,   0.0, 0.0],
              [  1.0,   0.0, 0.0],
              [  1.0,   1.0, 0.0],
              [  0.0,   1.0, 0.0],
              [  0.5,     d, 0.0],
              [1 - d,   0.5, 0.0],
              [  0.5, 1 - d, 0.0],
              [    d,   0.5, 0.0],
              [  2.0,   0.0, 0.0],
              [  2.0,   1.0, 0.0],
              [  1.5,    -d, 0.0],
              [2 + d,   0.5, 0.0],
              [  1.5, 1 + d, 0.0]],
    "elements": ["QUAD8", [[1, 2,  3, 4,  5,  6,  7, 8],
                           [2, 9, 10, 3, 11, 12, 13, 6]]]
}

tet4 = {
    "nodes": [[0.0, 0.0, 0.0],
              [1.0, 0.0, 0.0],
              [1.0, 1.0, 0.0],
              [0.0, 1.0, 0.0],
              [0.5, 0.5, 0.5]],
    "elements": ["TET4", [[1, 2, 3, 5],
                          [1, 3, 4, 5]]]
}

tet10 = {
    "nodes": [[0.00, 0.00, 0.00],
              [1.00, 0.00, 0.00],
              [1.00, 1.00, 0.00],
              [0.50, 0.50, 0.50],
              # midnodes
              [0.50, 0.00, 0.10],
              [1.00, 0.50, 0.10],
              [0.50, 0.50, 0.10],
              [0.25, 0.30, 0.25],
              [0.80, 0.25, 0.25],
              [0.70, 0.70, 0.30],
    ],
    "elements": ["TET10", [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]]
}

hex8 = {
    "nodes": [[0.0, 0.0, 0.0],
              [1.0, 0.0, 0.0],
              [1.0, 1.0, 0.0],
              [0.0, 1.0, 0.0],
              [0.0, 0.0, 1.0],
              [1.0, 0.0, 1.0],
              [1.0, 1.0, 1.0],
              [0.0, 1.0, 1.0]],
    "elements": ["HEX8", [[0, 1, 2, 3, 4, 5, 6, 7]]],
}

hex20 = {
    "nodes": [[0.0, 0.0, 0.0],
              [1.0, 0.0, 0.0],
              [1.0, 1.0, 0.0],
              [0.0, 1.0, 0.0],
              [0.0, 0.0, 1.0],
              [1.0, 0.0, 1.0],
              [1.0, 1.0, 1.0],
              [0.0, 1.0, 1.0],
              #
              [0.5, 0.0, 0.0],
              [1.0, 0.5, 0.0],
              [0.5, 1.0, 0.0],
              [0.0, 0.5, 0.0],
              #
              [0.0, 0.0, 0.5],
              [1.0, 0.0, 0.5],
              [1.0, 1.0, 0.5],
              [0.0, 1.0, 0.5],
              #
              [0.5, 0.0, 1.0],
              [1.0, 0.5, 1.0],
              [0.5, 1.0, 1.0],
              [0.0, 0.5, 1.0]],
    "elements": ["HEX20", [list(range(1, 21))]]
}

wedge6 = {
    "nodes": [[0.0, 0.0, 0.0],
              [1.0, 0.0, 0.0],
              [1.0, 1.0, 0.0],
              [0.0, 0.0, 1.0],
              [1.0, 0.0, 1.0],
              [1.0, 1.0, 1.0]],
    "elements": ["WEDGE6", [[1, 2, 3, 4, 5, 6]]]
}

pyra5 = {
    "nodes": [[0.0, 0.0, 0.0],
              [1.0, 0.0, 0.0],
              [1.0, 1.0, 0.0],
              [0.0, 1.0, 0.0],
              [0.5, 0.5, 1.0]],
    "elements": ["PYRA5", [[1, 2, 3, 4, 5]]],
}

models = [mass,
          rod, bar, beam,
          tria3, tria6, quad4, quad8,
          tet4, tet10, hex8, hex20, wedge6, pyra5]


stahl = {"name": "stahl",
         "fetype": "ISO",
         "E": 210000.,
         "nu": 0.3,
         "rho": 7.85e-9,
         "alpha": 1.2e-5,
         "G": None,
         "result": "pass"}

stahl_shear = {"name": "stahl_shear",
               "fetype": "ISO",
               "E": 210000.,
               "nu": 0.3,
               "rho": 7.85e-9,
               "alpha": 1.2e-5,
               "G": 80769.,
               "result": "pass"}

stahl_temp = {"name": "stahl_temp",
              "fetype": "ISO",
              "E": [[  0., 210000.],
                    [ 20., 210000.],
                    [100., 208000.],
                    [200., 203000.],
                    [300., 197000.],
                    [400., 192000.],
                    [500., 185000.],
                    [600., 178000.],
                    [700., 170000.],
                    [800., 161000.],
                    [900., 151000.]],
              "nu": [[  0., .300],
                     [700., .315],
                     [800., .317],
                     [900., .319]],
              "rho": [[  0.,  7.850e-09],
                      [100.,  7.816e-09],
                      [200.,  7.783e-09],
                      [300.,  7.749e-09],
                      [400.,  7.715e-09],
                      [500.,  7.682e-09],
                      [600.,  7.648e-09],
                      [700.,  7.615e-09],
                      [800.,  7.581e-09],
                      [900.,  7.547e-09]],
              "alpha": [[  0.,  1.200e-05],
                        [100.,  9.984e-04],
                        [200.,  2.318e-03],
                        [300.,  3.718e-03],
                        [400.,  5.198e-03],
                        [500.,  6.758e-03],
                        [600.,  8.398e-03],
                        [700.,  1.012e-02],
                        [800.,  1.100e-02],
                        [900.,  1.180e-02]],
              "G": None,
              "result": "pass"}

stahl_shear_temp = {"name": "stahl_shear_temp",
                    "fetype": "ISO",
                    "E": [[  0., 210000.],
                          [ 20., 210000.],
                          [100., 208000.],
                          [200., 203000.],
                          [300., 197000.],
                          [400., 192000.],
                          [500., 185000.],
                          [600., 178000.],
                          [700., 170000.],
                          [800., 161000.],
                          [900., 151000.]],
                    "nu": [[  0., .300],
                           [700., .315],
                           [800., .317],
                           [900., .319]],
                    "rho": [[  0.,  7.850e-09],
                            [100.,  7.816e-09],
                            [200.,  7.783e-09],
                            [300.,  7.749e-09],
                            [400.,  7.715e-09],
                            [500.,  7.682e-09],
                            [600.,  7.648e-09],
                            [700.,  7.615e-09],
                            [800.,  7.581e-09],
                            [900.,  7.547e-09]],
                    "alpha": [[  0.,  1.200e-05],
                              [100.,  9.984e-04],
                              [200.,  2.318e-03],
                              [300.,  3.718e-03],
                              [400.,  5.198e-03],
                              [500.,  6.758e-03],
                              [600.,  8.398e-03],
                              [700.,  1.012e-02],
                              [800.,  1.100e-02],
                              [900.,  1.180e-02]],
                    "G": [[  0.,  8.077e+04],
                          [100.,  7.987e+04],
                          [200.,  7.782e+04],
                          [300.,  7.540e+04],
                          [400.,  7.336e+04],
                          [500.,  7.057e+04],
                          [600.,  6.779e+04],
                          [700.,  6.464e+04],
                          [800.,  6.112e+04],
                          [900.,  5.724e+04]],
                    "result": "pass"}

stahl_fail01 = {"name": "1stahl_fail01",
                "fetype": "ISO",
                "E": 210000.,
                "nu": 0.3,
                "rho": 7.85e-9,
                "alpha": 1.2e-5,
                "G": None,
                "result": ValueError}

stahl_fail02 = {"name": "_stahl_fail02",
                "fetype": "ISO",
                "E": 210000.,
                "nu": 0.3,
                "rho": 7.85e-9,
                "alpha": 1.2e-5,
                "G": None,
                "result": ValueError}

stahl_fail03 = {"name": 1001,
                "fetype": "ISO",
                "E": 210000.,
                "nu": 0.3,
                "rho": 7.85e-9,
                "alpha": 1.2e-5,
                "G": None,
                "result": TypeError}

stahl_fail04 = {"name": "stahl_fail04",
                "fetype": "ISO",
                "E": 0.,
                "nu": 0.3,
                "rho": 7.85e-9,
                "alpha": 1.2e-5,
                "G": None,
                "result": ValueError}

stahl_fail05 = {"name": "stahl_fail05",
                "fetype": "ISO",
                "E": -1.,
                "nu": 0.3,
                "rho": 7.85e-9,
                "alpha": 1.2e-5,
                "G": None,
                "result": ValueError}

stahl_fail06 = {"name": "stahl_fail06",
                "fetype": "ISO",
                "E": "a",
                "nu": 0.3,
                "rho": 7.85e-9,
                "alpha": 1.2e-5,
                "G": None,
                "result": TypeError}

stahl_fail07 = {"name": "stahl_fail07",
                "fetype": "ISO",
                "E": 210000.,
                "nu": 0.,
                "rho": 7.85e-9,
                "alpha": 1.2e-5,
                "G": None,
                "result": ValueError}

stahl_fail08 = {"name": "stahl_fail08",
                "fetype": "ISO",
                "E": 210000.,
                "nu": -0.3,
                "rho": 7.85e-9,
                "alpha": 1.2e-5,
                "G": None,
                "result": ValueError}

stahl_fail09 = {"name": "stahl_fail09",
                "fetype": "ISO",
                "E": 210000.,
                "nu": "a",
                "rho": 7.85e-9,
                "alpha": 1.2e-5,
                "G": None,
                "result": TypeError}

stahl_fail10 = {"name": "stahl_fail10",
                "fetype": "ISO",
                "E": 210000.,
                "nu": 0.3,
                "rho": 0.,
                "alpha": 1.2e-5,
                "G": None,
                "result": ValueError}

stahl_fail11 = {"name": "stahl_fail11",
                "fetype": "ISO",
                "E": 210000.,
                "nu": 0.3,
                "rho": -7.85e-9,
                "alpha": 1.2e-5,
                "G": None,
                "result": ValueError}

stahl_fail12 = {"name": "stahl_fail12",
                "fetype": "ISO",
                "E": 210000.,
                "nu": 0.3,
                "rho": "a",
                "alpha": 1.2e-5,
                "G": None,
                "result": TypeError}

stahl_fail13 = {"name": "stahl_fail13",
                "fetype": "ISO",
                "E": 210000.,
                "nu": 0.3,
                "rho": 7.85e-9,
                "alpha": 0.,
                "G": None,
                "result": ValueError}

stahl_fail14 = {"name": "stahl_fail14",
                "fetype": "ISO",
                "E": 210000.,
                "nu": 0.3,
                "rho": -7.85e-9,
                "alpha": -1.2e-5,
                "G": None,
                "result": ValueError}

stahl_fail15 = {"name": "stahl_fail15",
                "fetype": "ISO",
                "E": 210000.,
                "nu": 0.3,
                "rho": 7.85e-9,
                "alpha": "a",
                "G": None,
                "result": TypeError}

stahl_fail16 = {"name": "stahl_fail16",
                "fetype": "ISO",
                "E": [[  0., 210000.],
                      [ 20., 210000.],
                      [100., 208000.],
                      [200.,      0.],
                      [300., 197000.],
                      [400., 192000.],
                      [500., 185000.],
                      [600., 178000.],
                      [700., 170000.],
                      [800., 161000.],
                      [900., 151000.]],
                "nu": 0.3,
                "rho": 7.85e-9,
                "alpha": 1.2e-5,
                "G": None,
                "result": ValueError}

stahl_fail17 = {"name": "stahl_fail17",
                "fetype": "ISO",
                "E": [[  0., 210000.],
                      [ 20., 210000.],
                      [100., 208000.],
                      [200.,-203000.],
                      [300., 197000.],
                      [400., 192000.],
                      [500., 185000.],
                      [600., 178000.],
                      [700., 170000.],
                      [800., 161000.],
                      [900., 151000.]],
                "nu": 0.3,
                "rho": 7.85e-9,
                "alpha": 1.2e-5,
                "G": None,
                "result": ValueError}

stahl_fail18 = {"name": "stahl_fail18",
                "fetype": "ISO",
                "E": [[  0, 210000.],
                      [ 20, 210000.],
                      [100, 208000.],
                      [200, 203000.],
                      [300, 197000.],
                      [400, 192000.],
                      [500, 185000.],
                      [600, 178000.],
                      [700, 170000.],
                      [800, 161000.],
                      [900, 151000.]],
                "nu": 0.3,
                "rho": 7.85e-9,
                "alpha": 1.2e-5,
                "G": None,
                "result": TypeError}

stahl_fail19 = {"name": "stahl_fail19",
                "fetype": "ISO",
                "E": [[  0., 210000],
                      [ 20., 210000],
                      [100., 208000],
                      [200., 203000],
                      [300., 197000],
                      [400., 192000],
                      [500., 185000],
                      [600., 178000],
                      [700., 170000],
                      [800., 161000],
                      [900., 151000]],
                "nu": 0.3,
                "rho": 7.85e-9,
                "alpha": 1.2e-5,
                "G": None,
                "result": TypeError}

def alpha(t):
    if t <= 20.:
        return 1.2e-5
    if t <= 750.:
        return 1.2e-5 * t + 0.4e-8 * (t ** 2) - 2.416e-4
    elif t <= 860.:
        return 1.1e-2
    else:
        return 2.0e-5 * t - 6.2e-3


def density(t):
    rho = 7.85e-9
    return (1.0 - (0.03 / 700. * t)) * rho


materials = [stahl, stahl_shear,
             stahl_temp, stahl_shear_temp,
             stahl_fail01, stahl_fail02, stahl_fail03,
             stahl_fail04, stahl_fail05, stahl_fail06,
             stahl_fail07, stahl_fail08, stahl_fail09,
             stahl_fail10, stahl_fail11, stahl_fail12,
             stahl_fail13, stahl_fail14, stahl_fail15,
             stahl_fail16, stahl_fail17,
             stahl_fail18, stahl_fail19]


mass3_1 = {"name":  "mass3_1",
           "fetype":    "MASS",
           "mxx":         0.1,
           "myy":         0.1,
           "mzz":         0.1}

mass3_2 = ["mass3_2", "MASS", 0.1, 0.1, 0.1]

mass3_3 = ("mass3_3", "MASS", 0.1, 0.1, 0.1)

mass6_1 = {"name":  "mass6_1",
           "fetype":    "MASS",
           "mxx":         0.1,
           "myy":         0.1,
           "mzz":         0.1,
           "Ixx":       0.001,
           "Iyy":       0.001,
           "Izz":       0.001,
           "Ixy":       0.001,
           "Iyz":       0.001,
           "Ixz":       0.001}

mass6_2 = ["mass6_3", "MASS", 0.1, 0.1, 0.1, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001]

mass6_3 = ("mass6_4", "MASS", 0.1, 0.1, 0.1, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001)

rod_1 = {"name":  "rod_1",
         "fetype":    "ROD",
         "A":      1000.0,
         "nsm":     0.001}

rod_2 = ["rod_2", "ROD", 5380.0, 0.001]

rod_3 = ("rod_3", "ROD", 5380.0, 0.001)

beam_1 = {"name":  "beam_1",
          "fetype":   "BEAM",
          "A":       5380.0,
          "Iyy": 83600000.0,
          "Izz":  6040000.0,
          "Iw": 126000000.0,
          "nsm":      0.001}

beam_2 = ["beam_2", "BEAM", 5380.0, 83600000.0, 6040000.0, 126000000.0, 0.001]

beam_3 = ("beam_3", "BEAM", 5380.0, 83600000.0, 6040000.0, 126000000.0, 0.001)

shell_1 = {"name":  "shell_1",
           "fetype":   "SHELL",
           "t":           10.,
           "nsm":     0.00015}

shell_2 = ["shell_2", "SHELL", 10., 0.00015]

shell_3 = ("shell_3", "SHELL", 10., 0.00015)

tria3_1 = {"name":  "tria3_1",
           "fetype":   "TRIA3",
           "t1":          10.,
           "t2":          15.,
           "t3":          20.,
           "nsm":     0.00015}

tria3_2 = ["tria3_2", "TRIA3", 10., 15., 20., 0.00015]
tria3_3 = ("tria3_3", "TRIA3", 10., 15., 20., 0.00015)

tria6_1 = {"name":  "tria6_1",
           "fetype":   "TRIA6",
           "t1":          10.,
           "t2":          15.,
           "t3":          20.,
           "t4":          25.,
           "t5":          30.,
           "t6":          35.,
           "nsm":     0.00015}

tria6_2 = ["tria6_2", "TRIA6", 10., 15., 20., 25., 30., 35., 0.00015]

tria6_3 = ("tria6_3", "TRIA6", 10., 15., 20., 25., 30., 35., 0.00015)

quad4_1 = {"name":  "quad4_1",
           "fetype":   "QUAD4",
           "t1":          10.,
           "t2":          15.,
           "t3":          20.,
           "t4":          25.,
           "nsm":     0.00015}

quad4_2 = ["quad4_2", "QUAD4", 10., 15., 20., 25., 0.00015]

quad4_3 = ("quad4_3", "QUAD4", 10., 15., 20., 25., 0.00015)

quad8_1 = {"name":  "quad8_1",
           "fetype":   "QUAD8",
           "t1":          10.,
           "t2":          15.,
           "t3":          20.,
           "t4":          25.,
           "t5":          30.,
           "t6":          35.,
           "t7":          40.,
           "t8":          45.,
           "nsm":     0.00015}

quad8_2 = ["quad8_2", "QUAD8", 10., 15., 20., 25., 30., 35., 40., 45., 0.00015]

quad8_3 = ("quad8_3", "QUAD8", 10., 15., 20., 25., 30., 35., 40., 45., 0.00015)

solid_1 = {"name":  "solid_1",
           "fetype":   "SOLID",
           "nsm":   0.0000015}

solid_2 = ["solid_2", "SOLID", 0.00015]

solid_3 = ("solid_3", "SOLID", 0.00015)

properties = [mass3_1, mass3_2, mass3_3,
              mass6_1, mass6_2, mass6_3,
              rod_1, rod_2, rod_3,
              beam_1, beam_2, beam_3,
              shell_1, shell_2, shell_3,
              tria3_1, tria3_2, tria3_3,
              tria6_1, tria6_2, tria6_3,
              quad4_1, quad4_2, quad4_3,
              quad8_1, quad8_2, quad8_3,
              solid_1, solid_2, solid_3]


load1 = {"fetype": "conload",
         "nid": 1001,
         "lpat": 101,
         "Fx":  100.,
         "Fy":  100.,
         "Fz":  100.,
         "Mx":  100.,
         "My":  100.,
         "Mz":  100.}

load2 = ["conload", 1002, 101, 100., 100., 100., 100., 100., 100.]

load3 = ("conload", 1002, 101, 100., 100., 100., 100., 100., 100.)

nodal_loads = [load1, load2, load3]

load_fail1 = {"fetype": "conload",
              "nid": -1,
              "lpat": 101,
              "Fx":  100.,
              "Fy":  100.,
              "Fz":  100.,
              "Mx":  100.,
              "My":  100.,
              "Mz":  100.,
              "result": ValueError}

load_fail2 = {"fetype": "conload",
              "nid": "a",
              "lpat": 101,
              "Fx":  100.,
              "Fy":  100.,
              "Fz":  100.,
              "Mx":  100.,
              "My":  100.,
              "Mz":  100.,
              "result": TypeError}

load_fail3 = {"fetype": "conload",
              "nid": 1001,
              "lpat":   0,
              "Fx":  100.,
              "Fy":  100.,
              "Fz":  100.,
              "Mx":  100.,
              "My":  100.,
              "Mz":  100.,
              "result": ValueError}

load_fail4 = {"fetype": "conload",
              "nid": 1001,
              "lpat": "a",
              "Fx":  100.,
              "Fy":  100.,
              "Fz":  100.,
              "Mx":  100.,
              "My":  100.,
              "Mz":  100.,
              "result": TypeError}

load_fail5 = {"fetype": "conload",
              "nid": 1001,
              "lpat": 101,
              "Fx":  None,
              "Fy":  None,
              "Fz":  None,
              "Mx":  None,
              "My":  None,
              "Mz":  None,
              "result": ValueError}

load_fail6 = {"fetype": "conload",
              "nid": 1001,
              "lpat": 101,
              "Fx":    0.,
              "Fy":    0.,
              "Fz":    0.,
              "Mx":    0.,
              "My":    0.,
              "Mz":    0.,
              "result": ValueError}

load_fail7 = {"fetype": "conload",
              "nid": 1001,
              "lpat": 101,
              "Fx":    0.,
              "Fy":   "a",
              "Fz":    0.,
              "Mx":    0.,
              "My":    0.,
              "Mz":    0.,
              "result": TypeError}

nodal_loads_fail = [load_fail1, load_fail2, load_fail3, load_fail4, load_fail5,
                    load_fail6, load_fail7]


nodal_loads_list = [
    ["conload", 1001, 100,    0., 100., 1000.,    0.,   0.,    0.],
    ["conload", 1001, 100, 1000.,   0.,    0.,    0.,   0.,    0.],
    ["conload", 1002, 100,    0., 100.,  100.,    0.,   0.,    0.],
    ["conload", 1003, 100,    0., 100.,  100.,    0.,   0.,    0.],
    ["conload", 1004, 100,    0., 100.,  100.,    0.,   0.,    0.],
    ["conload", 1005, 100,    0., 100.,  100.,    0.,   0.,    0.],
    ["conload", 1001, 200, -100.,   0.,    0.,    0.,   0.,    0.],
    ["conload", 1002, 200,    0., 100., 1000.,    0.,   0.,    0.],
    ["conload", 1003, 200,    0., 100., 1000.,    0.,   0.,    0.],
    ["conload", 1004, 200,    0., 100., 1000.,    0.,   0.,    0.],
    ["conload", 1005, 200,    0., 100., 1000.,    0.,   0.,    0.],
    ["conload", 1001, 300,    0.,   0.,    0., -100.,   0.,    0.],
    ["conload", 1002, 300,    0.,   0.,    0.,    0., 100., 1000.],
    ["conload", 1003, 300,    0.,   0.,    0.,    0., 100., 1000.],
    ["conload", 1004, 300,    0.,   0.,    0.,    0., 100., 1000.],
    ["conload", 1005, 300,    0.,   0.,    0.,    0., 100., 1000.],
]

nodal_loads_dict = {
    100: {
        1001: {
            "fetype": "conload",
            "nid":   1001,
            "lpat":   100,
            "Fx":   1000.,
            "Fy":    100.,
            "Fz":   1000.,
            "Mx":      0.,
            "My":      0.,
            "Mz":      0.},
        1002: {
            "fetype": "conload",
            "nid":   1002,
            "lpat":   100,
            "Fx":      0.,
            "Fy":    100.,
            "Fz":    100.,
            "Mx":      0.,
            "My":      0.,
            "Mz":      0.},
        1003: {
            "fetype": "conload",
            "nid":   1003,
            "lpat":   100,
            "Fx":      0.,
            "Fy":    100.,
            "Fz":    100.,
            "Mx":      0.,
            "My":      0.,
            "Mz":      0.},
        1004: {
            "fetype": "conload",
            "nid":   1004,
            "lpat":   100,
            "Fx":      0.,
            "Fy":    100.,
            "Fz":    100.,
            "Mx":      0.,
            "My":      0.,
            "Mz":      0.},
        1005: {
            "fetype": "conload",
            "nid":   1005,
            "lpat":   100,
            "Fx":      0.,
            "Fy":    100.,
            "Fz":    100.,
            "Mx":      0.,
            "My":      0.,
            "Mz":      0.}
        },
    200: {
        1001: {
            "fetype": "conload",
            "nid":   1001,
            "lpat":   200,
            "Fx":   -100.,
            "Fy":      0.,
            "Fz":      0.,
            "Mx":      0.,
            "My":      0.,
            "Mz":      0.},
        1002: {
            "fetype": "conload",
            "nid":   1002,
            "lpat":   200,
            "Fx":      0.,
            "Fy":    100.,
            "Fz":    100.,
            "Mx":      0.,
            "My":      0.,
            "Mz":      0.},
        1003: {
            "fetype": "conload",
            "nid":   1003,
            "lpat":   200,
            "Fx":      0.,
            "Fy":    100.,
            "Fz":    100.,
            "Mx":      0.,
            "My":      0.,
            "Mz":      0.},
        1004: {
            "fetype": "conload",
            "nid":   1004,
            "lpat":   200,
            "Fx":      0.,
            "Fy":    100.,
            "Fz":    100.,
            "Mx":      0.,
            "My":      0.,
            "Mz":      0.},
        1005: {
            "fetype": "conload",
            "nid":   1005,
            "lpat":   200,
            "Fx":      0.,
            "Fy":    100.,
            "Fz":    100.,
            "Mx":      0.,
            "My":      0.,
            "Mz":      0.}
        },
    300: {
        1001: {
            "fetype": "conload",
            "nid":   1001,
            "lpat":   300,
            "Fx":      0.,
            "Fy":      0.,
            "Fz":      0.,
            "Mx":   -100.,
            "My":      0.,
            "Mz":      0.},
        1002: {
            "fetype": "conload",
            "nid":   1002,
            "lpat":   300,
            "Fx":      0.,
            "Fy":      0.,
            "Fz":      0.,
            "Mx":      0.,
            "My":    100.,
            "Mz":   1000.},
        1003: {
            "fetype": "conload",
            "nid":   1003,
            "lpat":   300,
            "Fx":      0.,
            "Fy":      0.,
            "Fz":      0.,
            "Mx":      0.,
            "My":    100.,
            "Mz":   1000.},
        1004: {
            "fetype": "conload",
            "nid":   1004,
            "lpat":   300,
            "Fx":      0.,
            "Fy":      0.,
            "Fz":      0.,
            "Mx":      0.,
            "My":    100.,
            "Mz":   1000.},
        1005: {
            "fetype": "conload",
            "nid":   1005,
            "lpat":   300,
            "Fx":      0.,
            "Fy":      0.,
            "Fz":      0.,
            "Mx":      0.,
            "My":    100.,
            "Mz":   1000.}
    }
}


nodes = {100: np.array([0., 0., 0.], dtype=float),
         101: np.array([1., 0., 0.], dtype=float),
         102: np.array([1., 1., 0.], dtype=float),
         103: np.array([0., 1., 0.], dtype=float),
         200: np.array([0., 0., 1.], dtype=float),
         201: np.array([1., 0., 1.], dtype=float),
         202: np.array([1., 1., 1.], dtype=float),
         203: np.array([0., 1., 1.], dtype=float)}

scalar = {100: 0.,
          101: 0.,
          102: 0.,
          103: 0.,
          200: 1.,
          201: 1.,
          202: 1.,
          203: 1.}

vector = {100: np.array([0., 0., 0.], dtype=float),
          101: np.array([0., 0., 0.], dtype=float),
          102: np.array([0., 1., 0.], dtype=float),
          103: np.array([0., 1., 0.], dtype=float),
          200: np.array([1., 0., 1.], dtype=float),
          201: np.array([1., 0., 1.], dtype=float),
          202: np.array([1., 1., 1.], dtype=float),
          203: np.array([1., 1., 1.], dtype=float)}

tensor = {100: np.array([[0., 0., 0.],[0., 0., 0.],[0., 0., 0.]], dtype=float),
          101: np.array([[0., 0., 0.],[0., 0., 0.],[0., 0., 0.]], dtype=float),
          102: np.array([[0., 1., 0.],[0., 1., 0.],[0., 1., 0.]], dtype=float),
          103: np.array([[0., 1., 0.],[0., 1., 0.],[0., 1., 0.]], dtype=float),
          200: np.array([[1., 0., 1.],[1., 0., 1.],[1., 0., 1.]], dtype=float),
          201: np.array([[1., 0., 1.],[1., 0., 1.],[1., 0., 1.]], dtype=float),
          202: np.array([[1., 1., 1.],[1., 1., 1.],[1., 1., 1.]], dtype=float),
          203: np.array([[1., 1., 1.],[1., 1., 1.],[1., 1., 1.]], dtype=float)}

# vector = {100: np.array([0., 2.], dtype=float),
#           101: np.array([0., 2.], dtype=float),
#           102: np.array([0., 2.], dtype=float),
#           103: np.array([0., 2.], dtype=float),
#           200: np.array([1., 4.], dtype=float),
#           201: np.array([1., 4.], dtype=float),
#           202: np.array([1., 4.], dtype=float),
#           203: np.array([1., 4.], dtype=float)}

# tensor = {100: np.array([[0., 2.],[ 8., 32.]], dtype=float),
#           101: np.array([[0., 2.],[ 8., 32.]], dtype=float),
#           102: np.array([[0., 2.],[ 8., 32.]], dtype=float),
#           103: np.array([[0., 2.],[ 8., 32.]], dtype=float),
#           200: np.array([[1., 4.],[16., 64.]], dtype=float),
#           201: np.array([[1., 4.],[16., 64.]], dtype=float),
#           202: np.array([[1., 4.],[16., 64.]], dtype=float),
#           203: np.array([[1., 4.],[16., 64.]], dtype=float)}

new_nodes = {1: np.array([0.25, 0.25, 0.25], dtype=float),
             2: np.array([0.50, 0.50, 0.50], dtype=float),
             3: np.array([0.75, 0.75, 0.75], dtype=float)}

mapping = {"onodes":  nodes,
           "nnodes": new_nodes,
           "scalar":  scalar,
           "vector":  vector,
           "tensor":  tensor}






if __name__ == "__main__":
    for t in (0., 100., 200., 300., 400., 500., 600., 700., 800., 900.):
        print(f"[{t:3.0f}, {alpha(t):10.3e}],") # thermal expansion
        # print(f"[{t:3.0f}, {density(t):10.3e}],") # density
    # import numpy as np
    # E = np.array(stahl_temp["E"], dtype=float)
    # nu = np.array(stahl_temp["nu"], dtype=float)
    # for t in (0., 100., 200., 300., 400., 500., 600., 700., 800., 900.):
    #     e = np.interp(t, E[:,0], E[:,1])
    #     n = np.interp(t, nu[:,0], nu[:,1])
    #     print(f"[{t:3.0f}, {e / (2. * (1. + n)):10.3e}],") # density




