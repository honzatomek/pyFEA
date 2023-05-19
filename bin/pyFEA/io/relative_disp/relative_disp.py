#!/usr/bin/python3
"""author:  Jan Tomek
date:    19.04.2023
version: v1.0.0

description:
LS-DYNA displacements transformation to local coordinate system defined by three nodes.
The nodes are defining the Local Cartesian Coordinate system origin, point on one axis
and point in one plane respectively.

Example of input files:

Nodes input in LS-DYNA *.k format (only *NODE keyword necessary):
>>> *KEYWORD
>>> $CREATED BY MEDINA 8.3.1.4 Hotfix Release (64 Bit) program: dyna3d83/bifdyn	Ptf-Version 89 08/16/12
>>> $===============================================================================
>>> $
>>> *NODE
>>> $   NID|              X|              Y|              Z|     TC|     RC|
>>>        1  0.00000000E+00  0.00000000E+00  0.00000000E+00       0       0
>>>        2  0.10000000E+02  0.00000000E+00  0.00000000E+00       0       0
>>>        3  0.10000000E+02  0.10000000E+02  0.00000000E+00       0       0
>>>        4  0.00000000E+00  0.10000000E+02  0.00000000E+00       0       0
>>>        5  0.00000000E+00  0.00000000E+00  0.10000000E+02       0       0
>>>        6  0.10000000E+02  0.00000000E+00  0.10000000E+02       0       0
>>>        7  0.10000000E+02  0.10000000E+02  0.10000000E+02       0       0
>>>        8  0.00000000E+00  0.10000000E+02  0.10000000E+02       0       0
>>> $===============================================================================
>>> $
>>> *ELEMENT_SOLID
>>> $   EID|    PID|     N1|     N2|     N3|     N4|     N5|     N6|     N7|     N8|
>>>        1       1       1       2       3       4       5       6       7       8
>>> $===============================================================================
>>> *END

Nodes input in PERMAS *.dato format (only $COOR keyword is necessary):
>>> $ENTER COMPONENT NAME = KOMPO_1 DOFTYPE = DISP MATH
>>> $STRUCTURE
>>> $COOR
>>>          1  1.00000E+01  1.00000E+01  1.00000E+01
>>>          2  2.00000E+01  1.00000E+01  1.00000E+01
>>>          3  3.00000E+01  1.00000E+01  1.00000E+01
>>> $END STRUCTURE
>>> $EXIT COMPONENT
>>> $FIN


Displacements input in LS-DYNA *.k format (both *KEYWORD and $TIME_VALUE necessary):
>>> *KEYWORD
>>> $TIME_VALUE =  1.0000000e+00
>>> $STATE_NO = 1
>>> $Output for State 1 at time = 1.00000000
>>> *END
>>> $NODAL_DISPLACEMENT
>>>        1  0.00000000E+00  0.00000000E+00  0.00000000E+00
>>>        2  0.10000000E+00  0.00000000E+00  0.00000000E+00
>>>        3  0.10000000E+00 -1.00000000E+00  1.00000000E+00
>>>        4  0.00000000E+00 -1.00000000E+00  1.00000000E+00
>>>        5  0.00000000E+00 -1.00000000E+00 -1.00000000E+00
>>>        6  0.10000000E+00 -1.00000000E+00 -1.00000000E+00
>>>        7  0.10000000E+00 -2.00000000E+00  0.00000000E+00
>>>        8  0.00000000E+00 -2.00000000E+00  0.00000000E+00

Displacement input in PERMAS *.post format (frequency taken from results name):
>>> $ENTER COMPONENT  NAME = KOMPO_1  DOFTYPE = DISP MATH
>>> $RESULTS NAME = SST19_F100_RES
>>> $PARAMETER
>>>   ANALYSIS = 'MODAL FREQUENCY'
>>> $DATA X_XYDATA
>>> & TYPE                = DISP
>>> & ABSCISSAE           = TIME
>>> & NCOL                =        2
>>> & CURVE               = 'N50023979,U'
>>>          1  0.000000E+00  3.598476E-04
>>>          2  2.000000E-04  2.443228E-02
>>>          3  4.000000E-04  4.786231E-02
>>> $END RESULTS
>>> $EXIT COMPONENT
>>> $FIN

"""

# TODO:
# create an object for structure with possible import/export to multiple formats
# rather than separate objects or functions for reading and writing the data
# it should not matter if it is for LS-Dyna or PERMAS, the functionality should
# be the same, just different sorting

import os
import sys
import io
import gzip
import argparse
import time
import numpy as np

# --------------------------------------------------------------- LS-DYNA --- {{1
_HEADER = """*KEYWORD
$TIME_VALUE =  {0:16.7e}
$STATE_NO = {1:n}
$Output for State {1:n} at time = {0:16.7e}
*END
$NODAL_DISPLACEMENT
"""

_HEADER_CSV = """$column0=node_ids
$column1=x
$column2=y
$column3=z
$column4=r
"""


_LINE_FMT = lambda nid, coors: f'{int(nid):8n}{float(coors[0]):16.7e}{float(coors[1]):16.7e}{float(coors[2]):16.7e}\n'
_LINE_CSV_FMT = lambda nid, coors: f'{int(nid):8n} {float(coors[0]):15.7e} {float(coors[1]):15.7e} {float(coors[2]):15.7e} {np.linalg.norm(coors):15.7e}\n'


# ---------------------------------------------------------------- PERMAS --- {{1
_POST_HEADER = """$ENTER COMPONENT  NAME = KOMPO_1  DOFTYPE = DISP MATH
"""

_POST_HEADER_RESULTS = """$RESULTS NAME = {0:s}
$PARAMETER
  ANALYSIS = 'MODAL FREQUENCY'
"""

_POST_HEADER_CURVE = """$DATA X_XYDATA
& TYPE                = DISP
& ABSCISSAE           = {0:s}
& NCOL                =        2
& CURVE               = 'N{1:n},{2:s}'
"""

_POST_HEADER_DISP = """$DATA DISP TIME = {0:14.6E}
& NCOL                =        {1:n}
& DATTYPE             = REAL
"""

_POST_CURVE_LINE = lambda id, abscissae, disp: f" {int(id):9n} {float(abscissae):14.6E} {float(disp):14.6E}\n"
_POST_CURVE_DISP = lambda nid, disp: f" {int(nid):9n}" + "".join([f" {float(d):14.6E}" for d in disp]) + "\n"

_POST_FOOTER_RESULTS = """$END RESULTS
"""

_POST_FOOTER = """$EXIT COMPONENT
$FIN
"""


# ------------------------------------------------------------------ CSYS --- {{1
class CART:
    """Cartesian coordinate system"""

    @classmethod
    def test(cls):
        o = [10,  0,  0]
        z = [10, 10,  0]
        x = [10,  0, 10]

        cart = cls(o, z, x)

        nodes = np.array([[2, 0, 0],
                          [1, 0, 0],
                          [1, 1, 1]], dtype=float)
        tnodes = np.array([[0, 1, 0],
                           [0, 0, 0],
                           [1, 0, 1]], dtype=float)
        nnodes = cart.gcs2lcs(nodes)

        assert np.array_equal(cart.gcs2lcs(nodes), tnodes)
        assert np.array_equal(cart.lcs2gcs(tnodes), nodes)


    def __init__(self, node1: np.ndarray, node2: np.ndarray, node3: np.ndarray, form: str = "ZX"):
        self.o  = node1
        self.p1 = node2
        self.p2 = node3
        self.form = form

    @property
    def o(self) -> np.ndarray:
        """origin"""
        return self._o

    @o.setter
    def o(self, node: np.ndarray):
        """origin"""
        self._o = np.array(node, dtype=float)

    @property
    def p1(self) -> np.ndarray:
        """point on 1st axis"""
        return self._p1

    @p1.setter
    def p1(self, node: np.ndarray):
        """point on 1st axis"""
        self._p1 = np.array(node, dtype=float)

    @property
    def p2(self) -> np.ndarray:
        """point on plane of 1st and 2d axis"""
        return self._p2

    @p2.setter
    def p2(self, node: np.ndarray):
        """point on plane of 1st and 2d axis"""
        self._p2 = np.array(node, dtype=float)

    @property
    def form(self) -> str:
        """cartesian system formulation"""
        return self._form

    @form.setter
    def form(self, form: str):
        """cartesian system formulation"""
        if form.upper() not in ("XY", "YX", "XZ", "ZX", "YZ", "ZY"):
            raise ValueError(f'Unknown Cartesian Coordinate system formulation: {form:s}')
        else:
            self._form = form.upper()

    @property
    def T(self) -> np.ndarray:
        """create transformation matrix based on 3 points defining origin a vector and a plane"""
        o = self.o
        v1 = self.p1 - self.o
        v2 = self.p2 - self.o

        if self.form == "XY":
            x = v1
            y = v2
            z = np.cross(x, y)
            y = np.cross(z, x)
        elif self.form == "YX":
            y = v1
            x = v2
            z = np.cross(x, y)
            x = np.cross(y, z)
        elif self.form == "XZ":
            x = v1
            z = v2
            y = np.cross(z, x)
            z = np.cross(x, y)
        elif self.form == "ZX":
            z = v1
            x = v2
            y = np.cross(z, x)
            x = np.cross(y, z)
        elif self.form == "YZ":
            y = v1
            z = v2
            x = np.cross(y, z)
            z = np.cross(x, y)
        elif self.form == "ZY":
            z = v1
            y = v2
            x = np.cross(y, z)
            y = np.cross(z, x)
        else:
            raise ValueError(f'Unknown Cartesian Coordinate system formulation: {self.form:s}')

        x /= np.linalg.norm(x)
        y /= np.linalg.norm(y)
        z /= np.linalg.norm(z)

        return np.vstack((x, y, z))

    def gcs2lcs(self, coors: np.ndarray) -> np.ndarray:
        """transform from GCS to this LCS"""
        return (coors - self.o) @ self.T.T

    def lcs2gcs(self, coors: np.ndarray) -> np.ndarray:
        """transform from this LCS to GCS"""
        return coors @ self.T + self.o



# ---------------------------------------------------------- READ LS-DYNA --- {{1
def read_k_line(kfile, kfilename: str, line_no: int) -> (int, int, str):
    """read one line from *.k file"""
    last_pos = kfile.tell()
    line = kfile.readline()
    line_no += 1
    return last_pos, line_no, line


def read_k_nodes(kfile, kfilename: str, line_no: int) -> dict:
    """read nodes from LS-DYNA *.k file"""
    print(f'    Reading *NODE records')
    nodes = {}
    while True:
        try:
            last_pos, line_no, line = read_k_line(kfile, kfilename, line_no)
        except Exception as e:
            print(f'[-] ERROR while reading line {line_no+1:n} of {kfilename:s}')
            print(f'    last line read: {line:s}')
            break
        if not line:                  # EOF
            break
        line = line.rstrip()
        # print(line)
        if line.startswith('$'):    # skip comments
            continue
        elif line.startswith('*'):    # next command breaks it
            kfile.seek(last_pos)
            line_no -= 1
            break
        else:                         # read node record
            nid = int(line[:8].strip())
            coors = [float(line[8+16*i:8+16*(i+1)]) for i in range(3)]
            nodes.setdefault(nid, np.array(coors, dtype=float))

    print(f'    {len(nodes.keys()):n} *NODE records read.')

    return nodes, line_no


def read_k_displacements(kfile, kfilename: str, line_no: int, time_value: float) -> dict:
    """read displacements from LS-DYNA *.k file"""
    print(f'    Reading $NODAL_DISPLACEMENT records for time: {time_value:16.7e}')
    displacements = {}
    while True:
        try:
            last_pos, line_no, line = read_k_line(kfile, kfilename, line_no)
        except Exception as e:
            print(f'[-] ERROR while reading line {line_no+1:n} of {kfilename:s}')
            print(f'    last line read: {line:s}')
            break
        if not line:                  # EOF
            break
        line = line.rstrip()
        if line.startswith('$'):      # comment breaks it
            kfile.seek(last_pos)
            line_no -= 1
            break
        elif line.startswith('*'):    # next command breaks it
            kfile.seek(last_pos)
            line_no -= 1
            break
        else:                         # read node record
            nid = int(line[:8].strip())
            coors = [float(line[8+16*i:8+16*(i+1)]) for i in range(3)]
            displacements.setdefault(nid, np.array(coors, dtype=float))

    print(f'    {len(displacements.keys()):n} $NODAL_DISPLACEMENT records read.')

    return displacements, line_no


def read_k_file(kfilename: str) -> dict:
    """read *.k file"""
    if not os.path.isfile(kfilename):
        raise ValueError(f'[-] ERROR: {kfilename:s} does not exist.')
    else:
        print(f'[+] Reading {kfilename:s}')

    nodes = None
    displacements = None
    time_value = 0.

    with open(kfilename, 'rt') as kfile:
        line_no = 0
        while True:
            try:
                last_pos, line_no, line = read_k_line(kfile, kfilename, line_no)
            except Exception as e:
                print(f'[-] ERROR while reading line {line_no+1:n} of {kfilename:s}')
                print(f'    last line read: {line:s}')
                break

            if not line:                            # EOF
                break

            line = line.rstrip()

            if line.upper().startswith('$TIME_VALUE'):
                time_value = float(line.split('=')[1].strip())
                continue

            elif line.upper().startswith('$NODAL_DISPLACEMENT'): # read displacements
                disps, line_no = read_k_displacements(kfile, kfilename, line_no, time_value)
                if displacements is None:
                    displacements = {}
                displacements.setdefault(time_value, disps)

            elif line.startswith('$'):              # skip comments
                continue

            elif line.upper().startswith('*NODE'): # read nodes
                nds, line_no = read_k_nodes(kfile, kfilename, line_no)
                if nodes is None:
                    nodes = {}
                nodes.update(nds)
                # break # file is read

            else:                                   # other commands
                continue

    return nodes, displacements


def read_k_files(kfilename: str, dispfilename: str) -> (dict, dict):
    """reads both the nodes and the displacements"""
    # read original node coordinates
    nodes_orig, _ = read_k_file(kfilename)

    # read displacements in GCS
    _, displacements = read_k_file(dispfilename)

    return nodes_orig, displacements


# --------------------------------------------------------- WRITE LS-DYNA --- {{1
def write_k_displacements(basename: str, displacements: dict):
    """write displacements into *.k file"""
    filename = basename + '.k'
    print(f'[+] Writing $NODAL_DISPLACEMENT to {filename:s}')
    with open(filename, 'wt') as kfile:
        for i, time in enumerate(displacements.keys()):
            kfile.write(_HEADER.format(time, i + 1))
            for nid, coors in displacements[time].items():
                kfile.write(_LINE_FMT(nid, coors))
    print(f'    Done.')



def write_csv_displacements(basename: str, displacements: dict):
    """write displacements into *.csv file"""

    for i, time in enumerate(displacements.keys()):
        filename = basename + '_t=' + f'{time:16.7e}'.strip() + '.csv'
        print(f'[+] Writing $NODAL_DISPLACEMENT to {filename:s}')

        with open(filename, 'wt') as csvfile:
            csvfile.write(_HEADER_CSV.format(time, i + 1))

            for nid, coors in displacements[time].items():
                csvfile.write(_LINE_CSV_FMT(nid, coors))

        print(f'    Done.')


# ----------------------------------------------------------- READ PERMAS --- {{1
def get_params_map(string: str) -> str:
    if '=' in string:
        string = string.replace('=', ' = ')
    while '  ' in string:
        string = string.replace('  ', ' ')
    string = string.replace(' =', '=')

    params_map = {}
    has_spaces = False
    key = None
    for item in string.split(' '):
        if item.endswith('='):
            key = item.rstrip('=').upper()
            params_map.setdefault(key, [])

        elif key is None:
            params_map.setdefault(item, [])

        else:
            if item.startswith("'") and not item.endswith("'"):
                has_spaces = True
                params_map[key] = [item]
            elif has_spaces:
                if item.endswith("'"):
                    has_spaces = False
                params_map[key][-1] += ' ' + item
            else:
                params_map[key].append(item.upper())

    for key, val in params_map.items():
        if len(val) == 0:
            params_map[key] = None
        elif len(val) == 1:
            params_map[key] = val[0]

    return params_map


def strip_permas_line(line: str) -> str:
    """
    Processes a line string:
        - strips all whitespaces from beggining and end
        - replaces all mutlispaces with single spaces
        - if line starts with continuation character, separates if by space
        - surrounds all '=' by single spaces
        - strips all comments
    """
    line = line.strip()
    if '!' in line:
        line = line.split('!')[0]
    if '=' in line:
        line = line.replace('=', ' = ')
    if line.startswith('&'):
        line = line.replace('&', '& ')
    while '  ' in line:
        line = line.replace('  ', ' ')
    return line.strip(' ')


def read_permas_line_skip_empty(f) -> (int, str):
    """Read next line from file, skip empty and commented lines

    Returns:
        (int, str): last position in file, line stripped from comments,
                    whitespace characters and multiple spaces
    """
    last_pos = f.tell()
    line = f.readline()
    if not line:                           # EOF
        return last_pos, line
    line = strip_permas_line(line)
    if line in ('', ' '):
        last_pos, line = read_permas_line_skip_empty(f)
    return last_pos, line


def read_permas_line(f) -> (int, str):
    last_pos, line = read_permas_line_skip_empty(f)
    if not line:                          # EOF
        return last_pos, line

    while True:
        _last_pos, _line = read_permas_line_skip_empty(f)
        if not _line.startswith('&'):
            f.seek(_last_pos)
            break
        line += _line[1:]

    return last_pos, line


def read_dat_nodes(dat: io.TextIOWrapper, nodes: set = None):
    """reads nodes from file

    Args:
        dat (io.TextIOWrapper): PERMAS *.dat file handle
        nodes (set):            list with node IDs to read.
                                If None, reads all nodes into memory

    Returns:
        (dict): {nodeID: np.array([x, y, z])}
    """

    last_pos, line = read_permas_line(dat)
    param_map = get_params_map(line)
    if '$COOR' not in param_map.keys():
        raise IOError(f"Missing $COOR keyword in line: {line:s}")

    coors = {}
    while True:
        last_pos, line = read_permas_line(dat)
        if not line: # EOF
            break
        elif line.startswith('$'):
            break
        else:
            node = line.split()
            nid = int(node[0])
            coor = np.array([float(x) for x in node[1:4]], dtype=float)
            if nodes is None:
                coors[nid] = coor
            elif nid in nodes:
                coors[nid] = coor
    return coors


def read_dat_structure(dat: io.TextIOWrapper, nodes: set = None) -> dict:
    """reads *.dat file $STRUCTURE variant

    Args:
        dat (io.TextIOWrapper): PERMAS *.dat file handle
        nodes (set):            list with node IDs to read.
                                If None, reads all nodes into memory

    Returns:
        (dict): {nodeID: np.array([x, y, z])}
    """

    coors = {}
    last_pos, line = read_permas_line(dat)
    param_map = get_params_map(line)
    if '$STRUCTURE' not in param_map.keys():
        raise IOError(f"Missing $STRUCTURE keyword in line: {line:s}")

    while True:
        last_pos, line = read_permas_line(dat)
        if not line: # EOF
            break
        elif line.strip().upper().startswith('$FIN'):
            break
        elif line.strip().upper().startswith('$END STRUCTURE'):
            break
        elif line.strip().upper().startswith('$COOR'):
            dat.seek(last_pos)
            coors = read_dat_nodes(dat, nodes)
        else:
            continue

    return coors


def read_post_results(post: io.TextIOWrapper, nodes: set = None, skip_rotations: bool = True) -> dict:
    """Read one $RESULT variant from *.post file

    Args:
        post (io.TextIOWrapper): *.post file handle
        nodes (set):             list with node IDs to read.
                                 If None, reads all nodes into memory
        skip_rotations (bool):   True = read displacements only

    Returns:
        (dict): {time: {nodeid: np.array([dX, dY, dZ])}}
    """
    dirs = {'U': 0, 'V': 1, 'W': 2, 'PHIU': 3, 'PHIV': 4, 'PHIW': 5}
    node = 0
    time = 0.0

    last_pos, line = read_permas_line(post)
    param_map = get_params_map(line)
    if '$RESULTS' not in param_map.keys():
        raise IOError(f"Missing $RESULTS keyword in line: {line:s}")
    freq = [float(f[1:]) for f in param_map['NAME'].split('_') if f.lower().startswith('f')]
    if len(freq) == 0:
        raise ValueError(f"[-] no frequency found in $RESULTS NAME = {param_map['NAME']:s}.")
    freq = freq[0]

    print(f"    Frequency: {freq:13.5E}")

    displacements = {}
    while True:
        last_pos, line = read_permas_line(post)
        if not line: # EOF
            break
        elif line.strip().upper().startswith('$END RESULTS'):
            break
        elif line.strip().upper().startswith('$DATA X_XYDATA'):
            param_map = get_params_map(line)
            node = param_map['CURVE'].strip("'").split(',')
            dof = dirs[node[1]]
            node = int(node[0].lstrip('N'))
            if nodes is not None and node not in nodes: # skip unwanted
                continue
            if skip_rotations and dof > 2:              # skip rotation dof
                continue
            while True:
                last_pos, line = read_permas_line(post)
                if not line: # EOF
                    break
                elif line.strip().startswith('$'):
                    break
                else:
                    time, value = [float(x) for x in line.split(' ')[1:3]]
                    if time not in displacements.keys():
                        displacements.setdefault(time, {})
                    if node not in displacements[time].keys():
                        displacements[time].setdefault(node, np.array([0.] * (3 if skip_rotations else 6), dtype=float))
                    displacements[time][node][dof] = value
        else:
            continue

    return {freq: displacements}


def read_permas_file(filename: str, nodes: set = None) -> (dict, dict):
    """Read data from from PERMAS *.dat or *.post file

    Args:
        filename (str):  [*.dat/*.dat.gz/*.post/*.post.gz] filename
        nodes (set):     list with node IDs to read.
                         If None, reads all nodes into memory

    Returns:
        (dict, dict): {nodeID: [x, y, z]}, {frequency: {time: {nodeID: np.array([dX, dY, dZ])}}}
    """
    coors = {}
    displacements = {}

    print(f"[+] Reading {filename:s}")
    START = time.time()

    if nodes is not None and type(nodes) is not set:
        nodes = set(nodes)

    if filename.lower().endswith('.gz'):
        permas = gzip.open(filename, 'rt')
    else:
        permas = open(filename, 'rt')

    try:
        while True:
            last_pos, line = read_permas_line(permas)
            if not line: # EOF
                break

            elif line.strip().upper().startswith('$FIN'):
                break

            elif line.strip().upper().startswith('$STRUCTURE'):
                permas.seek(last_pos)
                coors.update(read_dat_structure(permas, nodes))

            elif line.strip().upper().startswith('$RESULTS'):
                permas.seek(last_pos)
                displacements.update(read_post_results(permas, nodes, skip_rotations=True))

            else:
                continue

    except Exception as e:
        permas.close()
        raise e

    permas.close()
    END = time.time()
    print(f"    File read in {END - START:f} seconds ({(END - START) / 60.:f} minutes).")

    return coors, displacements


# ---------------------------------------------------------- WRITE PERMAS --- {{1
def write_post_xy(common_prefix: str, displacements: dict, gz: bool = False):
    """write *.post $DATA X_XYDATA

    Args:
        common_prefix (str):  common prefix of the filename to write to
                              (_relative_disp_globalmax.post will be added)
        displacements (dict): a dictionary with displacements
                              {node: {frequency: [min, max, maxabs]}}
        gz (bool):            True = resulting file will be gzipped
    """
    filename = common_prefix + '_relative_xydisp_globalmax.post'
    if gz:
        filename += '.gz'
        post = gzip.open(filename, 'wt')
    else:
        post = open(filename, 'wt')

    nodes = list(sorted(list(displacements.keys())))
    freqs = list(sorted(list(displacements[nodes[0]].keys())))
    curves = list(displacements[nodes[0]][freqs[0]].keys())

    print(f"[+] Writing {filename:s}")
    try:
        post.write(_POST_HEADER)
        post.write(_POST_HEADER_RESULTS.format('SST_EVAL'))
        for node in nodes:
            for curve in curves:
                post.write(_POST_HEADER_CURVE.format('FREQUENCY', node, curve))
                for i, freq in enumerate(freqs):
                    post.write(_POST_CURVE_LINE(i + 1, freq, displacements[node][freq][curve]))
        post.write(_POST_FOOTER_RESULTS)
        post.write(_POST_FOOTER)

    except Exception as e:
        post.close()
        raise e

    post.close()
    print(f'    Done.')


def open_post_xy(common_prefix: str, resultsnamme: str = 'SST_EVAL', gz: bool = False):
    """creates *.post $DATA X_XYDATA file

    Args:
        common_prefix (str):  common prefix of the filename to write to
                              (_relative_disp_globalmax.post will be added)
        resultsname (str):    $RESULTS variant name
        gz (bool):            True = resulting file will be gzipped
    """
    filename = common_prefix + '_relative_xydisp_globalmax.post'
    if gz:
        filename += '.gz'
        post = gzip.open(filename, 'wt')
    else:
        post = open(filename, 'wt')

    print(f"[+] Writing {filename:s}")
    try:
        post.write(_POST_HEADER)
        post.write(_POST_HEADER_RESULTS.format(resultsname))
    except Exception as e:
        post.close()
        raise e

    print(f"[+] created File {filename:s}")
    return post


def close_post_xy(post: io.TextIOWrapper):
    try:
        post.write(_POST_FOOTER_RESULTS)
        post.write(_POST_FOOTER)

    except Exception as e:
        post.close()
        raise e

    post.close()
    print(f"[+] File {post.name:s} written.")


def append_post_xy(post: io.TextIOWrapper, displacements: dict, gz: bool = False):
    """write *.post $DATA X_XYDATA

    Args:
        post (io.TextIOWrapper):  common prefix of the filename to write to
                              (_relative_disp_globalmax.post will be added)
        displacements (dict): a dictionary with displacements
                              {node: {frequency: [min, max, maxabs]}}
        gz (bool):            True = resulting file will be gzipped
    """
    try:
        post.write(_POST_HEADER)
        post.write(_POST_HEADER_RESULTS.format('SST_EVAL'))
        for node in displacements.keys():
            for curve in ['min', 'max', 'maxabs']:
                post.write(_POST_HEADER_CURVE.format('FREQUENCY', node, curve))
                for i, freq in enumerate(sorted(list(displacements[node].keys()))):
                    post.write(_POST_CURVE_LINE(i + 1, freq, displacements[node][freq][curve]))
        post.write(_POST_FOOTER_RESULTS)
        post.write(_POST_FOOTER)

    except Exception as e:
        post.close()
        raise e


def write_post_disp(common_prefix: str, displacements: dict, gz: bool = False):
    """write *.post $DATA DISP

    Args:
        common_prefix (str):  common prefix of the filename to write to
                              (_fXXXX_relative_disp.post will be added)
        displacements (dict): a dictionary with displacements
                              {frequency: {time: {node: [dX, dY, dZ, ..]}}}
        gz (bool):            True = resulting file will be gzipped
    """

    frequencies = list(sorted(list(displacements.keys())))
    for freq in frequencies:
        filename = common_prefix + f'_f{int(freq):04n}_relative_disp.post'
        if gz:
            filename += '.gz'
            post = gzip.open(filename, 'wt')
        else:
            post = open(filename, 'wt')

        print(f"[+] Writing {filename:s}")
        print(f"    Frequency: {freq:13.5E}")
        try:
            post.write(_POST_HEADER)
            post.write(_POST_HEADER_RESULTS.format(f'SST_F{int(freq):04n}'))
            times = list(sorted(list(displacements[freq].keys())))
            for time in times:
                nodes = list(sorted(list(displacements[freq][time].keys())))
                numdof = len(displacements[freq][time][nodes[0]])
                post.write(_POST_HEADER_DISP.format(time, numdof))
                for node in nodes:
                    post.write(_POST_CURVE_DISP(node, displacements[freq][time][node]))
            post.write(_POST_FOOTER_RESULTS)
            post.write(_POST_FOOTER)

        except Exception as e:
            post.close()
            raise e

        post.close()
        print(f'    Done.')


# ------------------------------------------------------------------ MAIN --- {{1
def transform_displacements(nodes_orig: dict, displacements: dict,
                            csys_nodeids: list, csys_form: str) -> dict:
    """the main function of this script - transforms the displacements to
    be relative to moving reference frame

    Args:
        nodes_orig (dict):    nodes coordinates in GCS - undeformed {nodeID: [x, y, z]}
        displacements (dict): displacements of the nodes {loadcase: {nodeID: [dX, dY, dZ]}}
        csys_nodeids (list):  3 node IDs to define the CSys [origin, point on axis, point in plane]
        csys_form (str):      CSys formulation (XY/YX/XZ/ZX/YZ/ZY), first letter specifies axis,
                              both letters specify the plane of the csys_nodids

    Returns:
        (dict): transformed displacements relative to the moving reference frame
                in GCS {loadcase: {nodeID: [dX', dY', dZ']}}
    """
    # store the node IDs for CSYS
    O  = csys_nodeids[0] # origin
    P1 = csys_nodeids[1] # node on +Z axis
    P2 = csys_nodeids[2] # node in +XZ plane

    # get node IDs
    nids = list(sorted(list(nodes_orig.keys())))

    # sort the coors
    coor_orig = np.array([nodes_orig[nid] for nid in nids], dtype=float)

    # create LCS of undeformed state
    cart_orig = CART(nodes_orig[O], nodes_orig[P1], nodes_orig[P2], csys_form)

    # get undeformed coordinates in LCS
    print('[+] transforming original coordinates to LCS')
    coor_orig_lcs = cart_orig.gcs2lcs(coor_orig)

    # process timesteps
    for time in displacements.keys():
        print(f'[+] Processing time: {time:16.7e}')

        # get displacements for the orig coordinates
        disps_gcs = np.array([displacements[time][nid] for nid in nids], dtype=float)

        # get coordinates of deformed state
        print('    creating deformed state in GCS')
        coor_deformed = coor_orig + disps_gcs

        # create LCS of deformed state
        cart_deformed = CART(nodes_orig[O]  + displacements[time][O],
                             nodes_orig[P1] + displacements[time][P1],
                             nodes_orig[P2] + displacements[time][P2],
                             csys_form)

        # get deformed coordinates in LCS
        print('    transforming deformed state to LCS')
        coor_deformed_lcs = cart_deformed.gcs2lcs(coor_deformed)

        # get displacements in LCS
        print('    creating displacements in LCS: subtracting LCS original state from LCS deformed state')
        disps_lcs = coor_deformed_lcs - coor_orig_lcs

        # transform back to GCS of the undeformed state
        print('    transforming displacements from LCS to GCS')
        disps_gcs = disps_lcs @ cart_orig.T

        # store processed data
        displacements[time] = dict(zip(nids, disps_gcs))

    return displacements


def permas_transform(datfilename: str, postfilenames: list, csys_nodeids: list, csys_form: str,
                     globalmax: bool = True, all: bool = True):
    dirs = {'U': 0, 'V': 1, 'W': 2, 'PHIU': 3, 'PHIV': 4, 'PHIW': 5}
    dofs = {v: k for k, v in dirs.items()}

    # sequential - read displacements of the 1st file to get all node numbers
    _, displacements = read_permas_file(postfilenames[0])
    freq = list(displacements.keys())[0]
    nodes = set(displacements[freq][list(displacements[freq].keys())[0]].keys())

    nodes_orig, _ = read_permas_file(datfilename, None if all else nodes)
    print(f"[+] Filtering node coordinates")
    nodes_orig = {nid: coor for nid, coor in nodes_orig.items() if nid in nodes}

    if all:
        eval_nodes = nodes
    else:
        eval_nodes = [node for node in nodes if node not in csys_nodeids]

    common_prefix = os.path.commonprefix(postfilenames)
    common_prefix = '_'.join([cp for cp in common_prefix.split('_') if not cp.startswith('f')])

    disp_eval = {}

    # for freq in all_displacements.keys():
    for i, post_file in enumerate(postfilenames):
        # do not read first *.post file twice
        if i != 0:
            _, displacements = read_permas_file(post_file)
            freq = list(displacements.keys())[0]

        displacements = displacements[freq]

        print(f"[+] Processing Frequency {freq:.2f}")
        displacements = transform_displacements(nodes_orig, displacements, csys_nodeids, csys_form)

        if globalmax:
            for time in displacements.keys():
                for node in eval_nodes:
                    if node not in disp_eval.keys():
                        disp_eval.setdefault(node, {})

                    if freq not in disp_eval[node].keys():
                        disp_eval[node].setdefault(freq, {})

                    for dof in range(3):
                        val = displacements[time][node][dof]
                        d = dofs[dof]
                        for v in [d + 'min', d + 'max', d + 'maxabs']:
                            if v not in disp_eval[node][freq].keys():
                                disp_eval[node][freq].setdefault(v, val)

                        v = d + 'min'
                        disp_eval[node][freq][v] = val if val < disp_eval[node][freq][v] else disp_eval[node][freq][v]
                        v = d + 'max'
                        disp_eval[node][freq][v] = val if val > disp_eval[node][freq][v] else disp_eval[node][freq][v]
                        v = d + 'maxabs'
                        disp_eval[node][freq][v] = val if abs(val) > abs(disp_eval[node][freq][v]) else disp_eval[node][freq][v]

                    val = np.linalg.norm(displacements[time][node][:3])
                    d = 'R'
                    for v in [d + 'min', d + 'max']:
                        if v not in disp_eval[node][freq].keys():
                            disp_eval[node][freq].setdefault(v, val)

                    v = d + 'min'
                    disp_eval[node][freq][v] = val if val < disp_eval[node][freq][v] else disp_eval[node][freq][v]
                    v = d + 'max'
                    disp_eval[node][freq][v] = val if val > disp_eval[node][freq][v] else disp_eval[node][freq][v]

        else:
            # sequential writing results to free memory
            write_post_disp(common_prefix, {freq: displacements}, False)

    if globalmax:
        write_post_xy(common_prefix, disp_eval, False)



# ------------------------------------------------------------ ENTRYPOINT --- {{1
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-p', dest='permas', action='store_true', help='process permas SST results')

    parser.add_argument('-gm', dest='globalmax', action='store_true',
                        help="""if PERMAS SST result, create globalmax over frequency, if not switched,
then processes each time step of each frequency separately, otherwise create globalmax
over frequencies""")

    parser.add_argument('-a', dest='all', action='store_true',
                        help="""if PERMAS SST result, export all nodes if true, if false
export only nodes not used to define csys.""")

    parser.add_argument('-c', dest='nodeID', nargs=3, type=int, required=True,
                        help='node number of origin, node on vector and node in plane defining the csys based on formulation')

    parser.add_argument('-f', dest='csys_form', nargs=1, type=str, required=False, default=["XY"],
                        help='local coordinate system formulation, default is "XY"')


    parser.add_argument(nargs=1, dest='geo_file',
                        help='*.k file or *.dat file with the nodes of the structure')

    parser.add_argument(nargs='+', dest='disp_files',
                        help='ascii files with the nodal displacements')

    args = parser.parse_args()

    if args.permas:
        permas_transform(args.geo_file[0], args.disp_files, args.nodeID, args.csys_form[0], args.globalmax, args.all)

    else:
        for disp_file in args.disp_files:
            nodes_orig, displacements = read_k_files(args.geo_file[0], disp_file)

            displacements = transform_displacements(nodes_orig, displacements, args.nodeID, args.csys_form[0])

            basename = os.path.splitext(args.geo_file[0])[0] + '_relative_' + os.path.splitext(os.path.basename(disp_file))[0]

            write_k_displacements(basename, displacements)
            write_csv_displacements(basename, displacements)

