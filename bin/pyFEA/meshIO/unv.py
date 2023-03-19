import os
import sys
import io
import math

import pdb

DELIMITER = f"{-1:6n}"
DATASET = lambda x: f"{x:>6n}\n"

DATASETS = {  15: "NODE",
            2411: "NODE2P",
            2412: "ELEMENT",
              55: "NODAL",
              56: "ELEMENTAL",
             773: "MATERIAL",
              18: "CSYS"}

NODES = {  "NODE": "single",
         "NODE2P": "double"}

ELEMENTS = { 11: "ROD",    # Rod                                       Edge Lagrange P1
             21: "BEAM",   # Linear Beam                               Edge Lagrange P1
             22: "BEAMT",  # Tapered Beam                              Edge Lagrange P2
             24: "BEAMP",  # Parabolic Beam                            Edge Lagrange P2
             41: "TRIA3",  # Plane Stress Linear Triangle              Triangle Lagrange P1
             42: "TRIE6",  # Plane Stress Parabolic Triangle           Triangle Lagrange P2
             44: "QUAD4",  # Plane Stress Linear Quadrilateral         Quadrilateral Lagrange P1
             45: "QUAD8",  # Plane Stress Parabolic Quadrilateral      Quadrilateral Lagrange P2
             81: "TRIAS3", # Axisymetric Solid Linear Triangle         Triangle Lagrange P1
             82: "TRIAS6", # Axisymetric Solid Parabolic Triangle      Triangle Lagrange P2
             84: "QUAAS4", # Axisymetric Solid Linear Quadrilateral    Quadrilateral Lagrange P1
             85: "QUAAS8", # Axisymetric Solid Parabolic Quadrilateral Quadrilateral Lagrange P2
             91: "TRITS3", # Thin Shell Linear Triangle                Triangle Lagrange P1
             92: "TRITS6", # Thin Shell Parabolic Triangle             Triangle Lagrange P2
             94: "QUATS4", # Thin Shell Linear Quadrilateral           Quadrilateral Lagrange P1
             95: "QUATS8", # Thin Shell Parabolic Quadrilateral        Quadrilateral Lagrange P2
            111: "TET4",   # Solid Linear Tetrahedron                  Tetrahedron Lagrange P1
            112: "PENTA6", # Solid Linear Wedge                        Wedge Lagrange P1
            115: "HEX8",   # Solid Linear Brick                        Hexahedron Lagrange P1
            116: "HEX20",  # Solid Parabolic Brick                     Hexahedron Lagrange P2
            118: "TET10",  # Solid Parabolic Tetrahedron               Tetrahedron Lagrange P2
            122: "RBE2"}   # Rigid Element                             Quadrilateral Lagrange P1

MODEL_TYPE = {0: "Unknown",
              1: "Structural",
              2: "Heat Transfer",
              3: "Fluid Flow"}

ANALYSIS_TYPE = {0: "Unknown",
                 1: "Static",
                 2: "Normal Mode",
                 3: "Complex eigenvalue first order",
                 4: "Transient",
                 5: "Frequency Response",
                 6: "Buckling",
                 7: "Complex eigenvalue second order"}

DATA_CHARACTERISTIC = {0: "Unknown",
                       1: "Scalar",
                       2: "3 DOF Global Translation Vector",
                       3: "6 DOF Global Translation & Rotation Vector",
                       4: "Symmetric Global Tensor",
                       5: "General Global Tensor",
                       6: "Stress Resultants"}

SPECIFIC_DATA_TYPE = { 0: "Unknown",
                       1: "General",
                       2: "Stress",
                       3: "Strain",
                       4: "Element Force",
                       5: "Temperature",
                       6: "Heat Flux",
                       7: "Strain Energy",
                       8: "Displacement",
                       9: "Reaction Force",
                      10: "Kinetic Energy",
                      11: "Velocity",
                      12: "Acceleration",
                      13: "Strain Energy Density",
                      14: "Kinetic Energy Density",
                      15: "Hydro-Static Pressure",
                      16: "Heat Gradient",
                      17: "Code Checking Value",
                      18: "Coefficient Of Pressure"}

SPECIFIC_DATA_TYPE = { 0: "Unknown",
                       1: "General",
                       2: "Stress",
                       3: "Strain",
                       4: "Element force",
                       5: "Temperature",
                       6: "Heat flux",
                       7: "Strain energy",
                       8: "Displacement",
                       9: "Reaction force",
                      10: "Kinetic energy",
                      11: "Velocity",
                      12: "Acceleration",
                      13: "Strain energy density",
                      14: "Kinetic energy density",
                      15: "Hydro-static pressure",
                      16: "Heat gradient",
                      17: "Code checking value",
                      18: "Coefficient of pressure",
                      19: "Ply stress",
                      20: "Ply strain",
                      21: "Failure index for ply",
                      22: "Failure index for bonding",
                      23: "Reaction heat flow",
                      24: "Stress error density",
                      25: "Stress variation",
                      27: "Shell and plate elem stress resultant",
                      28: "Length",
                      29: "Area",
                      30: "Volume",
                      31: "Mass",
                      32: "Constraint forces",
                      34: "Plastic strain",
                      35: "Creep strain",
                      36: "Strain energy error",
                      37: "Dynamic stress at nodes",
                      93: "Unknown",
                      94: "Unknown scalar",
                      95: "Unknown 3DOF vector",
                      96: "Unknown 6DOF vector",
                      97: "Unknown symmetric tensor",
                      98: "Unknown global tensor",
                      99: "Unknown shell and plate resultant"}

DATA_TYPE = {1: "Integer",
             2: "Real",
             4: "Real Double",
             5: "Complex",
             6: "Complex Double"}

#               ID  [name, number of properties, max number of values, [material property IDs]]
MATERIAL_TYPE = {0: ["NULL", 0, 0, []],
                 1: ["ISOTROPIC", 29, 29, [1, 2, 3, 4, 5, 7, 8, 6, 9, 117, 10, 11, 12,
                                           13, 14, 15, 16, 17, 18, 19, 20, 118, 119, 120,
                                           121, 122, 123, 309, 310]],
                 2: ["ORTHOTROPIC", 35, 35, [101, 102, 103, 104, 105, 106, 3, 108, 109,
                                             110, 111, 112, 113, 7, 8, 114, 115, 116, 117,
                                             11, 13, 14, 9, 15, 16, 118, 119, 120, 121,
                                             122, 123, 124, 125, 309, 310]],
                 3: ["ANISOTROPIC", 22, 22, [201, 202, 203, 204, 205, 206, 3, 208, 7, 8,
                                             209, 9, 118, 119, 120, 121, 122, 123, 124, 125,
                                             309, 310]],
                 4: ["LAMINATE", 12, 12, [301, 302, 303, 304, 305, 306, 307, 3, 308, 309,
                                          310, 311]]}

#                     ID [value name, data type, number of values]
MATERIAL_PROPERTY = {
      1: ["E",      2, 1], # MODULUS OF ELASTICITY                      PRESSURE
      2: ["NU",     2, 1], # POISSONS RATIO                             NO UNITS
      3: ["DEN",    2, 1], # MASS DENSITY                               MASS DENSITY
      4: ["G",      2, 1], # SHEAR MODULUS                              PRESSURE
      5: ["A",      2, 1], # COEFFICIENT THERMAL EXPANSION              STRAIN/TEMPERATURE
      6: ["K",      2, 1], # THERMAL CONDUCTIVITY                       CONDUCTIVITY
      7: ["TREF",   2, 1], # THERMAL EXPANSION REFERENCE TEMPERATURE    TEMPERATURE
      8: ["GE",     2, 1], # STRUCTURAL ELEMENT DAMPING COEFFICIENT     NO UNITS
      9: ["CP",     2, 1], # SPECIFIC HEAT                              SPECIFIC HEAT
     10: ["YS",     2, 1], # YIELD STRESS                               PRESSURE
     11: ["CF",     2, 1], # CONVECTIVE FILM COEFFICIENT                CONVECTION COEFFICIE
     12: ["TC",     2, 1], # THERMAL CAPACITY PER UNIT AREA             THERMAL CAPACITY
     13: ["HF",     2, 1], # HEAT FLUX RATE                             HEAT FLUX / LENGTH
     14: ["SHF",    2, 1], # SURFACE HEAT FLUX RATE                     HEAT FLUX / AREA
     15: ["V",      2, 1], # VISCOSITY                                  FORCE*TIME/L**2
     16: ["MU",     2, 1], # COEFFICIENT OF FRICTION                    NO UNITS
     17: ["AF",     2, 1], # AREA FACTOR                                NO UNITS
     18: ["EM",     2, 1], # EMISSIVITY                                 NO UNITS
     19: ["AB",     2, 1], # ABSORPTIVITY                               NO UNITS
     20: ["SC",     2, 1], # SWELLING COEFFICIENT                       NO UNITS
    101: ["EX",     2, 1], # MODULUS OF ELASTICITY X                    PRESSURE
    102: ["EY",     2, 1], # MODULUS OF ELASTICITY Y                    PRESSURE
    103: ["EZ",     2, 1], # MODULUS OF ELASTICITY Z                    PRESSURE
    104: ["NUXY",   2, 1], # POISSONS RATIO XY                          NO UNITS
    105: ["NUYZ",   2, 1], # POISSONS RATIO YZ                          NO UNITS
    106: ["NUXZ",   2, 1], # POISSONS RATIO XZ                          NO UNITS
    108: ["GXY",    2, 1], # SHEAR MODULUS XY                           PRESSURE
    109: ["GYZ",    2, 1], # SHEAR MODULUS YZ                           PRESSURE
    110: ["GXZ",    2, 1], # SHEAR MODULUS XZ                           PRESSURE
    111: ["AX",     2, 1], # COEFFICIENT THERMAL EXPANSION X            STRAIN/TEMPERATURE
    112: ["AY",     2, 1], # COEFFICIENT THERMAL EXPANSION Y            STRAIN/TEMPERATURE
    113: ["AZ",     2, 1], # COEFFICIENT THERMAL EXPANSION Z            STRAIN/TEMPERATURE
    114: ["KX",     2, 1], # THERMAL CONDUCTIVITY X                     CONDUCTIVITY
    115: ["KY",     2, 1], # THERMAL CONDUCTIVITY Y                     CONDUCTIVITY
    116: ["KZ",     2, 1], # THERMAL CONDUCTIVITY Z                     CONDUCTIVITY
    117: ["Q",      2, 1], # HEAT GENERATION RATE                       HEAT/VOLUME*TIME
    118: ["XT",     2, 1], # ALLOWABLE STRESS IN TENSION IN X DIR       PRESSURE
    119: ["XC",     2, 1], # ALLOWABLE STRESS IN COMPRESSION IN X DIR   PRESSURE
    120: ["YT",     2, 1], # ALLOWABLE STRESS IN TENSION IN Y DIR       PRESSURE
    121: ["YT",     2, 1], # ALLOWABLE STRESS IN COMPRESSION IN Y DIR   PRESSURE
    122: ["S",      2, 1], # ALLOWABLE IN-PLANE SHEAR STRESS            PRESSURE
    123: ["F12",    2, 1], # INTERACTION TERM FOR TSAI-WU               NO UNITS
    124: ["SCX",    2, 1], # SWELLING COEFFICIENT IN X                  NO UNITS
    125: ["SCY",    2, 1], # SWELLING COEFFICIENT IN Y                  NO UNITS
    201: ["RW1",    2, 6], # ROW 1 MATERIAL PROPERTY MATRIX             PRESSURE
    202: ["RW2",    2, 5], # ROW 2 MATERIAL PROPERTY MATRIX             PRESSURE
    203: ["RW3",    2, 4], # ROW 3 MATERIAL PROPERTY MATRIX             PRESSURE
    204: ["RW4",    2, 3], # ROW 4 MATERIAL PROPERTY MATRIX             PRESSURE
    205: ["RW5",    2, 2], # ROW 5 MATERIAL PROPERTY MATRIX             PRESSURE
    206: ["RW6",    2, 1], # ROW 6 MATERIAL PROPERTY MATRIX             PRESSURE
    208: ["TEV",    2, 6], # THERMAL EXPANSION VECTOR                   STRAIN/TEMPERATURE
    209: ["KKM",    2, 6], # THERMAL CONDUCTIVITY MATRIX                CONDUCTIVITY
    301: ["AMTX",   2, 9], # MEMBRANE PROPERTIES (A MATRIX)             FORCE/LENGTH
    302: ["BMTX",   2, 9], # COUPLED PROPERTIES (B MATRIX)              FORCE
    303: ["DMTX",   2, 9], # BENDING PROPERTIES (D MATRIX)              TORQUE
    304: ["SMTX",   2, 4], # TRANSVERSE SHEAR PROPERTIES (S MATRIX)     FORCE/LENGTH
    305: ["EMTEC",  2, 3], # EFFECTIVE MEM THERMAL EXPANSION COEFF      STRAIN/TEMPERATURE
    306: ["EMBEC",  2, 3], # EFFECTIVE MEM BENDING EXPANSION COEFF      STRAIN/TEMPERATURE
    307: ["EBTEC",  2, 3], # EFFECTIVE BENDING THERMAL EXPANSION COEFF  STRAIN/TEMPERATURE
    308: ["NONSTM", 2, 1], # NON STRUCTURAL MASS                        MASS/AREA
    309: ["DMPCOF", 2, 1], # DAMPING COEFFICIENT                        NO UNITS
    310: ["REFTMP", 2, 1], # REFERENCE TEMPERATURE                      TEMPERATURE
    311: ["LAMTHK", 2, 1], # LAMINATE THICKNESS                         LENGTH
}

CSYS_TYPE = {0: "Cartesian",
             1: "Cylindrical",
             2: "Spherical"}

# not in UNV specification !!!
# CSYS_DEFINITION = {0: "xy",
#                    1: "yx",
#                    2: "xz",
#                    3: "zx",
#                    4: "yz",
#                    5: "zy"}
CSYS =_DEFINITION = {1: "xz"} # origin, +x axis, +xz plane


FORMAT = {"I": int,
          "E": float,
          "D": float,
          "A": str}
INTEGER = lambda x: f"{x:10n}"
SINGLE = lambda x: f"{x:13.5E}"
DOUBLE = lambda x: f"{x:25.16E}".replace("E", "D")
# CSINGLE = lambda x: f"{x.real:13.5E}{x.imag:13.5E}"
# CDOUBLE = lambda x: f"{x.real:25.16E}{x.imag:25.16E}".replace("E", "D")
CHAR = lambda x: f"{x[:80]:80s}".rstrip(" ") if x != "" else "NONE"

_LINENUMBER = 0


class ReadError(Exception):
    pass



def print_dict(dictionary: dict, offset: int = 0):
    for key, items in dictionary.items():
        if type(items) is dict:
            print("  " * offset + str(key))
            print_dict(items, offset + 1)
        else:
            print("  " * (offset + 1) + str(key) + ": " + str(items))



def _update_dict_of_dicts(base_dict: dict, new_dict: dict) -> dict:
    for key, value in new_dict.items():
        if key not in base_dict.keys():
            base_dict.setdefault(key, value)
        else:
            base_dict[key] = _update_dict_of_dicts(base_dict[key], new_dict[key])
    return base_dict



def _read_line(unv, msg_eof: str = None) -> (int, str):
    global _LINENUMBER

    lastPos = unv.tell()
    line = unv.readline().strip("\n")
    if not line: # EOF
        unv.seek(lastPos)
        if msg_eof is not None and msg_eof != "":
            raise ReadError(msg_eof)
        else:
            return lastPos, None
    else:
        _LINENUMBER += 1
        return lastPos, line



def _read_dataset_till_end(unv, dataset: int, err: str = ""):
    while True:
        lastPos, line = _read_line(unv,
            msg_eof = err + f"[-] File ended before closing the dataset {dataset:n}\n")
        if line == DELIMITER:  # end of DATASET
            break



def _read_and_parse_line(unv, format: str, dataset: int, record: int,
                        shorter: bool = False) -> (int, list):
    """
    4I10,3E13.5
    """
    global _LINENUMBER

    lastPos, line = _read_line(unv,
        msg_eof = f"[-] File ended before closing the dataset {dataset:n}\n")
    if line == DELIMITER:  # end of DATASET
        return lastPos, DELIMITER

    fields = []
    _formats = format.split(",")
    formats = []
    totlength = 0
    for f in _formats:
        if "I" in f:
            f = f.split("I")
            dtype = int
        elif "E" in f:
            f = f.split("E")
            dtype = float
        elif "D" in f:
            f = f.split("D")
            dtype = float
        count = int(f[0]) if f[0] != "" else 1
        length = int(f[1].split(".")[0])

        for i in range(count):
            formats.append([length, dtype])
        totlength += length * count

    if not shorter:
        if len(line) != totlength:
            err = f"[-] Wrong length of Record {record:n} on line {_LINENUMBER:n}:\n"
            err += f"    >>> {line:s}\n"
            _read_dataset_till_end(unv, dataset, err)
    else:
        if len(line) > totlength:
            err = f"[-] Wrong length of Record {record:n} on line {_LINENUMBER:n}:\n"
            err += f"    >>> {line:s}\n"
            _read_dataset_till_end(unv, dataset, err)

    position = 0
    for f in formats:
        field = line[position:position+f[0]].replace("D", "E")
        if shorter and field == "":
            break
        fields.append(f[1](field))
        position += f[0]

    return lastPos, fields



def _read_csys(unv) -> dict:
    print(f"[+] Reading Coordinate Systems.")

    lastPos, line = _read_line(unv) # DATASET number
    dataset = int(line.strip())

    err = ""
    csys = {}
    while True:
        lastPos, line = _read_and_parse_line(unv, "5I10", dataset, 1)
        if line == DELIMITER:  # end of DATASET
            break

        cid = line[0]
        ctype = CSYS_TYPE[line[1]]
        cref = line[2]
        color = line[3]
        # cdef = line[4]

        lastPos, line = _read_line(unv,
            msg_eof = f"[-] File ended before closing the dataset {dataset:n}\n")
        if line == DELIMITER:  # end of DATASET
            err = f"[-] Dataset {dataset:n} ended before finishing CSys {cid:n}\n"
            raise ReadError(err)

        cname = line[:40].strip()

        lastPos, line = _read_and_parse_line(unv, "6E13.5", dataset, 1)
        if line == DELIMITER:  # end of DATASET
            err = f"[-] Dataset {dataset:n} ended before finishing CSys {cid:n}\n"
            raise ReadError(err)

        origin = line[:3]
        xaxis = line[3:]

        lastPos, line = _read_and_parse_line(unv, "3E13.5", dataset, 1)
        if line == DELIMITER:  # end of DATASET
            err = f"[-] Dataset {dataset:n} ended before finishing CSys {cid:n}\n"
            raise ReadError(err)

        xzplane = line

        csys.setdefault(cid, {"type": ctype, "ref": cref, "origin": origin,
                              "xaxis": xaxis, "xzplane": xzplane})

    return csys



def _write_csys(csys: dict, comment: str = None) -> str:
    print(f"[+] Writing Coordinate Systems.")
    if comment is not None:
        dataset = "\n".join([CHAR(c) for c in comment.split("\n")]) + "\n"
    else:
        dataset = ""

    dataset += DELIMITER + "\n"

    dataset += DATASET({v: k for k, v in DATASETS.items()}["CSYS"])

    for cid, cs in csys.items():
        ctype = int({v: k for k, v in CSYS_TYPE.items()}[cs["type"]])
        cref = cs["ref"] if "ref" in cs.keys() else 0
        color = cs["color"] if "color" in cs.keys() else 1
        cdef = CSYS_DEFINITION["xz"]
        origin = cs["origin"]
        xaxis = cs["xaxis"]
        xzplane = cs["xzplane"]

        dataset += INTEGER(cid)
        dataset += INTEGER(cref)
        dataset += INTEGER(color)
        dataset += INTEGER(cdef) + "\n"

        dataset += SINGLE(origin) + SINGLE(xaxis) + "\n"
        dataset += SINGLE(xzplane) + "\n"

    dataset += DELIMITER + "\n"

    return dataset



def _read_nodes_single(unv) -> dict:
    print(f"[+] Reading Nodes single precision.")

    lastPos, line = _read_line(unv) # DATASET number
    dataset = int(line.strip())

    err = ""
    nodes = {}
    while True:
        lastPos, line = _read_and_parse_line(unv, "4I10,3E13.5", dataset, 1)
        if line == DELIMITER:  # end of DATASET
            break

        nid = line[0]
        defcsys = line[1]
        outcsys = line[2]
        color = line[3]
        coors = line[4:]

        nodes[nid] = {"id": nid, "def": defcsys, "out": outcsys, "coors": coors}

    return nodes



def _read_nodes_double(unv) -> dict:
    print(f"[+] Reading Nodes double precision.")

    lastPos, line = _read_line(unv) # DATASET number
    dataset = int(line.strip())

    err = ""
    nodes = {}
    while True:
        lastPos, line = _read_and_parse_line(unv, "4I10", dataset, 1)
        if line == DELIMITER:
            break

        nid = line[0]
        defcsys = line[1]
        outcsys = line[2]
        color = line[3]

        lastPos, line = _read_and_parse_line(unv, "3D25", dataset, 2)
        if line == DELIMITER:  # end of DATASET
            err = f"[-] Dataset {dataset:n} ended before finishing Node {nid:n}\n"
            raise ReadError(err)

        nodes[nid] = {"id": nid, "def": defcsys, "out": outcsys, "coors": line}

    return nodes



def _write_nodes_single(nodes: dict, comment: str = None) -> str:
    print(f"[+] Writing Nodes single precision.")
    if comment is not None:
        dataset = "\n".join([CHAR(c) for c in comment.split("\n")]) + "\n"
    else:
        dataset = ""
    dataset += DELIMITER + "\n"
    dataset += DATASET({v: k for k, v in DATASETS.items()}["NODE"])
    color = 1
    for nid, node in nodes.items():
        defsys = 1 if "def" not in node.keys() else node["def"]
        outsys = 1 if "out" not in node.keys() else node["out"]
        dataset += INTEGER(nid)
        dataset += INTEGER(defsys)
        dataset += INTEGER(outsys)
        dataset += INTEGER(color)
        for coor in node["coors"]:
            dataset += SINGLE(coor)
        dataset += "\n"
    dataset += DELIMITER + "\n"
    return dataset



def _write_nodes_double(nodes: dict, comment: str = None) -> str:
    print(f"[+] Writing Nodes double precision.")
    if comment is not None:
        dataset = "\n".join([CHAR(c) for c in comment.split("\n")]) + "\n"
    else:
        dataset = ""
    dataset += DELIMITER + "\n"
    dataset += DATASET({v: k for k, v in DATASETS.items()}["NODE2P"])
    color = 1
    for nid, node in nodes.items():
        defsys = 1 if "def" not in node.keys() else node["def"]
        outsys = 1 if "out" not in node.keys() else node["out"]
        dataset += INTEGER(nid)
        dataset += INTEGER(defsys)
        dataset += INTEGER(outsys)
        dataset += INTEGER(color)
        dataset += "\n"
        for coor in node["coors"]:
            dataset += DOUBLE(coor)
        dataset += "\n"
    dataset += DELIMITER + "\n"
    return dataset



def _write_nodes(nodes: dict, precision: str = "double") -> str:
    if precision == "single":
        dataset = _write_nodes_single(nodes)
    else:
        dataset = _write_nodes_double(nodes)

    return dataset



def _read_elements(unv):
    print(f"[+] Reading Elements.")

    lastPos, line = _read_line(unv) # DATASET number
    dataset = int(line.strip())

    elements = {}
    while True:
        lastPos, line = _read_and_parse_line(unv, "6I10", dataset, 1)
        if line == DELIMITER:
            break

        eid = line[0]
        FEid = line[1]
        etype = ELEMENTS[FEid]
        pid = line[2]
        mid = line[3]
        color = line[4]
        numnodes = line[5]

        # TODO:
        # BEAM elements
        if FEid in [21, 22, 24]:
            lastPos, line = _read_and_parse_line(unv, "3I10", dataset, 2)
            if line == DELIMITER:  # end of DATASET
                err = f"[-] Dataset {dataset:n} ended before finishing Element {eid:n}\n"
                raise ReadError(err)

            beamdef = line


        nid = []
        numlines = int(math.ceil(numnodes / 8))
        for i in range(numlines):
            lastPos, line = _read_and_parse_line(unv, "8I10", dataset, 2, (i + 1) == numlines)
            nid += line

        if etype not in elements.keys():
            elements.setdefault(etype, {})

        elements[etype][eid] = {"mid": mid,
                                "pid": pid,
                                "nodes": nid}

    return elements



def _write_elements(elements: dict, comment: str = None):
    print(f"[+] Writing Elements.")
    if comment is not None:
        dataset = "\n".join([CHAR(c) for c in comment.split("\n")]) + "\n"
    else:
        dataset = ""

    dataset += DELIMITER + "\n"

    dataset += DATASET({v: k for k, v in DATASETS.items()}["ELEMENT"])

    ELEMENT_KEYS = {v: k for k, v in ELEMENTS.items()}

    color = 1

    for etype, els in elements.items():
        for eid in els.keys():
            pid = 1 if "pid" not in els[eid].keys() else els[eid]["pid"]
            mid = 1 if "mid" not in els[eid].keys() else els[eid]["mid"]
            nodes = els[eid]["nodes"]
            numnodes = len(nodes)
            dataset += INTEGER(eid)
            dataset += INTEGER(ELEMENT_KEYS[etype])
            dataset += INTEGER(pid)
            dataset += INTEGER(mid)
            dataset += INTEGER(color)
            dataset += INTEGER(numnodes)
            dataset += "\n"

            # TODO:
            # beam elements
            if ELEMENT_KEYS[etype] in [21, 22, 24]:
                orinode = 0
                endAid = 0
                endBis = 0
                dataset += INTEGER(orinode)
                dataset += INTEGER(endAid)
                dataset += INTEGER(endBid)
                dataset += "\n"

            # nodes
            for i in range(numnodes):
                dataset += INTEGER(nodes[i])
                if (i + 1) % 8 == 0:
                    dataset += "\n"
            if dataset.endswith("\n"):
                if dataset.endswith("\n\n"):
                    dataset = dataset[:-1]
            else:
                dataset += "\n"

    dataset += DELIMITER + "\n"

    return dataset



def _read_material(unv) -> dict:
    print(f"[+] Reading Materials.")
    materials = {}

    lastPos, line = _read_line(unv) # DATASET number
    dataset = int(line.strip())

    # record 1
    lastPos, line = _read_and_parse_line(unv, "3I10", dataset, 1)
    if line == DELIMITER:
        err = f"[-] Dataset {dataset:n} ended before finishing Material Table\n"
        raise ReadError(err)

    matID = line[0]
    mat_desc = line[1]
    numvaltypes = line[2]

    material_desc = MATERIAL_TYPE[mat_desc]
    mattype = material_desc[0]

    # record 2
    lastPos, line = _read_line(unv) # DATASET number
    if line == DELIMITER:
        err = f"[-] Dataset {dataset:n} ended before finishing Material Table {mat_table_ID:n}\n"
        raise ReadError(err)

    mat_name = str(line[:40].strip())

    numvals = sum([MATERIAL_PROPERTY[p][2] for p in material_desc[3][:numvaltypes]])

    vals = []
    numlines = int(math.ceil(numvals / 6))
    i = 0
    while True:
        i += 1
        # record 3
        lastPos, line = _read_and_parse_line(unv, "6E13.5", dataset, 3, i == numlines)
        if line == DELIMITER:
            break

        vals += line

        # TODO:
        # discard at this moment, not sure what it is there for
        # record 4
        lastPos, line = _read_line(unv)
        if line == DELIMITER:
            err = f"[-] Dataset {dataset:n} ended before finishing Material Table {mat_table_ID:n}\n"
            raise ReadError(err)

    i = -1
    material = {"name": mat_name, "type": mattype}
    for p in material_desc[3][:numvaltypes]:
        prop_name = MATERIAL_PROPERTY[p][0]
        prop_count = MATERIAL_PROPERTY[p][2]

        if prop_count == 1:
            i += 1
            property = vals[i]
        else:
            property = []
            for j in range(prop_count):
                i += 1
                property.append(vals[i])
        material.setdefault(prop_name, property)

    materials.setdefault(matID, material)

    return materials



def _read_material2(unv) -> dict:
    print(f"[+] Reading Materials.")
    materials = {}

    lastPos, line = _read_line(unv) # DATASET number
    dataset = int(line.strip())

    # record 1
    lastPos, line = _read_and_parse_line(unv, "3I10", dataset, 1)
    if line == DELIMITER:
        err = f"[-] Dataset {dataset:n} ended before finishing Material Table\n"
        raise ReadError(err)

    matID = line[0]
    mat_desc = line[1]
    numvaltypes = line[2]

    material_desc = MATERIAL_TYPE[mat_desc]
    mattype = material_desc[0]

    # record 2
    lastPos, line = _read_line(unv) # DATASET number
    if line == DELIMITER:
        err = f"[-] Dataset {dataset:n} ended before finishing Material Table {mat_table_ID:n}\n"
        raise ReadError(err)

    mat_name = str(line[:40].strip())

    numvals = sum([MATERIAL_PROPERTY[p][2] for p in material_desc[3][:numvaltypes]])

    vals = []
    numlines = int(math.ceil(numvals / 6))
    l = 0
    while True:
        l += 1
        # record 3
        lastPos, line = _read_and_parse_line(unv, "6E13.5", dataset, 3, True)
        if line == DELIMITER:
            break

        vals += line

        # TODO:
        # discard at this moment, not sure what it is there for
        # record 4
        lastPos, line = _read_line(unv)
        if line == DELIMITER:
            err = f"[-] Dataset {dataset:n} ended before finishing Material Table {mat_table_ID:n}\n"
            raise ReadError(err)

    i = -1
    material = {"name": mat_name, "type": mattype}
    for p in material_desc[3][:numvaltypes+1]:
        prop_name = MATERIAL_PROPERTY[p][0]
        prop_count = MATERIAL_PROPERTY[p][2]

        if prop_count == 1:
            i += 1
            property = vals[i]
        else:
            property = []
            for j in range(prop_count):
                i += 1
                property.append(vals[i])
        material.setdefault(prop_name, property)

    materials.setdefault(matID, material)

    return materials



def _write_material(matID, material: dict, comment: str = None) -> str:
    print(f"[+] Writing Material {matID:n}.")
    if comment is not None:
        dataset = "\n".join([CHAR(c) for c in comment.split("\n")]) + "\n"
    else:
        dataset = ""

    dataset += DELIMITER + "\n"

    dataset += DATASET({v: k for k, v in DATASETS.items()}["MATERIAL"])

    mattype = material["type"]
    mattype = {v[0]: k for k, v in MATERIAL_TYPE.items()}[mattype]
    matname = material["name"]

    available_properties = list([MATERIAL_PROPERTY[p][0] for p in MATERIAL_TYPE[mattype][3]])
    property_count = list([MATERIAL_PROPERTY[p][2] for p in MATERIAL_TYPE[mattype][3]])

    supplied_properties = list([k for k in material.keys() if k not in ("type", "name")])

    props = []
    numprops = 0
    for i in range(len(available_properties)):
        if available_properties[i] not in supplied_properties:
            if property_count[i] == 1:
                props.append(0.)
            else:
                props.extend([0.] * property_count[i])
            numprops += 1

        else:
            props.append(material[available_properties[i]])
            numprops += 1
            supplied_properties.remove(available_properties[i])
            if len(supplied_properties) == 0:
                break

    dataset += INTEGER(matID)
    dataset += INTEGER(mattype)
    dataset += INTEGER(numprops)
    dataset += "\n"

    dataset += CHAR(matname[:40]) + "\n"

    for i in range(len(props)):
        dataset += SINGLE(props[i])
        if (i + 1) % 6 == 0:
            dataset += "\n" + INTEGER(0) + "\n"

    if dataset[-1] != "\n":
        dataset += "\n" + INTEGER(0) + "\n"

    dataset += DELIMITER + "\n"

    return dataset



def _write_material2(matID: int, material: dict, comment: str = None) -> str:
    print(f"[+] Writing Material {matID:n}.")
    if comment is not None:
        dataset = "\n".join([CHAR(c) for c in comment.split("\n")]) + "\n"
    else:
        dataset = ""

    dataset += DELIMITER + "\n"

    dataset += DATASET({v: k for k, v in DATASETS.items()}["MATERIAL"])

    mattype = material["type"]
    mattype = {v[0]: k for k, v in MATERIAL_TYPE.items()}[mattype]
    matname = material["name"]

    available_properties = list([MATERIAL_PROPERTY[p][0] for p in MATERIAL_TYPE[mattype][3]])
    property_count = list([MATERIAL_PROPERTY[p][2] for p in MATERIAL_TYPE[mattype][3]])

    properties = {k: v for k, v in material.items() if k not in ("type", "name")}
    props = list(properties.keys())

    for i in range(len(available_properties)):
        if available_properties[i] not in props:
            if property_count[i] == 1:
                properties.setdefault(available_properties[i], [0.])
            else:
                properties.setdefault(available_properties[i], [0.] * property_count[i])
        else:
            if type(properties[available_properties[i]]) is not list:
                properties[available_properties[i]] = [properties[available_properties[i]]]
            props.remove(available_properties[i])
            if len(props) == 0:
                break

    props = list([p for p in available_properties if p in properties])

    numprops = len(properties.keys())

    dataset += INTEGER(matID)
    dataset += INTEGER(mattype)
    dataset += INTEGER(numprops)
    dataset += "\n"

    dataset += CHAR(matname[:40]) + "\n"

    for pname in props:
        pvals = properties[pname]
        nvals = len(pvals)
        for i in range(len(pvals)):
            dataset += SINGLE(pvals[i])
            if (i + 1) % 6 == 0:
                if i < 6:
                    dataset += "\n" + INTEGER(nvals) + "  " + CHAR(pname)[:68] + "\n"
                    nvals -= 6
                else:
                    dataset += "\n" + INTEGER(6) + "  " + CHAR(pname)[:68] + "\n"
        if dataset[-1] != "\n":
            dataset += "\n" + INTEGER(nvals) + "  " + CHAR(pname)[:68] + "\n"

    dataset += DELIMITER + "\n"

    return dataset



def _write_materials(materials: dict) -> str:
    datasets = ""

    for matID, material in materials.items():
        datasets += _write_material(matID, material)

    return datasets



def _read_nodal_data(unv) -> dict:
    # print(f"[+] Reading Nodal Results.")
    global _LINENUMBER

    lastPos, line = _read_line(unv) # DATASET number
    dataset = int(line.strip())

    err = ""
    result = {}
    results = {}

    # description
    description = []
    for i in range(5):
        lastPos, line = _read_line(unv,
            msg_eof = f"[-] File ended before closing the dataset {dataset:n}\n")
        if line == DELIMITER:  # end of DATASET
            err = f"[-] Dataset {dataset:n} ended before finishing Nodal results\n"
            raise ReadError(err)

        if line not in ("", "NONE"):
            description += line

    result["description"] = "\n".join(description)

    # data definition - record 6
    lastPos, line = _read_and_parse_line(unv, "6I10", dataset, 6)
    if line == DELIMITER:  # end of DATASET
        err = f"[-] Dataset {dataset:n} ended before finishing Nodal results\n"
        raise ReadError(err)

    model_type, analysis_type, data_char, spec_data_type, data_type, nvals = line
    result["model type"] = MODEL_TYPE[model_type]
    result["analysis type"] = ANALYSIS_TYPE[analysis_type]
    result["data characteristic"] = DATA_CHARACTERISTIC[data_char]
    result["specific data type"] = SPECIFIC_DATA_TYPE[spec_data_type]
    dtype = DATA_TYPE[data_type]
    result["data type"] = dtype
    result["values per node"] = nvals

    if dtype == "Integer":
        fmt = "I10"
        datalen = 10
        maxlen = 8

    elif dtype == "Real":
        fmt = "E13.5"
        datalen = 13
        maxlen = 6

    elif dtype == "Real Double":
        fmt = "D25.5"
        datalen = 25
        maxlen = 3

    elif dtype == "Complex":
        nvals *= 2
        fmt = "E13.5"
        datalen = 13
        maxlen = 6

    elif dtype == "Complex Double":
        nvals *= 2
        fmt = "D25.5"
        datalen = 25
        maxlen = 3

    # Analysis Specifics - record 7
    lastPos, line = _read_and_parse_line(unv, "8I10", dataset, 7, True)
    if line == DELIMITER:  # end of DATASET
        err = f"[-] Dataset {dataset:n} ended before finishing Nodal results\n"
        raise ReadError(err)

    if len(line) >= 3:
        numints = line[0]
        numfloats = line[1]
    else:
        err = f"[-] Dataset {dataset:n} has wrong length of record 7 on line {_LINENUMBER:n}:\n"
        unv.seek(lastPos)
        line = unv.readline().strip("\n")
        err += f"    >>> {line:s}\n"
        _read_dataset_till_end(unv, dataset, err)

    # TODO:
    # what if numints > 6 ?
    if len(line) == 2 + numints:
        ints = line[2:]
    else:
        err = f"[-] Dataset {dataset:n} has wrong length of record 7 on line {_LINENUMBER:n}:\n"
        unv.seek(lastPos)
        line = unv.readline().strip("\n")
        err += f"    >>> {line:s}\n"
        _read_dataset_till_end(unv, dataset, err)

    # TODO:
    # what if numfloats > 6 ?
    # Analysis Specifics - record 8
    lastPos, line = _read_and_parse_line(unv, f"{numfloats:n}E13.5", dataset, 8)
    if line == DELIMITER:  # end of DATASET
        err = f"[-] Dataset {dataset:n} ended before finishing Nodal results\n"
        raise ReadError(err)

    floats = line

    if analysis_type == 0: # analysis type = Unknown
        id = ints[0]
        step = 1
        result["id"] = id

        result["value"] = float[0]

    elif analysis_type == 1: # analysis type = Static
        id = ints[0]
        step = 0
        result["lcase"] = id

        result["value"] = floats[0]

    elif analysis_type == 2: # analysis type = normal mode
        id = ints[0]
        step = ints[1]
        result["id"] = id
        result["mode"] = step

        result["frequency"] = floats[0]
        result["modal mass"] = floats[1]
        result["modal viscous damping ratio"] = floats[2]
        result["modal hysteric damping ratio"] = floats[3]

    elif analysis_type == 3: # analysis type = complex eigenvalue
        id = ints[0]
        step = ints[1]
        result["lcase"] = id
        result["mode"] = step

        result["eigenvalue"] = complex(floats[0], floats[1])
        result["modal A"] = complex(floats[2], floats[3])
        result["modal B"] = complex(float[4], floats[5])

    elif analysis_type == 4: # analysis type = transient
        id = ints[0]
        step = ints[1]
        result["lcase"] = id
        result["time step"] = step

        result["time"] = floats[0]

    elif analysis_type == 5: # analysis type = frequency response
        id = ints[0]
        step = ints[1]
        result["lcase"] = id
        result["frequency step"] = step

        result["frequency"] = floats[0]

    elif analysis_type == 6: # analysis type = buckling
        id = ints[0]
        step = 1
        result["lcase"] = id

        result["eigenvalue"] = floats[0]

    else:
        result["num of intvals"] = numints
        result["num of realvals"] = numfloats
        result["specific integer parameters"] = ints
        id = ints[0]
        step = 1

        result["specific real parameters"] = floats

    print(f"[+] Reading {result['analysis type']:s} Analysis Nodal Data ID {id:n}.")

    # nodal results record 9 and 10
    nodes = {}
    numlines = math.ceil(nvals / maxlen)
    while True:
        lastPos, line = _read_and_parse_line(unv, "1I10", dataset, 9)
        if line == DELIMITER:  # end of DATASET
            break

        nid = line[0]

        vals = []
        for i in range(numlines):
            lastPos, line = _read_and_parse_line(unv,
                f"{min(maxlen, nvals-len(vals)):n}" + fmt, dataset, 10)
            if line == DELIMITER:  # end of DATASET
                err = f"[-] Dataset {dataset:n} ended before finishing Nodal results\n"
                raise ReadError(err)

            vals += line

        if dtype in ("Integer", "Real", "Real Double"):
            nodes[nid] = vals
        else:
            nodes[nid] = [complex(vals[i], vals[i+1]) for i in range(0, len(vals), 2)]

    result.setdefault("values", nodes)

    results = dict()
    results.setdefault(id, dict())
    results[id].setdefault(step, result)

    return results



def _write_nodal_data(lcID: int, lsID: int, result: dict, comment: str = None) -> str:
    print(f"[+] Writing LCase {lcID:n} LStep {lsID:n} Nodal Data.")
    if comment is not None:
        dataset = "\n".join([CHAR(c) for c in comment.split("\n")]) + "\n"
    else:
        dataset = ""

    dataset += DELIMITER + "\n"

    dataset += DATASET({v: k for k, v in DATASETS.items()}["NODAL"])

    description = ["NONE"] * 5 if "description" not in result.keys() else result["description"].split("\n")
    if len(description) < 5:
        description += ["NONE"] * (5 - len(description))

    dataset += "\n".join([CHAR(d) for d in description]) + "\n"

    if "model type" in result.keys():
        mt = {v: k for k, v in MODEL_TYPE.items()}[result["model type"]]
    else:
        mt = 0

    if "analysis type" in result.keys():
        at = {v: k for k, v in ANALYSIS_TYPE.items()}[result["analysis type"]]
    else:
        at = 0

    if "data characteristic" in result.keys():
        dc = {v: k for k, v in DATA_CHARACTERISTIC.items()}[result["data characteristic"]]
    else:
        dc = 0

    if "specific data type" in result.keys():
        sd = {v: k for k, v in SPECIFIC_DATA_TYPE.items()}[result["specific data type"]]
    else:
        sd = 0

    if "data type" in result.keys():
        dt = {v: k for k, v in DATA_TYPE.items()}[result["data type"]]
    else:
        first_val = result["values"][list(result["values"].keys())[0]][0]
        if type(first_val) in (int, np.int):
            dt = DATA_TYPE["Integer"]        # 1
        elif type(first_val) in (float, np.float, np.float16, np.float32, np.float64):
            dt = DATA_TYPE["Real"]           # 2
        elif type(first_val) is np.float128:
            dt = DATA_TYPE["Real Double"]    # 2
        elif type(first_val) in (complex, np.complex, np.complex128):
            dt = DATA_TYPE["Complex"]        # 5
        elif type(first_val) is np.complex256:
            dt = DATA_TYPE["Complex Double"] # 6
        else: # default
            dt = DATA_TYPE["Real"]           # 2

    if dt == 1:
        FMT = INTEGER
        numvals = 8
    elif dt == 2:
        FMT = SINGLE
        numvals = 6
    elif dt == 3:
        FMT = DOUBLE
        numvals = 3
    elif dt == 5:
        FMT = SINGLE
        numvals = 6
    elif dt == 6:
        FMT = DOUBLE
        numvals = 3
    else:
        FMT = SINGLE
        numvals = 6

    if "values per node" in result.keys():
        nvals = result["values per node"]
    else:
        nvals = len(result["values"][list(result["values"].keys())[0]])

    dataset += INTEGER(mt)
    dataset += INTEGER(at)
    dataset += INTEGER(dc)
    dataset += INTEGER(sd)
    dataset += INTEGER(dt)
    dataset += INTEGER(nvals)
    dataset += "\n"

    if at == 0: # Unknown
        dataset += INTEGER(1)
        dataset += INTEGER(1)
        dataset += INTEGER(lcID) + "\n"
        dataset += SINGLE(0.0) + "\n"

    elif at == 1: # Static
        dataset += INTEGER(1)
        dataset += INTEGER(1)
        dataset += INTEGER(lcID) + "\n"
        dataset += SINGLE(0.0) + "\n"

    elif at == 2: # Normal Mode
        dataset += INTEGER(2)
        dataset += INTEGER(4)
        dataset += INTEGER(lcID)
        mode = result["mode"]
        if mode != lsID:
            print(f"[i] renumbering LCASE {lcID:n} mode number from {mode:n} to {lsID:n}.")
            mode = lsID
        dataset += INTEGER(mode) + "\n"

        mm = result["modal mass"] if "modal mass" in result.keys() else 0.0
        mvdr = result["modal viscous damping ratio"] if "modal viscous damping ratio" in result.keys() else 0.0
        mhdr = result["modal hysteric damping ratio"] if "modal hysteric damping ratio" in result.keys() else 0.0

        dataset += SINGLE(result["frequency"])
        dataset += SINGLE(mm)
        dataset += SINGLE(mvdr)
        dataset += SINGLE(mhdr)
        dataset += "\n"

    elif at == 3: # Complex Eigenvalue
        dataset += INTEGER(2)
        dataset += INTEGER(6)
        dataset += INTEGER(lcID)
        mode = result["mode"]
        if mode != lsID:
            print(f"[i] renumbering LCASE {lcID:n} mode number from {mode:n} to {lsID:n}.")
            mode = lsID
        dataset += INTEGER(mode) + "\n"

        ev = result["eigenvalue"]
        mA = result["modal A"] if "modal A" in result.keys() else complex(0., 0.)
        mB = result["modal B"] if "modal B" in result.keys() else complex(0., 0.)

        if type(ev) not in (complex, np.complex, np.complex128, np.complex256):
            ev = complex(ev, 0.)

        if type(mA) not in (complex, np.complex, np.complex128, np.complex256):
            mA = complex(mA, 0.)

        if type(mB) not in (complex, np.complex, np.complex128, np.complex256):
            mB = complex(mB, 0.)

        dataset += SINGLE(ev.real) + SINGLE(ev.imag)
        dataset += SINGLE(mA.real) + SINGLE(mA.imag)
        dataset += SINGLE(mB.real) + SINGLE(mB.imag)
        dataset += "\n"

    elif at == 4: # Transient
        dataset += INTEGER(2)
        dataset += INTEGER(1)
        dataset += INTEGER(lcID)
        tstep = result["time step"]
        if tstep != lsID:
            print(f"[i] renumbering LCASE {lcID:n} mode number from {tstep:n} to {lsID:n}.")
            tstep = lsID
        dataset += INTEGER(tstep) + "\n"

        dataset += SINGLE(result["time"]) + "\n"

    elif at == 5: # Frequency Response
        dataset += INTEGER(2)
        dataset += INTEGER(1)
        dataset += INTEGER(lcID)
        fstep = result["frequency step"]
        if fstep != lsID:
            print(f"[i] renumbering LCASE {lcID:n} mode number from {fstep:n} to {lsID:n}.")
            fstep = lsID
        dataset += INTEGER(fstep) + "\n"

        dataset += SINGLE(result["frequency"]) + "\n"

    elif at == 6: # Buckling
        dataset += INTEGER(1)
        dataset += INTEGER(1)
        dataset += INTEGER(lcID) + "\n"

        dataset += SINGLE(result["eigenvalue"]) + "\n"

    else: # General
        intvals = result["num of intvals"]
        floatvals = result["num of realvals"]
        dataset += INTEGER(intvals)
        dataset += INTEGER(floatvals)

        intvals = result["specific intger parameters"]
        floatvals = result["specific real parameters"]
        i = 1
        for iv in intvals:
            i += 1
            dataset += INTEGER(iv)
            if (i + 1) % 8 == 0:
                dataset += "\n"
        if dataset[-1] != "\n":
            dataset += "\n"

        for i, fv in enumerate(floatvals):
            dataset += SINGLE(fv)
            if (i + 1) % 8 == 0:
                dataset += "\n"
        if dataset[-1] != "\n":
            dataset += "\n"

    for nid, values in result["values"].items():
        dataset += INTEGER(nid) + "\n"
        if DATA_TYPE[dt] in ("Complex", "Complex Double"):
            vals = []
            for val in values:
                vals.append(FMT(val.real))
                vals.append(FMT(val.imag))
            values = vals
        else:
            values = [FMT(val) for val in values]

        for i, val in enumerate(values):
            dataset += val
            if (i + 1) % numvals == 0:
                dataset += "\n"

        if dataset[-1] != "\n":
            dataset += "\n"

    dataset += DELIMITER + "\n"
    return dataset


# TODO:
# rewrite for simplified results model
def _write_data(nresults: dict = None, eresults: dict = None) -> str:
    datasets = ""
    if nresults is not None and len(nresults.keys()):
        for lcID in nresults.keys():
            for lsID in nresults[lcID].keys():
                datasets += _write_nodal_data(lcID, lsID, nresults[lcID][lsID])

    if eresults is not None and len(eresults.keys()):
        for lcID in eresults.keys():
            for lsID in eresults[lcID].keys():
                datasets += _write_elemental_data(lcID, lsID, eresults[lcID][lsID])

    return datasets



def _read_elemental_data(unv) -> dict:
    pass



def _read_dataset(unv) -> (dict, dict, dict, dict):
    global _LINENUMBER

    lastPos, line = _read_line(unv)
    dataset_num = int(line.strip())

    if dataset_num in DATASETS.keys():
        dataset = DATASETS[dataset_num]
    else:
        dataset = "UNKNOWN"

    nodes = None
    elements = None
    csys = None
    materials = None
    nresults = None
    eresults = None

    unv.seek(lastPos)
    _LINENUMBER -= 1

    if dataset == "NODE":
        nodes = _read_nodes_single(unv)

    elif dataset == "NODE2P":
        nodes = _read_nodes_double(unv)

    elif dataset == "ELEMENT":
        elements = _read_elements(unv)

    elif dataset == "CSYS":
        csys = _read_csys(unv)

    elif dataset == "MATERIAL":
        materials = _read_material(unv)

    elif dataset == "NODAL":
        nresults = _read_nodal_data(unv)

    elif dataset == "ELEMENTAL":
        eresults = _read_elemental_data(unv)

    else:
        _read_dataset_till_end(unv, dataset_num)

    return nodes, elements, csys, materials, nresults, eresults



def read(filename: str) -> (dict, dict, dict, dict, dict):
    global _LINENUMBER

    print(f"[+] Reading {filename:s}:")

    if not os.path.isfile(filename):
        raise ValueError(f"[-] File {filename:s} does not exist.")
        sys.exit()

    unv = open(filename, "rt")
    _LINENUMBER = 0

    nodes = {}
    elements = {}
    csys = {}
    materials = {}
    nresults = {}
    eresults = {}

    lineNo = 0
    err = False
    while True:
        try:
            lastPos, line = _read_line(unv)
            if line is None: # EOF
                break
            elif line == DELIMITER:
                _n, _e, _cs, _m, _nr, _er = _read_dataset(unv)
                for d, _d in zip([nodes, elements, csys, materials, nresults, eresults],
                                 [_n, _e, _cs, _m, _nr, _er]):
                    if _d is not None:
                        d = _update_dict_of_dicts(d, _d)

        except ReadError as re:
            message = "\n    ".join(str(re).split("\n"))
            print(f"[-] Read Error:\n{message:s}")
            err = True

    unv.close()

    if err:
        raise ReadError(f"[-] Errors found in file: {filename:s}")

    return nodes, elements, csys, materials, nresults, eresults



def write(filename: str, nodes: dict, elements: dict=None, csys: dict=None,
          materials: dict=None, nresults: dict=None, eresults: dict=None,
          precision="double"):
    print(f"[i] Writing {os.path.realpath(filename):s}")

    if os.path.isdir(os.path.dirname(os.path.realpath(filename))):

        datasets = ""

        if nodes is not None and len(nodes.keys()) != 0:
            datasets += _write_nodes(nodes, precision)

        if elements is not None and len(elements.keys()) != 0:
            datasets += _write_elements(elements)

        if csys is not None and len(csys.keys()) != 0:
            datasets += _write_csys(csys)

        if materials is not None and len(materials.keys()) != 0:
            datasets += _write_materials(materials)

        if nresults is not None or eresults is not None:
            datasets += _write_data(nresults, eresults)

        with open(filename, "w") as unv:
            unv.write(datasets)
    else:
        raise ValueError(f"[-] Directory for output ({os.path.dirname(os.path.realpath(filename)):s}) does not exist.")



if __name__ == "__main__":
    unv_file = "./res/test_hex_double.unv"
    nodes, elements, csys, materials, nresults, eresults = read(unv_file)


    material_in = {"type": "ISOTROPIC",
                   "name": "steel",
                   "E": 210000.,
                   "NU": 0.3,
                   "DEN": 7.85E-9,
                   "A": 1.2E-6,
                   "G": 80769.,
                   "TREF": 20.,
                   "YS": 210.,
                   "MU": 0.2,
                   "DMPCOF": 0.005,
                   "REFTMP": 20.}
    materials = {1: material_in}

    unv_file_2 = "./res/test_hex_double_material.unv"

    write(unv_file_2, nodes, elements, csys, materials,
          nresults, eresults)

    nodes, elements, csys, materials, nresults, eresults = read(unv_file_2)

    material_out = materials[1]

    for key, value in material_in.items():
        assert value == material_out[key]


