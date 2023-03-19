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
              55: "NODAL"}

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
    print(f"[+] Writing nodes single precision).")
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
    print(f"[+] Writing nodes double precision).")
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
    print(f"[+] Writing elements.")
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



# TODO:
# change to the simplified results {load case ID: {load step ID: {results}}}
def _write_nodal_data(lcID: int, lsID: int, result: dict, comment: str = None) -> str:
    print(f"[+] Writing LCASE {lcID:n} LSTEP {lsID:n}.")
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
            FMT = INTEGER
            numvals = 8
        elif type(first_val) in (float, np.float, np.float16, np.float32, np.float64):
            dt = DATA_TYPE["Real"]           # 2
            FMT = SINGLE
            numvals = 6
        elif type(first_val) is np.float128:
            dt = DATA_TYPE["Real Double"]    # 2
            FMT = DOUBLE
            numvals = 3
        elif type(first_val) in (complex, np.complex, np.complex128):
            dt = DATA_TYPE["Complex"]        # 5
            FMT = SINGLE
            numvals = 6
        elif type(first_val) is np.complex256:
            dt = DATA_TYPE["Complex Double"] # 6
            FMT = DOUBLE
            numvals = 3
        else: # default
            dt = DATA_TYPE["Real"]           # 2
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
    if nresults is not None:
        for lcID in nresults.keys():
            for lsID in nresults.keys():
                datasets += _write_nodal_data(lcID, lsID, nresults[lcID][lsID])

    if eresults is not None:
        for lcID in eresults.keys():
            for lsID in eresults.keys():
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

    elif dataset == "NODAL":
        nresults = _read_nodal_data(unv)

    elif dataset == "ELEMENTAL":
        eresults = _read_elemental_data(unv)

    else:
        _read_dataset_till_end(unv, dataset_num)

    return nodes, elements, nresults, eresults



def read(filename: str) -> (dict, dict, dict, dict):
    global _LINENUMBER

    print(f"[+] Reading {filename:s}:")

    if not os.path.isfile(filename):
        raise ValueError(f"[-] File {filename:s} does not exist.")
        sys.exit()

    unv = open(filename, "rt")
    _LINENUMBER = 0

    nodes = {}
    elements = {}
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
                _n, _e, _nr, _er = _read_dataset(unv)
                for d, _d in zip([nodes, elements, nresults, eresults], [_n, _e, _nr, _er]):
                    if _d is not None:
                        d = _update_dict_of_dicts(d, _d)

        except ReadError as re:
            message = "\n    ".join(str(re).split("\n"))
            print(f"[-] Read Error:\n{message:s}")
            err = True

    unv.close()

    if err:
        raise ReadError(f"[-] Errors found in file: {filename:s}")

    return nodes, elements, nresults, eresults



def write(filename: str, nodes: dict, elements: dict=None, nresults: dict=None,
          eresults: dict=None, precision="double"):
    print(f"[i] Writing {os.path.realpath(filename):s}")

    if os.path.isdir(os.path.dirname(os.path.realpath(filename))):

        datasets = ""

        if nodes is not None:
            datasets += _write_nodes(nodes, precision)

        if elements is not None:
            datasets += _write_elements(elements)

        if nresults is not None or eresults is not None:
            datasets += _write_data(nresults, eresults)

        with open(filename, "w") as unv:
            unv.write(datasets)
    else:
        raise ValueError(f"[-] Directory for output ({os.path.dirname(os.path.realpath(filename)):s}) does not exist.")


if __name__ == "__main__":
    unv_file = "./res/test_hex_double.unv"
    nodes, elements, nresults, eresults = read(unv_file)
    print_dict(nresults)

