import os
import sys
import io
import math

import pdb

DELIMITER = f"{-1:6n}"
DATASET = "{0:>6n}\n"

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

_LINENUMBER = 0


class ReadError(Exception):
    pass



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



def _write_nodes(nodes: dict, precision: str = "single", comment: str = None) -> str:
    print(f"[+] Writing nodes ({precision:s} precision).")
    if comment is not None:
        dataset = comment + "\n"
    else:
        dataset = ""
    dataset += DELIMITER + "\n"
    if precision == "double":
        dataset += DATASET.format({v: k for k, v in DATASETS.items()}["NODE2P"])
        # dataset += DATASET.format(781)
    else: # single precision
        dataset += DATASET.format({v: k for k, v in DATASETS.items()}["NODE"])
        # dataset += DATASET.format(15)
    defsys = 1
    outsys = 1
    color = 1
    for nid, coor in nodes.items():
        dataset += f"{nid:10n}{defsys:10n}{outsys:10n}{color:10n}"
        if precision == "double":
            dataset += f"\n{coor[0]:25.16E}{coor[1]:25.16E}{coor[2]:25.16E}\n".replace("E", "D")
        else:
            dataset += f"{coor[0]:13.5E}{coor[1]:13.5E}{coor[2]:13.5E}\n"
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

        elements[etype][eid] = {"material": mid,
                                "property": pid,
                                "nodes": nid}

    return elements



def _write_elements(elements: dict, comment: str = None):
    print(f"[+] Writing elements.")
    if comment is not None:
        dataset = comment + "\n"
    else:
        dataset = ""
    dataset += DELIMITER + "\n"

    dataset += DATASET.format({v: k for k, v in DATASETS.items()}["ELEMENT"])

    ELEMENT_KEYS = {v: k for k, v in ELEMENTS.items()}

    pid = 1
    mid = 1
    color = 1

    for etype, els in elements.items():
        for eid in els.keys():
            # TODO:
            if "property" in els[eid].keys():
                pid = els[eid]["property"]
            else:
                pid = 1
            if "material" in els[eid].keys():
                mid = els[eid]["material"]
            else:
                mid = 1
            nodes = els[eid]["nodes"]
            numnodes = len(nodes)
            dataset += f"{eid:10n}{ELEMENT_KEYS[etype]:10n}{pid:10n}{mid:10n}{color:10n}{numnodes:10n}\n"

            # beam elements
            if ELEMENT_KEYS[etype] in [21, 22, 24]:
                orinode = 0
                endAid = 0
                endBis = 0
                dataset += f"{orinode:10n}{endAid:10n}{endBid:10n}\n"

            # nodes
            for i in range(numnodes):
                dataset += f"{nodes[i]:10n}"
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

    if len(line) == 2 + numints:
        ints = line[2:]
    else:
        err = f"[-] Dataset {dataset:n} has wrong length of record 7 on line {_LINENUMBER:n}:\n"
        unv.seek(lastPos)
        line = unv.readline().strip("\n")
        err += f"    >>> {line:s}\n"
        _read_dataset_till_end(unv, dataset, err)

    # Analysis Specifics - record 8
    lastPos, line = _read_and_parse_line(unv, f"{numfloats:n}E13.5", dataset, 8)
    if line == DELIMITER:  # end of DATASET
        err = f"[-] Dataset {dataset:n} ended before finishing Nodal results\n"
        raise ReadError(err)

    floats = line

    if analysis_type == 0: # analysis type = Unknown
        result["id"] = ints[0]
        id = result["id"]
        step = 1

        result["value"] = float[0]

    elif analysis_type == 1: # analysis type = Static
        result["lcase"] = ints[0]
        id = result["lcase"]
        step = 0

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
        result["frequency step number"] = step

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
    # results.setdefault(result["model type"], dict())
    # results[result["model type"]].setdefault(result["analysis type"], dict())
    # results[result["model type"]][result["analysis type"]].setdefault("NODAL", dict())
    # results[result["model type"]][result["analysis type"]]["NODAL"][id] = nodes
    results.setdefault(id, dict())
    results[id].setdefault(step, result)

    return results



# TODO:
# change to the simplified results {load case ID: {load step ID: {results}}}
def _write_nodal_data(rid, result: dict, comment: str = None) -> str:
    """
    Returns a string with nodal data results
    In:
        int  rid    - result ID
        dict result - nodal result data with following keys:
                        - "id" result lcase/mode id
                        - "description" (max 5 lines of 80 chars each
                        - "model type" (Unknown, Structural, Heat Transfer, Fluid Flow)
                        - "analysis type"
                            - Unknown
                            - Static
                            - Normal Mode
                            - Complex eigenvalue first order
                            - Transient
                            - Frequency Response
                            - Buckling
                            - Complex eigenvalue second order
                        - "data characteristic"
                            - Unknown
                            - Scalar
                            - 3 DOF Global Translation Vector
                            - 6 DOF Global Translation & Rotation Vector
                            - Symmetric Global Tensor
                            - General Global Tensor
                        - "specific data type"
                            - Unknown
                            - General
                            - Stress
                            - Strain
                            - Element Force
                            - Temperature
                            - Heat Flux
                            - Strain Energy
                            - Displacement
                            - Reaction Force
                            - Kinetic Energy
                            - Velocity
                            - Acceleration
                            - Strain Energy Density
                            - Kinetic Energy Density
                            - Hydro-Static Pressure
                            - Heat Gradient
                            - Code Checking Value
                            - Coefficient Of Pressure
                        - "data type" (Real, Complex)
                        - "values per node" int

    Out:
        str
    """
    print(f"[+] Writing {result['analysis type']:s} Analysis Nodal results ID {rid:n}.")
    if comment is not None:
        dataset = comment + "\n"
    else:
        dataset = ""
    dataset += DELIMITER + "\n"

    dataset += DATASET.format({v: k for k, v in DATASETS.items()}["NODAL"])

    if "id" in result.keys():
        rid = result["id"]
    else:
        rid = rid

    if "description" in result.keys():
        dataset += "\n".join([d[:80] for d in result["description"].split("\n")[:5]]) + "\n"
    else:
        dataset += "\n".join(["NONE"] * 5) + "\n"

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
        if type(result["values"][list(result["values"].keys())[0]][0]) is float:
            dt = DATA_TYPE["Real"] # 2
            datalen = 1
        else:
            dt = DATA_TYPE["Complex"] # 5
            datalen = 2
    if "values per node" in result.keys():
        nvals = result["values per node"]
    else:
        nvals = len(result["values"][list(result["values"].keys())[0]]) * datalen

    dataset += f"{mt:10n}{at:10n}{dc:10n}{sd:10n}{dt:10n}{nvals:10n}\n"

    if at == 0: # Unknown
        dataset += f"{1:10n}{1:10n}{rid:10n}\n"
        dataset += f"{0.0:13.5E}\n"

    elif at == 1: # Static
        dataset += f"{1:10n}{1:10n}{result['lcase']:10n}\n"
        dataset += f"{0.0:13.5E}\n"

    elif at == 2: # Normal Mode
        dataset += f"{2:10n}{4:10n}{result['id']:10n}{result['mode']:10n}\n"
        dataset += f"{result['frequency']:13.5E}"
        if "modal mass" in result.keys():
            dataset += f"{result['modal mass']:13.5E}"
        else:
            dataset += f"{0.0:13.5E}"
        if "modal viscous damping ratio" in result.keys():
            dataset += f"{result['modal viscous damping ratio']:13.5E}"
        else:
            dataset += f"{0.0:13.5E}"
        if "modal hysteric damping ratio" in result.keys():
            dataset += f"{result['modal hysteric damping ratio']:13.5E}\n"
        else:
            dataset += f"{0.0:13.5E}\n"

    elif at == 3: # Complex Eigenvalue
        dataset += f"{2:10n}{6:10n}{result['lcase']:10n}{result['mode']:10n}\n"
        dataset += f"{result['eigenvalue'].real:13.5E}{result['eigenvalue'].imag:13.5E}"
        dataset += f"{result['modal A'].real:13.5E}{result['modal A'].imag:13.5E}"
        dataset += f"{result['modal B'].real:13.5E}{result['modal B'].imag:13.5E}\n"

    elif at == 4: # Transient
        dataset += f"{2:10n}{1:10n}{result['lcase']:10n}{result['time step']:10n}\n"
        dataset += f"{result['time']:13.5E}\n"

    elif at == 5: # Frequency Response
        dataset += f"{2:10n}{1:10n}{result['lcase']:10n}{result['frequency step number']:10n}\n"
        dataset += f"{result['frequency']:13.5E}\n"

    elif at == 6: # Buckling
        dataset += f"{2:10n}{1:10n}{result['lcase']:10n}\n"
        dataset += f"{result['eigenvalue']:13.5E}\n"

    else: # General
        dataset += f"{result['num of intvals']:10n}{result['num of realvals']:10n}"
        for i in range(result["num of intvals"]):
            dataset += f"{result['specific integer parameters'][i]:10n}"
        dataset += "\n"
        for i in range(result["num of realvals"]):
            dataset += f"{result['specific real parameters'][i]:13.5E}"
        dataset += "\n"

    for nid, values in result["values"].items():
        dataset += f"{nid:10n}\n"
        if DATA_TYPE[dt] == "Real":
            values = [f"{val:13.5E}" for val in values]
        else:
            values = [f"{val.real:13.5E}{val.imag:13.5E}" for val in values]
        dataset += "".join(values) + "\n"

    dataset += DELIMITER + "\n"
    return dataset


# TODO:
# rewrite for simplified results model
def _write_data(results: dict) -> str:
    datasets = ""
    for atype, rtype in results.items():
        for rtype in rtype.keys():
            if rtype == "NODAL":
                for rid in results[atype]["NODAL"].keys():
                    datasets += _write_nodal_data(rid, results[atype]["NODAL"][rid])

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



def write(filename: str, nodes: dict, elements: dict=None, results: dict=None,
          precision="double"):
    print(f"[i] Writing {os.path.realpath(filename):s}")
    if os.path.isdir(os.path.dirname(os.path.realpath(filename))):
        datasets = ""
        if nodes is not None:
            datasets += _write_nodes(nodes, precision)
        if elements is not None:
            datasets += _write_elements(elements)
        if results is not None:
            datasets += _write_data(results)

        with open(filename, "w") as unv:
            unv.write(datasets)
    else:
        raise ValueError(f"[-] Directory for output ({os.path.dirname(os.path.realpath(filename)):s}) does not exist.")


if __name__ == "__main__":
    unv_file = "./res/test_hex_double.unv"
    nodes, elements, nresults, eresults = read(unv_file)
    print(nresults)

