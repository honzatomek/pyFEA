#!/opt/python385/bin/python3.8

import sys
import os
import subprocess
import linecache
import io

from _helpers.Console import Info, Warn, Error

_VERBOSE = True

GREP_COOR_DATO = '$COOR'

DAT_KEYWORDS = {'$COOR': 0, '$ELEMENT': 3,
                '$NSET': 0, '$ESET': 0}
NODES = {'$COOR': 3}
ELEMENTS = {'MASS3': 1,
            'MASS6': 1,
            'BECOS': 2,
            'X2STIFF3': 2,
            'X2STIFF6': 2,
            'TRIA3': 3,
            'TRIA6': 6,
            'QUAD4': 4,
            'TET4' : 4,
            'TET10': 10,
            'PENTA6': 6,
            'HEXE8': 8}


def _get_params_map(string: str) -> str:
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


class DAT:
    @classmethod
    def _stripline(cls, line: str) -> str:
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
        if line.startswith('$COMMENT'):
            line = ''
        return line.strip(' ')


    @classmethod
    def _read_line_skip_empty(cls, f) -> (int, str):
        """
        Read next line from file, skip empty and commented lines
        """
        last_pos = f.tell()
        line = f.readline()
        if not line:                           # EOF
            return last_pos, line
        line = cls._stripline(line)
        if line in ('', ' '):
            last_pos, line = cls._read_line_skip_empty(f)
        return last_pos, line


    @classmethod
    def _readline(cls, f) -> (int, str):
        last_pos, line = cls._read_line_skip_empty(f)
        if not line:                          # EOF
            return last_pos, line

        while True:
            _last_pos, _line = cls._read_line_skip_empty(f)
            if not _line.startswith('&'):
                f.seek(_last_pos)
                break
            line += _line[1:]

        return last_pos, line


    @staticmethod
    def _strip(string):
        # strip whitespaces
        retval = string.strip()
        # strip comments
        if '!' in retval:
            retval = retval.split('!')[0]
        # remove multiple whitespaces
        while '  ' in retval:
            retval = retval.replace('  ', ' ')
        return retval


    @staticmethod
    def _join_lines_split(block):
        dataBlock = block.getvalue()
        if '\n&' in dataBlock:
            dataBlock = dataBlock.replace('\n&', ' ')

        return dataBlock.split('\n')


    @classmethod
    def _extract_coordinates(cls, block, ntype):
        points = {}
        for line in cls._join_lines_split(block):
            if line.startswith('$'):
                continue
            parts = line.split()
            if len(parts) == NODES[ntype] + 1:
                pointID = int(parts[0])
                coordinates = [float(c) for c in parts[1:]]
                points[pointID] = coordinates

        return points


    @classmethod
    def _extract_elements(cls, block, etype):
        elements = {}
        for line in cls._join_lines_split(block):
            if line.startswith('$'):
                continue
            parts = line.split()
            if len(parts) == ELEMENTS[etype] + 1:
                elementID = int(parts[0])
                nodes = [int(n) for n in parts[1:]]
                elements[elementID] = nodes
        return elements


    @classmethod
    def _extract_set(cls, block, settype: str):
        idset = {}
        setname = settype
        for line in cls._join_lines_split(block):
            if line.startswith('$'):
                setname = line.split(' ')[3]
                idset.setdefault(setname, [])
                continue
            parts = line.split()
            idset[setname].extend([int(i) for i in parts])
        idset[setname] = sorted(list(set(idset[setname])))
        return idset


    @classmethod
    def _read_nodes(cls, dat, params_map, nodes=None, nsets=None) -> (dict, dict):
        if _VERBOSE: print(f'        Nodes     :', end='')
        if nodes is None:
            nodes = {}

        if nsets is None:
            nsets = {}

        if 'NSET' in params_map.keys():
            nset = params_map['NSET']
            if nset not in nsets.keys():
                nsets.setdefault(nset, [])
        else:
            nset = None

        while True:
            last_pos, line = cls._readline(dat)
            if not line:                    # EOF
                break
            elif line.startswith('$'):  # end of nodes block
                dat.seek(last_pos)       # revert back one line
                break                   # end reading
            coor = line.split(' ')
            nodes[int(coor[0])] = [float(c) for c in coor[1:]]
            if nset is not None:
                nsets[nset].append(int(coor[0]))

        if _VERBOSE: print(f'{len(nodes.keys()):12n}')
        return nodes, nsets


    @classmethod
    def _write_nodes(cls, dat, nodes: dict, nset: str = None, offset: int = 2):
        offset = ' ' * offset
        lines = offset + '$COOR' + ('\n' if nset is None else f' NSET = {nset.upper():s}\n')

    @classmethod
    def _read_elements(cls, dat, params_map, elements=None, esets=None) -> (dict, dict):
        if elements is None:
            elements = {}

        etype = params_map['TYPE']
        if _VERBOSE: print(f'        {etype:<10s}:', end='')

        if etype not in elements.keys():
            elements.setdefault(etype, {})
        if esets is None:
            esets = {}
        if 'ESET' in params_map.keys():
            eset = params_map['ESET']
            if eset not in esets.keys():
                esets.setdefault(eset, [])
        else:
            eset = None

        while True:
            last_pos, line = cls._readline(dat)
            if not line:                    # EOF
                break
            elif line.startswith('$'):  # end of elemets block
                dat.seek(last_pos)       # revert back one line
                break                   # end reading
            parts = line.split(' ')
            elements[etype][int(parts[0])] = [int(nid) for nid in parts[1:]]
            if eset is not None:
                esets[eset].append(int(parts[0]))

        if _VERBOSE: print(f'{len(elements[etype].keys()):12n}')
        return elements, esets


    @classmethod
    def _read_set(cls, dat, params_map, sets) -> dict:
        if sets is None:
            sets = {}

        set = params_map['NAME']
        # print(f'        Set {set:s}')

        if set not in sets.keys():
            sets.setdefault(set, [])

        while True:
            last_pos, line = cls._readline(dat)
            if not line:                    # EOF
                break
            elif line.startswith('$COMMENT'):
                continue
            elif line.startswith('$'):  # end of elemets block
                dat.seek(last_pos)       # revert back one line
                break                   # end reading
            parts = line.split(' ')
            for p in parts:
                if p.isnumeric():
                    sets[set].append(int(p))
                else:
                    sets[set].append(str(p))

        return sets



    @classmethod
    def _read_structure(cls, dat, params_map, nodes=None, elements=None, nsets=None,
                        esets=None) -> dict:
        if _VERBOSE: print(f'    $STRUCTURE')

        if type(dat) is str:
            dat = open(dat, 'rt')
            was_file_passed = False
        else:
            was_file_passed = True

        if nodes is None:
            nodes = {}
        if nsets is None:
            nsets = {}
        if elements is None:
            elements = {}
        if esets is None:
            esets = {}

        while True:
            last_pos, line = cls._readline(dat)
            if not line:                    # EOF
                break
            elif not line.startswith('$'):  # skip non-keyword lines
                continue

            params_map = _get_params_map(line)

            if '$FIN' in params_map.keys(): # EOF
                break
            elif '$END' in params_map.keys() and 'STRUCTURE' in params_map.keys():
                break

            if '$COOR' in params_map.keys():
                nodes, nsets = cls._read_nodes(dat, params_map, nodes, nsets)

            elif '$ELEMENT' in params_map.keys():
                elements, esets = cls._read_elements(dat, params_map, elements, esets)

            elif '$NSET' in params_map.keys():
                nsets = cls._read_set(dat, params_map, nsets)

            elif '$ESET' in params_map.keys():
                esets = cls._read_set(dat, params_map, esets)
            else:
                pass

        if not was_file_passed:
            dat.close()

        # process sets
        is_int = False
        loop_cnt = 0
        while not is_int:
            loop_cnt += 1
            is_int = True
            for nset, nids in nsets.items():
                for nid in nids:
                    if type(nid) is str:
                        is_int = False
                        if nid in nsets.keys():
                            nsets[nset].pop(nid)
                            nsets[nset].extend(nsets[nid])
                            if nid in not_found: not_found.pop(nid)
                        else:
                            not_found.append(nid)
                            not_found = list(set(not_found))
            if loop_cnt > 100:
                Warn('Exceeded Loop Count for $NSET name to ID resulution' +
                     f'(names not found: {", ".join(not_found):s}')
                break


        is_int = False
        loop_cnt = 0
        not_found = []
        while not is_int:
            loop_cnt += 1
            is_int = True
            for eset, eids in esets.items():
                for eid in eids:
                    if type(eid) is str:
                        is_int = False
                        if eid in esets.keys():
                            esets[eset].pop(eid)
                            esets[eset].extend(esets[eid])
                            if eid in not_found: not_found.pop(eid)
                        else:
                            not_found.append(eid)
                            not_found = list(set(not_found))
            if loop_cnt > 100:
                Warn('Exceeded Loop Count for $ESET name to ID resulution ' +
                     f'(names not found: {", ".join(not_found):s}')
                break

        setnames = list(nsets.keys())
        if _VERBOSE: print('        Node Sets : ', end='')
        for i in range(0, len(setnames), 4):
            if _VERBOSE: print(', '.join(setnames[i:min(i+4, len(setnames))]) + ',')
            if i + 4 <= len(setnames):
                if _VERBOSE: print('                    ', end='')

        setnames = list(esets.keys())
        if _VERBOSE: print('        Elem. Sets: ', end='')
        for i in range(0, len(setnames), 4):
            if _VERBOSE: print(', '.join(setnames[i:min(i+4, len(setnames))]) + ',')
            if i + 4 <= len(setnames):
                if _VERBOSE: print('                    ', end='')

        return nodes, elements, nsets, esets


    @classmethod
    def _read_geodat(cls, dat, system, params_map, geodat) -> dict:
        if geodat is None:
            geodat = {}

        if system not in geodat.keys():
            geodat.setdefault(system, {})

        gtype = list(params_map.keys())[1].upper()

        cnt = 0
        while True:
            last_pos, line = cls._readline(dat)
            if not line:                    # EOF
                break
            elif line.startswith('$'):  # skip non-keyword lines
                dat.seek(last_pos)
                break

            gname = line.split(' ')[0].upper()
            cont = params_map['CONT']

            if type(cont) is str or len(cont) == 1:
                parts = line.split(' ')
                geodat[system][gname] = {'type': gtype,
                                         cont: parts[1]}

            else:
                geodat[system][gname] = {'type': gtype}
                parts = line.upper().lstrip(gname).strip(' ').split(':')
                for i, part in enumerate(parts):
                    geodat[system][gname][cont[i]] = [float(p) for p in part.split(' ') if p != '']
            cnt += 1

        if _VERBOSE: print(f'       Geodat {gtype:10s} :{cnt:12n}')
        return geodat


    @classmethod
    def _read_elprop(cls, dat, system, params_map, elprop=None) -> dict:
        if elprop is None:
            elprop = {}

        if system not in elprop.keys():
            elprop.setdefault(system, {})

        cnt = 0
        while True:
            last_pos, line = cls._readline(dat)
            if not line:                    # EOF
                break
            elif line.startswith('$COMMENT'):
                continue
            elif line.startswith('$'):  # skip non-keyword lines
                break

            params_map = _get_params_map(line)

            if '$FIN' in params_map.keys(): # EOF
                break
            elif '$END' in params_map.keys() and 'SYSTEM' in params_map.keys(): # end of $STRUCTURE BLOCK
                break

            element = list(params_map.keys())[0]

            if element.isnumeric():
                element = int(element)

            if element not in elprop[system].keys():
                elprop[system][element] = {}

            if 'MATERIAL' in params_map.keys():
                elprop[system][element]['material'] = params_map['MATERIAL'].upper()
            else:
                if 'material' not in elprop[system][element].keys():
                    elprop[system][element]['material'] = None

            if 'GEODAT' in params_map.keys():
                elprop[system][element]['geodat'] = params_map['GEODAT'].upper()
            else:
                if 'geodat' not in elprop[system][element].keys():
                    elprop[system][element]['geodat'] = None
            cnt += 1

        if _VERBOSE: print(f'       Elprop            :{cnt:12n}')
        return elprop


    @classmethod
    def _read_system(cls, dat, params_map, geodat=None, elprop=None) -> (dict, dict):
        if geodat is None:
            geodat = {}
        if elprop is None:
            elprop = {}

        # TODO:
        if 'NAME' in params_map.keys():
            system = params_map['NAME']
        else:
            system = 'SYSVAR'

        if _VERBOSE: print(f'    $SYSTEM NAME = {system:s}')

        if system not in geodat.keys():
            geodat.setdefault(system, {})

        if system not in elprop.keys():
            elprop.setdefault(system, {})

        while True:
            last_pos, line = cls._readline(dat)
            if not line:                    # EOF
                break
            elif not line.startswith('$'):  # skip non-keyword lines
                continue

            params_map = _get_params_map(line)

            if '$FIN' in params_map.keys(): # EOF
                break
            elif '$END' in params_map.keys() and 'SYSTEM' in params_map.keys(): # end of $STRUCTURE BLOCK
                break

            if '$GEODAT' in params_map.keys():
                geodat = cls._read_geodat(dat, system, params_map, geodat)

            elif '$ELPROP' in params_map.keys():
                elprop = cls._read_elprop(dat, system, params_map, elprop)

            else:
                pass

        return geodat, elprop



    @classmethod
    def _readall(cls, filename: str):
        if _VERBOSE: print(f'[+] Reading *.dat file: {filename:s}.')

        nodes = {}
        elements = {}
        nsets = {}
        esets = {}
        geodat = {}
        elprop = {}

        with open(filename, 'rt') as dat:
            while True:
                last_pos, line = cls._readline(dat)
                if not line:                    # EOF
                    break
                elif not line.startswith('$'):  # skip non-keyword lines
                    continue
                params_map = _get_params_map(line)
                # print(params_map.keys())
                if '$FIN' in params_map.keys(): # EOF
                    break

                if '$STRUCTURE' in params_map.keys():
                    nodes, elements, nsets, esets = cls._read_structure(dat, params_map,
                                                                        nodes, elements,
                                                                        nsets, esets)

                elif '$SYSTEM' in params_map.keys():
                    geodat, elprop = cls._read_system(dat, params_map,
                                                      geodat, elprop)

                else:
                    pass

        return nodes, elements, nsets, esets, geodat, elprop


    @classmethod
    def readall(cls, filename):
        if _VERBOSE: print(f'Reading *.dat file: {filename:s}.')
        fi = open(filename, 'rt')

        fileBlockDict = {}

        for line in fi:
            ln = cls._strip(line)
            # strip whitespaces and comments
            parts = ln.split()

            # skip empty lines
            if len(parts) == 0:
                continue

            # read data
            if parts[0] in DAT_KEYWORDS.keys():
                blockName = parts[DAT_KEYWORDS[parts[0]]].upper()
                dataBlock = io.StringIO()
                if blockName not in fileBlockDict.keys():
                    fileBlockDict.setdefault(blockName, [])
                fileBlockDict[blockName].append(dataBlock)
                # continue
            # blackhole stream for all other commands
            elif parts[0].startswith('$'):
                dataBlock = io.StringIO()
                continue

            dataBlock.write(cls._strip(line) + '\n')

        fi.close()

        # read nodes
        nodes = {}
        for k in list(NODES.keys()):
            if k in fileBlockDict.keys():
                for block in fileBlockDict[k]:
                    nodes.update(cls._extract_coordinates(block, k))

        # read elements
        elements = {}
        for k in list(ELEMENTS.keys()):
            elements.setdefault(k, {})
            if k in fileBlockDict.keys():
                for block in fileBlockDict[k]:
                    elements[k].update(cls._extract_elements(block, k))

        # read node sets
        nsets = {}
        k = '$NSET'
        if k in fileBlockDict.keys():
            for block in fileBlockDict[k]:
                nsets.update(cls._extract_set(block, k))

        # read element sets
        esets = {}
        k = '$ESET'
        if k in fileBlockDict.keys():
            for block in fileBlockDict[k]:
                esets.update(cls._extract_set(block, k))

        return nodes, elements, nsets, esets


    @staticmethod
    def read(filename):
        '''
        Reads only node information.
        '''
        if _VERBOSE: print(f'Reading *.dat file: {filename:s}.')
        counter = 1
        nodes = {}

        # get total number of lines in dato
        linestotal = str(subprocess.check_output("wc -l {0}".format(filename), shell=True).decode('ascii'))
        linestotal = int(linestotal.split(' ')[0])

        # get line number of $COOR tag in dato file
        output = str(subprocess.check_output("grep -n '{0:s}' {1:s}".format(GREP_COOR_DATO, filename),
                                             shell=True).decode('ascii'))
        output = output.split('\n')[:-1]
        if len(output) == 0:
            if _VERBOSE: print(f'Error: No {GREP_COOR_DATO:s} tag in {filename:s} file.')
            sys.exit()
        else:
            coor_lines = [int(o.split(':')[0].strip()) for o in output]

        # read all nodes in dato into memory and store their coordinates in nodes dictionary
        nodes = {}

        for coor_line in coor_lines:
            counter = 0

            while coor_line + counter < linestotal:
                counter += 1
                line = linecache.getline(filename, coor_line + counter).strip('\n')
                if line == '' or line == ' ' or line == '!':
                    break

                if True:
                    records = line.split()
                    nodes.setdefault(int(records[0]), [float(records[1]), float(records[2]), float(records[3])])

                    if counter % 100 == 0:
                        if _VERBOSE: print('    {0: 8d} nodes read into memory.. (Current node ID: {1:n})'.format(len(nodes),
                                                                                                   int(records[0])),
                              end='\r')
                        sys.stdout.write('\033[K')

        edges = []
        data = {}
        trias = []

        return nodes, edges, data, trias


    @staticmethod
    def write(filename, nodes, edges=None, trias=None, sameID=False, translate=False):
        if _VERBOSE: print(f'Writing *.dat file: {filename:s}.')
        dirpath = os.path.dirname(os.path.realpath(filename))
        if not os.path.exists(dirpath):
            try:
                os.makedirs(dirpath)
            except OSError as exc: # guard against race condition
                pass

        lines = []
        lines.append('$ENTER COMPONENT NAME = KOMPO_1 DOFTYPE = DISP')
        lines.append('  $STRUCTURE')

        nList = list(nodes)
        nList.sort()
        pxList = [x[0] for x in list(nodes.values())]
        maxX = max(pxList)
        minX = min(pxList)
        xSize = maxX - minX

        if sameID:
            shift = 0
        else:
            shift = 10000000

        lines.append('    $COOR')
        for i, nid in enumerate(nList):
            NID = shift + int(nid)
            if translate:
                X1 = nodes[nid][0] + 1.5 * xSize
            else:
                X1 = nodes[nid][0]
            X2 = nodes[nid][1]
            X3 = nodes[nid][2]
            lines.append(f'      {NID:8n} {X1:15.5f} {X2:15.5f} {X3:15.5f}')

        if edges:
            lines.append('    $ELEMENT TYPE = BECOS')
            for i, e in enumerate(edges):
                EID = i + 1 + shift
                NID1 = shift + int(e[0])
                NID2 = shift + int(e[1])
                lines.append(f'      {EID:8n} {NID1:11n} {NID2:11n}')

        if trias:
            lines.append('    $ELEMENT TYPE = TRIA3')
            for i, e in enumerate(trias):
                EID = i + 1 + shift
                NID1 = shift + int(e[0])
                NID2 = shift + int(e[1])
                NID3 = shift + int(e[2])
                lines.append(f'      {EID:8n} {NID1:11n} {NID2:11n} {NID3:11n}')

        lines.append('  $END STRUCTURE')
        lines.append('$EXIT COMPONENT')

        with open(filename, 'w') as dat:
            dat.write('\n'.join(lines))

if __name__ == '__main__':
    filename = './res/rm6358A_lnk_stds_2917_eig.dato'
    nodes, elements, nsets, esets, geodat, elprop = DAT._readall(filename)
    # nodes, elements, nsets, esets = DAT._read_structure(filename, None)
    print([f'{k}: {len(v.keys())}' for k, v in elements.items()])

