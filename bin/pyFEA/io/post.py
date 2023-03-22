#!/opt/python385_std/bin/python3

import sys
import os


def _XOR(a: bool, b: bool):
    # return (bool(a) and not bool(b)) or (not bool(a) and bool(b))
    return bool(a) ^ bool(b)


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


class POST:
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
            return line

        while True:
            _last_pos, _line = cls._read_line_skip_empty(f)
            if not _line.startswith('&'):
                f.seek(_last_pos)
                break
            line += _line[1:]

        return last_pos, line


    @classmethod
    def _strip(cls, line):
        line = line.strip().upper()
        while '  ' in line:
            line = line.replace('  ', ' ')
        return line


    @classmethod
    def _update_dict(cls, results, node, dir, displacements):
        for f in displacements.keys():
            if f not in results.keys():
                results.setdefault(f, {})

            if node not in results[f].keys():
                results[f].setdefault(node, [.0, .0, .0])

            results[f][node][dir] = displacements[f][node][dir]
        return results


    @classmethod
    def _command(cls, post):
        last_line = cls._strip(post.readline())
        while True:
            last_position = post.tell()
            new_line = cls._strip(post.readline())
            if new_line.startswith('!'):
                continue
            elif new_line.startswith('&'):
                last_line += ' ' + new_line[1:]
            else:
                post.seek(last_position)
                break
        return cls._strip(last_line)

    @staticmethod
    def _parse_command(command):
        clist = command.split()  # command split to a list
        cdict = {}               # dict with command arguments

        # indexes of all '=' characters
        idx = [i for i in range(len(clist)) if clist[i] == '=']

        if len(idx) > 0:
            cdict['COMMAND'] = ' '.join(clist[:idx[0]-1])
        else:
            cdict['COMMAND'] = ' '.join(clist[:])

        for i in range(len(idx)):
            if i == len(idx) - 1:
                next_i = len(clist)
            else:
                next_i = idx[i+1] - 1
            cdict[clist[idx[i]-1]] = ' '.join(clist[idx[i]+1:next_i])

        return cdict


    @classmethod
    def _data_xydata(cls, post, params_map: dict, displacements: dict=None):
        dir = {'U': 0, 'V': 1, 'W': 2, 'PHIU': 3, 'PHIV': 4, 'PHIW': 5}
        node = 0
        frequency = 0.0

        if displacements is None:
            displacements = {}

        ncols = params_map['NCOL']
        node = params_map['CURVE'].strip("'").split(',')
        dof = dir[node[1]]
        node = int(node[0].lstrip('N'))

        while True:
            last_pos, line = cls._readline(post)
            if not line:
                break
            elif line.startswith('$'):
                post.seek(last_pos)
                break
            else:
                r = [float(x) for x in line.split(' ')[1:]]
                frequency = r[0]
                if frequency not in displacements.keys():
                    displacements.setdefault(frequency, {})
                if node not in displacements[frequency].keys():
                    displacements[frequency].setdefault(node, [0.0, 0.0, 0.0])
                displacements[frequency][node][dof] = r[1]

        return displacements


    @classmethod
    def __data_disp(cls, post, params_map: dict):
        keys = None
        displacements = {}
        frequency = 0.0
        datatype = 'REAL'
        dattype = datatype

        while True:
            last_position = post.tell()
            line = cls._command(post)
            if line.startswith('$DATA'):
                keys = cls._parse_command(line)
                frequency = float(keys['FREQUENCY'])
                if 'DATTYPE' in keys.keys():
                    dattype = keys['DATTYPE']
                else:
                    dattype = datatype
                displacements.setdefault(frequency, {})
            elif line.startswith('$'):
                post.seek(last_position)
                break
            else:
                if dattype == 'COMPLEX':
                    line = line.replace(', ', ',')
                    r = line.split(' ')
                    node = int(r[0])
                    d = []
                    for rr in r[1:4]:
                        r_r = float(rr.split(',')[0])   # real part
                        r_i = float(rr.split(',')[1])   # imaginary part
                        d.append((r_r ** 2 + r_i ** 2) ** 0.5) # geometric sum
                        # d.append(r_r)                   # only the amplitude is needed
                    displacements[frequency][node] = d
                else:
                    r = line.split(' ')
                    node = int(r[0])
                    displacements[frequency][node] = [float(rr) for rr in r[1:4]]

        return displacements


    @classmethod
    def _data_disp(cls, post, params_map: dict, displacements: dict=None):
        if displacements is None:
            displacements = {}
        frequency = float(params_map['FREQUENCY']) if 'FREQUENCY' in params_map.keys() else 0.0
        dattype = params_map['DATTYPE'] if 'DATTYPE' in params_map.keys() else 'REAL'

        if frequency not in displacements.keys():
            displacements.setdefault(frequency, dict())

        while True:
            last_pos, line = cls._readline(post)
            if not line:
                break
            elif line.startswith('$'):
                post.seek(last_pos)
                break
            else:
                line = line.replace(', ', ',')
                line = line.split(' ')
                nid = int(line[0])
                vals = line[1:]
                if dattype == 'COMPLEX':
                    vals = [v.split(',') for v in vals]
                    vals = [abs(complex(float(r), float(i))) for r, i in vals] # geometric sum
                else:
                    vals = [float(v) for v in vals]
                displacements[frequency][nid] = vals

        return displacements


    @classmethod
    def __data_acce(cls, post, params_map: dict):
        return cls._data_disp(post)


    @classmethod
    def _data_acce(cls, post, params_map: dict, displacements: dict=None):
        return cls._data_disp(post, params_map, displacements)


    @classmethod
    def _read(cls, filename):
        print(f'[i] Reading *.post file: {filename:s}.')
        post = open(filename, 'r')
        displacements = {}

        while True:
            last_position = post.tell()
            line = cls._strip(post.readline())
            if line.startswith('!'):
                continue
            elif line.startswith('$FIN'):
                break
            elif line.startswith('$DATA'):
                post.seek(last_position)
                keys = cls._parse_command(cls._command(post))
                post.seek(last_position)
                if keys['COMMAND'] == '$DATA X_XYDATA':
                    displacements = cls._data_xydata(post)
                elif keys['COMMAND'] == '$DATA DISP':
                    displacements = cls._data_disp(post)
                    break
                elif keys['COMMAND'] == '$DATA ACCE':
                    displacements = cls._data_acce(post)
                    break
                else:
                    raise NotImplementedError(f"Reading command {keys['COMMAND']:s} not Implemented, "
                                               "use either $DATA X_XYDATA, $DATA DISP or $DATA ACCE.")

        post.close()

        nodes = {}
        edges = []
        trias = []

        return nodes, edges, displacements, trias

    @classmethod
    def read(cls, filename, verbose: bool=False):
        print(f'[i] Reading *.post file: {filename:s}.')
        displacements = []
        component = 'KOMPO_1'
        results = 'MODAL_ABGLEICH'
        analysis = 'MODAL ASSURANCE CRITERION'
        settings = []
        # displacements.append(dict())
        # settings.append(dict())

        with open(filename, 'r') as post:
            while True:
                last_pos, line = cls._readline(post)
                if not line:                    # EOF
                    break
                elif not line.startswith('$'):  # skip non-keyword lines
                    continue
                params_map = _get_params_map(line)
                if '$FIN' in params_map.keys(): # EOF
                    break
                elif '$ENTER' in params_map.keys() and 'COMPONENT' in params_map.keys():
                    component = params_map['NAME']
                elif '$RESULTS' in params_map.keys():
                    if results != params_map['NAME']:
                        displacements.append(dict())
                        settings.append(dict())
                    results = params_map['NAME']
                elif '$PARAMETER' in params_map.keys() and 'ANALYSIS' in params_map.keys():
                    analysis = params_map['ANALYSIS']
                elif '$DATA' in params_map.keys():
                    settings[-1]['COMPONENT'] = component
                    settings[-1]['RESULTS NAME'] = results
                    settings[-1]['ANALYSIS'] = analysis
                    for key, value in params_map.items():
                        if key != 'CURVE':
                            settings[-1][key] = value
                    if verbose:
                        print(f'  [i] Reading block: {line:s}.')
                    if 'X_XYDATA' in params_map.keys():
                        input = 'X_XYDATA'
                        displacements[-1] = cls._data_xydata(post, params_map, displacements[-1])
                    elif 'DISP' in params_map.keys():
                        input = 'DISP'
                        # post.seek(last_pos)
                        # displacements.update(cls._data_disp(post, params_map))
                        displacements[-1] = cls._data_disp(post, params_map, displacements[-1])
                    elif 'ACCE' in params_map.keys():
                        input = 'ACCE'
                        # post.seek(last_pos)
                        # displacements.update(cls._data_acce(post, params_map))
                        displacements[-1] = cls._data_acce(post, params_map, displacements[-1])
                    else:
                        keyword = list(params_map.keys())[1].upper()
                        raise NotImplementedError(f'Reading command $DATA {keyword:s} not Implemented, '
                                                   'use either $DATA X_XYDATA, $DATA DISP or $DATA ACCE.')
                else:
                    pass
                    # print(f'[i] Skipping block: {line:s}.')

        nodes = {}
        edges = []
        trias = []

        if len(displacements) == 1:
            return nodes, edges, displacements[0], trias
        else:
            return nodes, edges, displacements, trias


    @staticmethod
    def __formatFloat(number):
        if number < 0:
            return f'{number:1.6E}'
        else:
            return f' {number:1.6E}'

    @classmethod
    def write(cls, filename, data, sameID=False, output='DISP', settings={'COMPONENT': 'KOMPO_1',
                                                                          'RESULTS NAME': 'MODAL_ABGLEICH',
                                                                          'ANALYSIS': 'MODAL ASSURANCE CRITERION',
                                                                          'TYPE': 'DISP',
                                                                          'ABSCISSAE': 'FREQUENCY',
                                                                          'NCOL': 2,
                                                                          'DATTYPE': '',
                                                                          'CURVE': ['U', 'V', 'W']}):
        print(f'[i] Writing *.post file: {filename:s}.')
        dirpath = os.path.dirname(os.path.realpath(filename))
        if not os.path.exists(dirpath):
            try:
                os.makedirs(dirpath)
            except OSError as exc: # guard against race condition
                pass

        output_types = ['X_XYDATA', 'DISP']

        if sameID:
             idShift = 0
        else:
             idShift = 10000000

        if 'COMPONENT' in settings.keys():
            COMPONENT = settings['COMPONENT']
        else:
            COMPONENT = 'KOMPO_1'

        if 'RESULTS NAME' in settings.keys():
            RESULTS_NAME = settings['RESULTS NAME']
        else:
            RESULTS_NAME = 'MODAL_ABGLEICH'

        if 'ANALYSIS' in settings.keys():
            ANALYSIS = settings['ANALYSIS']
        else:
            ANALYSIS = 'MODAL ASSURANCE CRITERION'
        if ' ' in ANALYSIS:
            ANALYSIS = "'" + ANALYSIS + "'"

        if 'TYPE' in settings.keys():
            TYPE = settings['TYPE']
        else:
            TYPE = 'DISP'

        if 'ABSCISSAE' in settings.keys():
            ABSCISSAE = settings['ABSCISSAE']
        else:
            ABSCISSAE = 'FREQUENCY'

        if 'NCOL' in settings.keys():
            NCOL = settings['NCOL']
        else:
            NCOL = 2

        if 'DATTYPE' in settings.keys():
            DATTYPE = settings['DATTYPE']
        else:
            DATTYPE = None

        if 'CURVE' in settings.keys():
            CURVE = settings['CURVE']
        else:
            CURVE = ['U', 'V', 'W']

        freqs = sorted(list(data.keys()))
        if all([type(nid) is int for nid in data[freqs[0]].keys()]):
        nodes = sorted(list(data[freqs[0]].keys()))
        else:
            nodes = sorted(list([nid for nid in data[freqs[0]].keys() if type(nid) is int]))
            nodes.extend(sorted(list([nid for nid in data[freqs[0]].keys() if type(nid) is str])))

        lines = []
        lines.append('$ENTER COMPONENT NAME = KOMPO_1 DOFTYPE = DISP')
        lines.append(f'$RESULTS NAME = {RESULTS_NAME:s}')
        lines.append('$PARAMETER')
        lines.append(f'& ANALYSIS = {ANALYSIS:s}')

        if output == 'X_XYDATA':
            for n in nodes:
                for i, dir in enumerate(CURVE):
                    lines.append('$DATA X_XYDATA')
                    lines.append(f'& TYPE = {TYPE:s}')
                    lines.append(f'& ABSCISSAE = {ABSCISSAE:s}')
                    lines.append(f'& NCOL = {NCOL:n}')
                    if DATTYPE is not None:
                        lines.append(f'& DATTYPE = {DATTYPE:s}')
                    if type(n) is int:
                        lines.append('& CURVE = \'N{0:n},{1:s}\''.format(int(n) + idShift, dir))
                    else:
                        lines.append('& CURVE = \'{0:s},{1:s}\''.format(str(n), dir))
                    for j, f in enumerate(freqs):
                        if len(CURVE) == 1:
                            lines.append('{0:10n} {1:13.6E} {2:13.6E}'.format(j+1, f, data[f][n]))
                        else:
                            lines.append('{0:10n} {1:13.6E} {2:13.6E}'.format(j+1, f, data[f][n][i]))

        elif output == 'DISP':
            for j, f in enumerate(freqs):
                lines.append('$DATA DISP')
                lines.append('& MODE = {0:n}'.format(j+1))
                lines.append('& FREQUENCY = {0:s}'.format(cls.__formatFloat(f)))
                lines.append('& NCOL = 6')
                for n in nodes:
                    lines.append('{0:10n} {1:s} {2:s} {3:s} {4:s}'.format(int(n) + idShift,
                                                                          cls.__formatFloat(data[f][n][0]),
                                                                          cls.__formatFloat(data[f][n][1]),
                                                                          cls.__formatFloat(data[f][n][2]),
                                                                          cls.__formatFloat(0.)))
                    lines.append('   &       {0:s} {1:s}'.format(cls.__formatFloat(0.),
                                                                 cls.__formatFloat(0.)))

        else:
            raise NotImplementedError(f'Output type "{output:s}" is not supported for "*.post" file, ' +
                                      f'select one of ({", ".join(output_types):s}).')
        lines.append('$END RESULTS')
        lines.append('$EXIT COMPONENT')
        lines.append('$FIN')

        with open(filename, 'w') as f:
            f.write('\n'.join(lines) + '\n')



# test
if __name__ == '__main__':
    post_file = './res/rm6358A_lnk_stds_2917_fre07_acc_Y_acce_GSYS.post'
    _, _, disp, _ = POST.read(post_file)
    print(len(disp))
    print(disp[list(disp.keys())[0]])

    # post_file = './res/rm6358A_lnk_stds_2917_eig_EIG_XY.post'
    # _, _, disp, _ = POST.read(post_file)
    # print(disp)

    # import pdb

    # post_file = './res/zuea10A_cos_stds_0103_spectral03_pcb1.post'

    # pdb.set_trace()
    # with open(post_file, 'rt') as f:
    #     while True:
    #         line = POST._readline(f)
    #         # print(line)
    #         if not line:
    #             print('EOF')
    #             break
    #         elif line.startswith('$'):
    #             print(_get_params_map(line))
    #         else:
    #             print(line)

    # exit()

    # string = '$ENTER COMPONENT NAME = KOMPO_1 DOFTYPE = DISP TEMP PRES POTE MATH'
    # params_map = _get_params_map(string)
    # print(params_map)

    # exit()

    # post_file = '../res/rm6358A_lnk_stds_2917_fre07_acc_Y.post'
    # disp = POST.read(post_file)
    # print(disp[list(disp.keys())[0]])

    # post_file = '../res/rm6358A_lnk_stds_2917_fre07_acc_Y_acce_GSYS.post'
    # disp = POST.read(post_file)
    # print(disp[list(disp.keys())[0]])

    # pdb.set_trace()

    # command = "$DATA X_MODFAC TYPE = ACCE FREQUENCY = 1.00000E+01 NCOL = 5 DATATYPE = COMPLEX CURVE = 'N901,u'"
    # print(POST._parse_command(command))

    # command = "$ENTER COMPONENT NAME = KOMPO_1 DOFTYPE = DISP PRES MATH"
    # print(POST._parse_command(command))

    # command = "$EXIT COMPONENT"
    # print(POST._parse_command(command))

    # command = "$COOR"
    # print(POST._parse_command(command))


