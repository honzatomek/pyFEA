import os
import sys
import io

DAT_KEYWORDS = {'$COOR': 0, '$ELEMENT': 3}
BEGIN_COMMAND = '$'
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


def strip(text: str) -> str:
    # strip comment
    if '!' in text:
        text = text.split('!')[0]
    # strip whitespace
    while '  ' in text:
        text = text.replace('  ', ' ')
    # strip beggining and ending whitespace
    return text.strip()

def find_all(a_str, sub):
    start = 0
    while True:
        start = a_str.find(sub, start)
        if start == -1: return
        yield start
        start += len(sub) # use start += 1 to find overlapping matches

def find_continuation(text):
    lineidx = list()
    for i, line in enumerate(text.split('\n')):
        if line.startswith('&'):
            lineidx.append(i)
    return lineidx

def join_continuation_split(block, lineNos: list) -> list:
    dataBlock = block.getvalue()
    # adjust line numbers
    cidx = find_continuation(dataBlock)
    if len(cidx) > 0:
        for idx in reversed(cidx):
            lineNos.pop(idx)
    # concatenate continuation lines
    if '\n&' in dataBlock:
        dataBlock = dataBlock.replace('\n&', ' ')
    # delete multiples of spaces
    if '  ' in dataBlock:
        dataBlock = dataBlock.replace('  ', ' ')
    return dataBlock.split('\n')[:-1], lineNos

def extract_nodes(block, lineNos: list) -> dict:
    print('[+] Reading nodes.')
    nodes = dict()
    lines, lineNos = join_continuation_split(block, lineNos)
    err = ''
    for i, line in enumerate(lines):
        fields = line.split(' ')
        if len(fields) == 4:
            nodes[int(fields[0])] = [float(coor) for coor in fields[1:]]
        else:
            err += f'[-] Wrong number of entries for Node on line {lineNos[i]:n}: {line:s}.\n'
    # if err != '':
    #     print(err[:-1])
    #     sys.exit()
    return nodes, err

def extract_elements(block, etype: str, lineNos: list) -> dict:
    print(f'[+] Reading {etype:s} elements.')
    elements = dict()
    lines, lineNos = join_continuation_split(block, lineNos)
    err = ''
    for i, line in enumerate(lines):
        fields = line.split(' ')
        if len(fields) == ELEMENTS[etype] + 1:
            elements[int(fields[0])] = [int(nid) for nid in fields[1:]]
        else:
            err += f'[-] Wrong number of nodes for element type {etype:s} on line {lineNos[i]:n}: {line:s}.\n'
    # if err != '':
    #     print(err[:-1])
    #     sys.exit()
    return elements, err

def read(filename: str):
    print(f'[i] Reading {filename:s}:')

    if not os.path.isfile(filename):
        raise ValueError(f'[-] File {filename:s} does not exist.')
        sys.exit()

    fi = open(filename, 'rt')
    fileBlockDict = {}
    fileBlockLines = {}

    for lineNo, line in enumerate(fi):
        # strip whitespaces and comments
        ln = strip(line)
        # split line into separate fields
        fields = ln.split()
        # skip empty lines
        if len(fields) == 0:
            continue
        # read data
        if fields[0].upper() in DAT_KEYWORDS.keys():
            blockName = fields[DAT_KEYWORDS[fields[0]]].upper()
            dataBlock = io.StringIO()
            dataLines = []
            if blockName not in fileBlockDict.keys():
                fileBlockDict.setdefault(blockName, [])
                fileBlockLines.setdefault(blockName, [])
            fileBlockDict[blockName].append(dataBlock)
            fileBlockLines[blockName].append(dataLines)
            continue
        # blackhole stream for all other commands
        elif fields[0].upper().startswith(BEGIN_COMMAND):
            dataBlock = io.StringIO()
            dataLines = []
            continue

        # write to datablock
        dataBlock.write(ln + '\n')
        dataLines.append(lineNo + 1)

    fi.close()

    err = ''
    # extract nodes
    nodes = {}
    for i, block in enumerate(fileBlockDict['$COOR']):
        nds, e = extract_nodes(block, fileBlockLines['$COOR'][i])
        if len(e) > 0:
            err += e
        nodes.update(nds)

    # extract elements
    elements = {}
    for etype in ELEMENTS.keys():
        if etype in fileBlockDict.keys():
            elements.setdefault(etype, dict())
            for i, block in enumerate(fileBlockDict[etype]):
                els, e = extract_elements(block, etype, fileBlockLines[etype][i])
                if len(e) > 0:
                    err += e
                elements[etype].update(els)

    # if error in read
    if len(err) > 0:
        print(err[:-1])
        sys.exit()

    return nodes, elements


if __name__ == '__main__':
    filename = '../res/rm6358A_lnk_stds_2917_eig_err.dato'
    nodes, elements = read(filename)

  
