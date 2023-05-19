
import os
import sys
import io
import math
import logging

import numpy as np

import pdb

try:
    import pyFEA.model as model

except ImportError as e:
    __PATH__ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

    sys.path.append(__PATH__)

    import pyFEA.model as model


logger = logging.getLogger(__name__)

DELIMITER = f"{-1:>6n}"
DATASET = lambda x: f"{x:>6n}\n"

UNV_TO_PYFEA = {  15: "NODE",
                 781: "NODE",
                2411: "NODE",}
PYFEA_TO_UNV = {v: k for k, v in UNV_TO_PYFEA.items()}


class UNVReadError(Exception):
    pass


class UNV:
    _DELIMITER = -1

    @staticmethod
    def format(format: str) -> list:
        format = format.split(",")
        fields = []
        totlength = 0
        for f in format:
            if "I" in f:
                f = f.split("I")
                dtype = int
            elif "E" in f:
                f = f.split("E")
                dtype = float
            elif "D" in f:
                f = f.split("D")
                dtype = np.float128
            count = int(f[0]) if f[0] != "" else 1
            length = int(f[1].split(".")[0])

            for i in range(count):
                fields.append([length, dtype])
            totlength += length * count

        return fields, totlength


    def __init__(self, filename: str, mode: str = "read"):
        self._line = 0
        self._lastpos = None # np.empty((1, 2), dtype=int)
        self._last5 = []
        self._file = None

        self._filename = filename

        self._component = None
        self._materials = None


    def __del__(self):
        if self._file is not None:
            self._file.close()


    def close(self):
        if self.file is not None:
            self._file.close()
            self._file = None


    @property
    def component(self) -> model.Component:
        return self._component


    @property
    def materials(self) -> model.Materials:
        return self._materials


    @property
    def file(self):
        return self._file


    @property
    def filename(self) -> str:
        return self._filename


    def lastpos(self, num: int = None) -> int:
        if num is None:
            return self._lastpos[-1,0]
        else:
            return self._lastpos[num,0]


    def lineNo(self, num: int = None) -> int:
        if self._lastpos is None:
            return 0
        elif num is None:
            return self._lastpos[-1,1]
        else:
            return self._lastpos[num,1]


    def revertline(self, num: int = None):
        if num is None:
            self.file.seek(self.lastpos())
            self._lastpos = self._lastpos[:-1,:]
        else:
            self.file.seek(self.lastpos(-num))
            self._lastpos = self._lastpos[:-num,:]


    @property
    def last5(self) -> str:
        msg = ""
        lineNo = self.lineNo() - len(self._last5)
        for i, line in enumerate(self._last5):
            if (i + 1) == len(self._last5):
                msg += f"{lineNo+i:>9n}: >>> {line:s}\n"
            else:
                msg += f"{lineNo+i:>9n}:     {line:s}\n"
        return msg


    def readline(self, format: str = None, shorter: bool = False,
                 msg_eof: str = None, msg_eod: str = None) -> (str, int):
        if self.file is None:
            self._file = open(self.filename, "r")

        lineNo = self.lineNo()
        if self._lastpos is None:
            self._lastpos = np.array([[self.file.tell(), lineNo + 1]], dtype=int)
        else:
            self._lastpos = np.append(self._lastpos, [[self.file.tell(), lineNo + 1]], axis = 0)

        line = self.file.readline()

        if line is None: # EOF
            if msg_eof is None:
                return None, 0
            else:
                raise UNVReadError(msg_eof)
        elif line.strip() == "":
            return None, 0

        line = line.strip("\n")
        if len(self._last5) == 5:
            self._last5.pop(0)
        self._last5.append(line)

        if line == DELIMITER:
            if msg_eod is not None:
                raise UNVReadError(msg_eod + "\n" + self.last5)
            else:
                return line, len(line)

        if format is None:
            return line, len(line)

        else:
            format, totlength = self.format(format)
            if not shorter and len(line) != totlength:
                msg = f"[-] {type(self).__name__:s} wrong length of record:\n"
                raise UNVReadError(msg + "\n" + self.last5)

            fields = []
            start = 0
            for f in format:
                if start + f[0] > len(line):
                    break
                val = line[start:start+f[0]]
                if f[1].__name__ == "float128":
                    val = val.replace("D", "E")
                fields.append(f[1](val))
                start += f[0]

            return fields, start


    def read_nodes(self) -> model.Nodes:
        dataset, _ = self.readline()
        dataset = int(dataset)

        nodes = self.component.get_structure().nodes

        if dataset == 15:     # single precision
            while True:
                line, _ = self.readline("4I10,3E13.5",
                    msg_eof = f"File ended before closing dataset {dataset:n}.")
                if line == DELIMITER:
                    break

                nid =    line[0]
                defsys = line[1]
                outsys = line[2]
                color =  line[3]
                coors =  np.array(line[4:], dtype=float)

                node = model.Node(id=nid, coors=coors, defsys=defsys, outsys=outsys)

                nodes.add(node)

        elif dataset in (781, 2411):  # double precision
            while True:
                line, _ = self.readline("4I10",
                    msg_eof = f"File ended before closing dataset {dataset:n}.")
                if line == DELIMITER:
                    break

                nid =    line[0]
                defsys = line[1]
                outsys = line[2]
                color =  line[3]

                line, _ = self.readline("3D25.16",
                    msg_eof = f"File ended before closing dataset {dataset:n}.",
                    msg_eod = f"Dataset {dataset:n} missing record 2 for node {nid:n}.")

                coors =  np.array(line[:3], dtype=np.float128)

                node = model.Node(id=nid, coors=coors, defsys=defsys, outsys=outsys)

                nodes.add(node)

        return nodes



    def read_dataset(self):
        dataset, _ = self.readline()
        dataset = int(dataset)

        print(f"[i] Reading dataset {dataset:n}.")

        if self.component is None:
            self._component = model.Component("KOMPO_1")

        # skip unknown datasets
        if dataset not in UNV_TO_PYFEA.keys():
            while True:
                line, _ = self.readline()
                if line is None:
                    break
                elif line == DELIMITER:
                    break

        else:
            dataset = UNV_TO_PYFEA[dataset]
            self.revertline()

        if dataset == "NODE":
            self.component.get_structure().nodes.add(self.read_nodes())



    def read(self) -> (model.Component, model.Materials):
        while True:
            try:
                line, _ = self.readline()
                print(line)
                if line is None:
                    break

                elif line == DELIMITER:
                    self.read_dataset()

            except UNVReadError as re:
                self.close()
                raise re

        self.close()

        return self.component, self.materials



if __name__ == "__main__":
    unv = UNV("./res/test_hex_double.unv")

    # pdb.set_trace()
    compo, mat = unv.read()

    print(str(compo.get_structure().nodes))


