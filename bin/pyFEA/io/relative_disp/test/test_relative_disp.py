#!/usr/bin/python3

import os
import sys
import numpy as np

import pytest

SCRIPT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(SCRIPT_PATH)
from relative_disp import transform_displacements, read_k_file, read_permas_file, write_csv_displacements

BASE_FILE = "testdata/base_cube"
KFILES = ["testdata/cube_transl",
          "testdata/cube_skew",
          "testdata/cube_rotated",
          "testdata/cube_rotated2"]

CSYS_FORM = ["XY", "YX", "XZ", "ZX", "YZ", "ZY"]

RESULT = np.array([[0, 0, 0],
                   [1, 0, 0],
                   [2, 0, 0],
                   [0, 0, 0],
                   [0, 0, 0],
                   [3, 0, 0],
                   [4, 0, 0],
                   [0, 0, 0]], dtype=float)


@pytest.mark.parametrize("ext", [".k", ".dat1"])
@pytest.mark.parametrize("form", CSYS_FORM)
@pytest.mark.parametrize("deformed", KFILES)
def test_transform_displacements_XY(deformed, form, ext):
    basename = os.path.splitext(os.path.basename(deformed))[0]

    read_file = read_k_file if ext == ".k" else read_permas_file

    nodes_orig, _ = read_file(BASE_FILE + ext)
    nodes_defor, _ = read_file(deformed + ext)

    nids = list(sorted(list(nodes_orig.keys())))
    coor_orig = np.array([nodes_orig[nid] for nid in nids], dtype=float)
    coor_defor = np.array([nodes_defor[nid] for nid in nids], dtype=float)

    displacements = {nids[i]: coor_defor[i] - coor_orig[i] for i in range(len(nids))}
    displacements = {1.0: displacements}

    displacements = transform_displacements(nodes_orig, displacements, [1, 2, 5], form)

    basename = "failed_test_" + basename + "_" + form
    write_csv_displacements(basename , displacements)

    assert np.allclose(RESULT, np.array([displacements[1.0][nid] for nid in nids], dtype=float),
                       atol=1.e-5)

    # delete *.csv file of passed tests
    csv = [f for f in os.listdir() if basename in f and f.endswith(".csv")]
    for c in csv:
        os.remove(c)

