#!/usr/bin/python3

import os
import sys
import numpy as np

PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))), "bin")
print(PATH)
sys.path.append(PATH)

import pytest
import pdb

from pyFEA.meshIO._helpers import CustomDict, DictID, DictName


clist_01 = [["A", 1],
            ["B", 2]]
clist_02 = [("A", 1),
            ("B", 2)]
clist_03 = [["A", [1, 2]],
            ["B", [3, 4]]]
clist_04 = [["A", (1, 2)],
            ["B", (3, 4)]]
clist_05 = [("A", [1, 2]),
            ("B", [3, 4])]
clist_06 = [("A", (1, 2)),
            ("B", (3, 4))]
clist_07 = [["A", [[1, 2],[3, 4]]],
            ["B", [[5, 6],[7, 8]]]]
clist_08 = [("A", [[1, 2],[3, 4]]),
            ("B", [[5, 6],[7, 8]])]
clist_09 = [["A", np.array([1, 2])],
            ["B", np.array([3, 4])]]
clist_10 = [("A", np.array([1, 2])),
            ("B", np.array([3, 4]))]
clist_11 = [["A", np.array([[1, 2],[3, 4]])],
            ["B", np.array([[5, 6],[7, 8]])]]
clist_12 = [("A", np.array([[1, 2],[3, 4]])),
            ("B", np.array([[5, 6],[7, 8]]))]
clist = [clist_01, clist_02, clist_03, clist_04, clist_05, clist_06, clist_07,
         clist_08, clist_09, clist_10, clist_11, clist_12]

cdict_01 = {"A": 1,
            "B": 2}
cdict_02 = {"A": [1, 2],
            "B": [3, 4]}
cdict_03 = {"A": (1, 2),
            "B": (3, 4)}
cdict_04 = {"A": [[1, 2],[3, 4]],
            "B": [[5, 6],[7, 8]]}
cdict_05 = {"A": np.array([1, 2]),
            "B": np.array([3, 4])}
cdict_06 = {"A": np.array([[1, 2],[3, 4]]),
            "B": np.array([[5, 6],[7, 8]])}
cdict = [cdict_01, cdict_02, cdict_03, cdict_04, cdict_05, cdict_06]


fail_01 = {"A": 1,
             1: 2,
           "key": str,
           "fail": KeyError}
fail_02 = {-1: 1,
            2: 2,
           "key": int,
           "fail": ValueError}
fail_03 = {"_A": 1,
           "B": 2,
           "key": str,
           "fail": ValueError}
fail_04 = {"A": 1,
           "B": "2",
           "key": str,
           "fail": ValueError}
fail = [fail_01, fail_02, fail_03, fail_04]


class TestCustomDict:
    @pytest.mark.parametrize("check", clist)
    def test_list(self, check):
        key_type = str
        val_type = type(check[0][1])
        cd = CustomDict("ID", key_type, "Value", val_type, check)

        assert cd.count == len(check)
        assert cd.key_name == "ID"
        assert cd.val_name == "Value"
        for key, val in check:
            if type(val) in (tuple, list):
                assert all([val[i] == cd[key.upper()][i] for i in range(len(val))])
            elif type(val) is np.ndarray:
                assert np.array_equal(val, cd[key.upper()])
            else:
                assert val == cd[key.upper()]

    @pytest.mark.parametrize("check", cdict)
    def test_dict(self, check):
        key_type = str
        val_type = type(check[list(check.keys())[0]])
        cd = CustomDict("ID", key_type, "Value", val_type, check)

        assert cd.count == len(check)
        assert cd.key_name == "ID"
        assert cd.val_name == "Value"
        for key, val in check.items():
            if type(val) in (tuple, list):
                assert all([val[i] == cd[key.upper()][i] for i in range(len(val))])
            elif type(val) is np.ndarray:
                assert np.array_equal(val, cd[key.upper()])
            else:
                assert val == cd[key.upper()]

    @pytest.mark.parametrize("check", fail)
    def test_fail(self, check):
        key_type = check.pop("key")
        err = check.pop("fail")
        val_type = type(check[list(check.keys())[0]])
        with pytest.raises(err):
            cd = CustomDict("ID", key_type, "Value", val_type, check)

    def test_same_as_dict(self):
        a = {'A': 1}
        b = CustomDict("ID", str, "Value", int, a)
        assert a == b

    def test_tolist(self):
        a = {"A": [1, 2],
             "B": [3, 4]}
        cd = CustomDict("Name", str, "Values", list, a)

        assert cd.tolist() == [["A", 1, 2], ["B", 3, 4]]

    def test_todict(self):
        a = {"A": [1, 2],
             "B": [3, 4]}
        cd = CustomDict("Name", str, "Values", list, a)

        assert cd.todict() == a

    def test_toarray(self):
        a = {"A": [1, 2],
             "B": [3, 4]}
        cd = CustomDict("Name", str, "Values", list, a)

        assert np.array_equal(cd.toarray(), np.array(list(a.values())))

    def test_union(self):
        a = {"A": [1, 2]}
        b = {"B": [3, 4]}
        ca = CustomDict("Name", str, "Values", list, a)
        cb = CustomDict("Name", str, "Values", list, b)

        cc = ca | cb
        a.update(b)
        assert cc == a

    def test_difference(self):
        a = {"A": [1, 2],
             "B": [3, 4]}
        b = {"B": [3, 4],
             "C": [5, 6]}
        ca = CustomDict("Name", str, "Values", list, a)
        cb = CustomDict("Name", str, "Values", list, b)

        cc = ca - cb
        a = {"A": [1, 2]}
        assert cc == a

    def test_intersection(self):
        a = {"A": [1, 2],
             "B": [3, 4]}
        b = {"B": [3, 4],
             "C": [5, 6]}
        ca = CustomDict("Name", str, "Values", list, a)
        cb = CustomDict("Name", str, "Values", list, b)

        cc = ca & cb
        c = {key: a[key] for key in list(sorted(a.keys())) if key in b.keys()}
        assert cc == c

    def test_symmetric_difference(self):
        a = {"A": [1, 2],
             "B": [3, 4]}
        b = {"B": [3, 4],
             "C": [5, 6]}
        ca = CustomDict("Name", str, "Values", list, a)
        cb = CustomDict("Name", str, "Values", list, b)

        cc = ca ^ cb
        c = {"A": [1, 2], "C": [5, 6]}
        assert cc == c

    def test_str_str(self):
        a = {"A": [1, 2],
             "B": [3, 4]}
        ca = CustomDict("Name", str, "Values", list, a)
        assert str(ca) == "A : [1, 2]\nB : [3, 4]"

    def test_str_int(self):
        a = {1: [1, 2],
             2: [3, 4]}
        ca = CustomDict("ID", int, "Values", list, a)
        assert str(ca) == " 1: [1, 2]\n 2: [3, 4]"


class TestDictID:
    def test_pass(self):
        a = {2: [1, 2],
             1: [3, 4]}

        di = DictID(a)
        assert list(di.keys()) == [1, 2]
        assert di[1] == a[1]
        assert di[2] == a[2]

    def test_fail_value(self):
        a = {-1: [1, 2],
             2: [3, 4]}

        with pytest.raises(ValueError):
            di = DictID(a)

    def test_fail_key(self):
        a = {"a": [1, 2],
             2: [3, 4]}

        with pytest.raises(KeyError):
            di = DictID(a)


class TestDictName:
    def test_pass(self):
        a = {"B": [1, 2],
             "a": [3, 4]}

        dn = DictName(a)
        assert list(dn.keys()) == ["A", "B"]
        assert dn["A"] == a["a"]
        assert dn["B"] == a["B"]

    def test_fail_value(self):
        a = {"_A": [1, 2],
             "B": [3, 4]}

        with pytest.raises(ValueError):
            dn = DictName(a)

    def test_fail_key(self):
        a = {"a": [1, 2],
             2: [3, 4]}

        with pytest.raises(KeyError):
            dn = DictName(a)


