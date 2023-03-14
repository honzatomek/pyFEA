import os
import sys
import pytest
import numpy as np
from random import randint

import pdb

try:
    from pyFEA.meshIO.model import Node, Nodes, Element, Elements
    from pyFEA.meshIO.model import Material, MaterialISO, Materials
    from pyFEA.meshIO.model import Property, Properties
    from pyFEA.meshIO.model import PMass, PRod, PBeam
    from pyFEA.meshIO.model import PShell, PTria3, PTria6
    from pyFEA.meshIO.model import PQuad4, PQuad8, PSolid
    from pyFEA.meshIO.model import CLoad, LoadsN, LoadE, LoadsE, Loading

except ImportError as e:
    SRC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    SRC = os.path.join(SRC, 'bin')
    sys.path.append(SRC)
    from pyFEA.meshIO.model import Node, Nodes, Element, Elements
    from pyFEA.meshIO.model import Material, MaterialISO, Materials
    from pyFEA.meshIO.model import Property, Properties
    from pyFEA.meshIO.model import PMass, PRod, PBeam
    from pyFEA.meshIO.model import PShell, PTria3, PTria6
    from pyFEA.meshIO.model import PQuad4, PQuad8, PSolid
    from pyFEA.meshIO.model import CLoad, LoadsN, LoadE, LoadsE, Loading

from _models import models, materials, properties, nodal_loads, nodal_loads_fail
from _models import nodal_loads_list, nodal_loads_dict
from _models import mapping


class TestNode:
    def test_init_(self):
        nid = 1
        X = np.random.rand(3)
        n = Node(nid, X)
        assert n.id == nid
        assert np.array_equal(n.coors, X)

    def test_id_lower_zero(self):
        nid = -1
        X = np.random.rand(3)
        with pytest.raises(ValueError):
            n = Node(nid, X)

    def test_id_equal_zero(self):
        nid = 0
        X = np.random.rand(3)
        with pytest.raises(ValueError):
            n = Node(nid, X)

    def test_id_not_int(self):
        nid = '1'
        X = np.random.rand(3)
        with pytest.raises(TypeError):
            n = Node(nid, X)

        nid = 1.
        X = np.random.rand(3)
        with pytest.raises(TypeError):
            n = Node(nid, X)

    def test_coor_not_list(self):
        nid = 1.
        X = 1 # np.random.rand(3)
        with pytest.raises(TypeError):
            n = Node(nid, X)

    def test_coor_short(self):
        nid = 1001
        X =  np.random.rand(2)
        n = Node(nid, X)
        assert n.id == nid
        assert np.array_equal(n.coors, np.hstack((X, [0.])))

        nid = 1001
        X =  np.random.rand(1)
        n = Node(nid, X)
        assert n.id == nid
        assert np.array_equal(n.coors, np.hstack((X, [0.] * 2)))

    def test_eq_(self):
        X = np.random.rand(3)
        n1 = Node(1, X)
        n2 = Node(2, X)
        assert not (n1 == n2)

        X = np.random.rand(3)
        n1 = Node(1, X)
        n2 = Node(1, X)
        assert n1 == n2

    def test_neq_(self):
        X = np.random.rand(3)
        n1 = Node(1, X)
        n2 = Node(2, X)
        assert n1 != n2

        X = np.random.rand(3)
        n1 = Node(1, X)
        assert not (n1 == n2)

    def test_add_(self):
        test_data = [1,
                     [1, 1, 1],
                     [1., 1., 1.],
                     np.array([1, 1, 1], dtype=float),
                     np.array([1, 1, 1], dtype=int)]
        for Y in test_data:
            X = np.random.rand(3)
            n1 = Node(1, X)
            n2 = n1 + Y
            assert np.array_equal(n2.coors, X + Y)

    def test_iadd_(self):
        test_data = [1,
                     [1, 1, 1],
                     [1., 1., 1.],
                     np.array([1, 1, 1], dtype=float),
                     np.array([1, 1, 1], dtype=int)]
        for Y in test_data:
            X = np.random.rand(3)
            n = Node(1, X)
            n += Y
            assert np.array_equal(n.coors, X + Y)

    def test_sub_(self):
        test_data = [1,
                     [1, 1, 1],
                     [1., 1., 1.],
                     np.array([1, 1, 1], dtype=float),
                     np.array([1, 1, 1], dtype=int)]
        for Y in test_data:
            X = np.random.rand(3)
            n1 = Node(1, X)
            n2 = n1 - Y
            assert np.array_equal(n2.coors, X - Y)

    def test_isub_(self):
        test_data = [1,
                     [1, 1, 1],
                     [1., 1., 1.],
                     np.array([1, 1, 1], dtype=float),
                     np.array([1, 1, 1], dtype=int)]
        for Y in test_data:
            X = np.random.rand(3)
            n = Node(1, X)
            n -= Y
            assert np.array_equal(n.coors, X - Y)

    def test_mul_(self):
        test_data = [2,
                     [2, 2, 2],
                     [2., 2., 2.],
                     np.array([2, 2, 2], dtype=float),
                     np.array([2, 2, 2], dtype=int)]
        for Y in test_data:
            X = np.random.rand(3)
            n1 = Node(1, X)
            n2 = n1 * Y
            assert np.array_equal(n2.coors, X * Y)

    def test_imul_(self):
        test_data = [2,
                     [2, 2, 2],
                     [2., 2., 2.],
                     np.array([2, 2, 2], dtype=float),
                     np.array([2, 2, 2], dtype=int)]
        for Y in test_data:
            X = np.random.rand(3)
            n = Node(1, X)
            n *= Y
            assert np.array_equal(n.coors, X * Y)

    def test_truediv_(self):
        test_data = [2,
                     [2, 2, 2],
                     [2., 2., 2.],
                     np.array([2, 2, 2], dtype=float),
                     np.array([2, 2, 2], dtype=int)]
        for Y in test_data:
            X = np.random.rand(3)
            n1 = Node(1, X)
            n2 = n1 / Y
            assert np.array_equal(n2.coors, X / Y)

    def test_itruediv_(self):
        test_data = [2,
                     [2, 2, 2],
                     [2., 2., 2.],
                     np.array([2, 2, 2], dtype=float),
                     np.array([2, 2, 2], dtype=int)]
        for Y in test_data:
            X = np.random.rand(3)
            n = Node(1, X)
            n /= Y
            assert np.array_equal(n.coors, X / Y)

    def test_pow_(self):
        X = np.random.rand(3)
        Y = 2
        n1 = Node(1, X)
        n2 = n1 ** Y
        assert np.array_equal(n2.coors, X ** Y)

    def test_transform_33(self):
        X = np.array([1, 0, 0], dtype=float)
        T = np.array([[ 0, 1, 0],
                      [-1, 0, 0],
                      [ 0, 0, 1]], dtype=float)
        nd = Node(1, X)
        nd.transform(T)
        assert np.array_equal(nd.coors, np.array([0, -1, 0], dtype=float))

    def test_transform_34(self):
        X = np.array([1, 0, 0], dtype=float)
        T = np.array([[ 0, 1, 0, 10],
                      [-1, 0, 0, 10],
                      [ 0, 0, 1, 10]], dtype=float)
        nd = Node(1, X)
        nd.transform(T)
        assert np.array_equal(nd.coors, np.array([10, 9, 10], dtype=float))

    def test_transform_44(self):
        X = np.array([1, 0, 0], dtype=float)
        T = np.array([[ 0, 1, 0, 10],
                      [-1, 0, 0, 10],
                      [ 0, 0, 1, 10],
                      [ 0, 0, 0,  1]], dtype=float)
        nd = Node(1, X)
        nd.transform(T)
        assert np.array_equal(nd.coors, np.array([10, 9, 10], dtype=float))

    def test_transform_else(self):
        X = np.array([1, 0, 0], dtype=float)
        nd = Node(1, X)
        with pytest.raises(TypeError):
            nd.transform('a')

        with pytest.raises(TypeError):
            nd.transform([[ 0, 1, 0],
                          [-1, 0, 0],
                          [ 0, 0, 1]])

        with pytest.raises(ValueError):
            nd.transform(np.array([[0, 1],[-1, 0]], dtype=float))



class TestNodes:
    def test_init_list_Node(self):
        n1 = Node(1, np.random.rand(3))
        n2 = Node(2, np.random.rand(3))
        nds = Nodes([n1, n2])
        assert len(nds) == 2

    def test_init_list_list(self):
        n1 = np.random.rand(3).tolist()
        n2 = np.random.rand(3).tolist()
        # pdb.set_trace()
        nds = Nodes([n1, n2])
        assert len(nds) == 2

    def test_init_list_tuple(self):
        n1 = tuple(np.random.rand(3).tolist())
        n2 = tuple(np.random.rand(3).tolist())
        nds = Nodes([n1, n2])
        assert len(nds) == 2

    def test_init_list_str(self):
        with pytest.raises(TypeError):
            nds = Nodes(['a', 'b'])

    def test_init_dict_Node(self):
        n1 = Node(1, np.random.rand(3))
        n2 = Node(2, np.random.rand(3))
        nds = Nodes({n1.id: n1, n2.id: n2})
        assert len(nds) == 2

    def test_init_dict_list(self):
        n1 = np.random.rand(3).tolist()
        n2 = np.random.rand(3).tolist()
        nds = Nodes({1: n1, 2: n2})
        assert len(nds) == 2

    def test_init_dict_array(self):
        n1 = np.random.rand(3)
        n2 = np.random.rand(3)
        nds = Nodes({1: n1, 2: n2})
        assert len(nds) == 2

    def test_init_duplicate(self):
        n1 = Node(1, np.random.rand(3))
        n2 = Node(1, np.random.rand(3))
        with pytest.raises(ValueError):
            nds = Nodes([n1, n2])

    def test_min_max_ID(self):
        n1 = Node(2001, np.random.rand(3))
        n2 = Node(1001, np.random.rand(3))
        nds = Nodes([n1, n2])
        assert nds.min == 1001
        assert nds.max == 2001

    def test_extents(self):
        n1 = Node(1001, [ 1., -1.,  1.])
        n2 = Node(1002, [-1.,  1., -1.])
        nds = Nodes([n1, n2])

        ext = nds.extents

        assert np.array_equal(ext[0], np.array([-1, -1, -1], dtype=float))
        assert np.array_equal(ext[1], np.array([ 1,  1,  1], dtype=float))

    def test_transform(self):
        x1 = np.array([ 1., -1.,  1.], dtype=float)
        n1 = Node(1001, x1)
        x2 = np.array([-1.,  1.,  -1.], dtype=float)
        n2 = Node(1002, x2)
        nds = Nodes([n1, n2])

        T = np.array([[ 0, 1, 0],
                      [-1, 0, 0],
                      [ 0, 0, 1]], dtype=float)

        nds.transform(T)

        assert np.array_equal(T @ x1, nds[1001].coors)
        assert np.array_equal(T @ x2, nds[1002].coors)

    def test_aslist(self):
        X = [[1., 2., 3.],
             [4., 5., 6.]]
        n1 = Node(1001, X[0])
        n2 = Node(1002, X[1])
        nds = Nodes([n1, n2])

        assert nds.aslist() == X

    def test_asdict(self):
        X = {1001: np.array([1., 2., 3.], dtype=float),
             1002: np.array([4., 5., 6.], dtype=float)}
        nds = Nodes(X)

        dnds = nds.asdict()
        for nid in dnds.keys():
            assert np.array_equal(X[nid], dnds[nid])

    def test_asarray(self):
        X = {1001: np.array([1., 2., 3.], dtype=float),
             1002: np.array([4., 5., 6.], dtype=float)}
        nds = Nodes(X)

        assert np.array_equal(np.array(list(X.values()), dtype=float), nds.asarray())

    def test_overwrite(self):
        X = Node(1001, np.array([1., 2., 3.], dtype=float))
        Y = X + np.array([3, 4, 5], dtype=float)

        nds = Nodes([X])
        nds.nodes[1001].coors = Y.coors

        assert np.array_equal(nds.nodes[1001].coors, Y.coors)

    def test_repr(self):
        X = {1001: np.array([1., 2., 3.], dtype=float),
             1002: np.array([4., 5., 6.], dtype=float)}
        nds = Nodes(X)
        assert repr(nds) == "<Nodes container object>\n  Count: 2\n  Min ID: 1001\n  Max ID: 1002"

    def test_str(self):
        X = {1001: np.array([1., 2., 3.], dtype=float),
             1002: np.array([4., 5., 6.], dtype=float)}
        nds = Nodes(X)
        assert str(nds) == "Nodes count: 2, min ID: 1001, max ID: 1002"

    def test_max(self):
        X = {1001: np.array([1., 2., 3.], dtype=float),
             1002: np.array([4., 5., 6.], dtype=float)}
        nds = Nodes(X)
        assert nds.max == 1002

    def test_min(self):
        X = {1001: np.array([1., 2., 3.], dtype=float),
             1002: np.array([4., 5., 6.], dtype=float)}
        nds = Nodes(X)
        assert nds.min == 1001

    def test_count(self):
        X = {1001: np.array([1., 2., 3.], dtype=float),
             1002: np.array([4., 5., 6.], dtype=float)}
        nds = Nodes(X)
        assert nds.count == 2

    def test_keys(self):
        X = {1001: np.array([1., 2., 3.], dtype=float),
             1002: np.array([4., 5., 6.], dtype=float)}
        nds = Nodes(X)
        assert list(nds.keys()) == [1001, 1002]

    def test_items(self):
        X = {1001: np.array([1., 2., 3.], dtype=float),
             1002: np.array([4., 5., 6.], dtype=float)}
        nds = Nodes(X)
        for nid, node in nds.items():
            assert nid in X.keys()
            assert np.array_equal(node.coors, X[nid])

    def test_values(self):
        X = {1001: np.array([1., 2., 3.], dtype=float),
             1002: np.array([4., 5., 6.], dtype=float)}
        nds = Nodes(X)
        for node in nds.values():
            assert node.id in X.keys()
            assert np.array_equal(node.coors, X[node.id])

    def test_add(self):
        X1 = {1001: np.array([1., 2., 3.], dtype=float),
              1002: np.array([4., 5., 6.], dtype=float)}
        nds = Nodes(X1)

        X2 = {1003: np.array([1., 2., 3.], dtype=float),
              1004: np.array([4., 5., 6.], dtype=float)}
        nds = nds + Nodes(X2)

        assert nds.count == 4
        assert list(nds.keys()) == [1001, 1002, 1003, 1004]

        with pytest.raises(ValueError):
            nds = nds + Nodes(X2)

    def test_iadd(self):
        X1 = {1001: np.array([1., 2., 3.], dtype=float),
              1002: np.array([4., 5., 6.], dtype=float)}
        nds = Nodes(X1)

        X2 = {1003: np.array([1., 2., 3.], dtype=float),
              1004: np.array([4., 5., 6.], dtype=float)}
        nds += Nodes(X2)

        assert nds.count == 4
        assert list(nds.keys()) == [1001, 1002, 1003, 1004]

        with pytest.raises(ValueError):
            nds += Nodes(X2)

    def test_map_scalar(self):
        onodes = Nodes(mapping["onodes"])
        nnodes = Nodes(mapping["nnodes"])
        scalar = mapping["scalar"]

        results, distances = nnodes.map_scalar(onodes, scalar, 20., True)
        # results, distances = nnodes.map_tensor(onodes, scalar, 20., True)
        assert np.array_equal(nnodes.coors[:,2], np.array(list(results.values()), dtype=float))

    def test_map_vector(self):
        onodes = Nodes(mapping["onodes"])
        nnodes = Nodes(mapping["nnodes"])
        vector = mapping["vector"]

        results, distances = nnodes.map_vector(onodes, vector, 20., True)
        # results, distances = nnodes.map_tensor(onodes, vector, 20., True)
        assert np.array_equal(nnodes.coors, np.array(list(results.values()), dtype=float))

    def test_map_tensor(self):
        onodes = Nodes(mapping["onodes"])
        nnodes = Nodes(mapping["nnodes"])
        tensor = mapping["tensor"]

        results, distances = nnodes.map_tensor(onodes, tensor, 20., True)
        assert np.array_equal(nnodes.coors, np.array(list(results.values()), dtype=float)[:,0,:])
        assert np.array_equal(nnodes.coors, np.array(list(results.values()), dtype=float)[:,1,:])
        assert np.array_equal(nnodes.coors, np.array(list(results.values()), dtype=float)[:,2,:])



class TestElement:
    @pytest.mark.parametrize("model", models)
    def test_init_(self, model):
        nds = model["nodes"]
        fetype = model["elements"][0]
        elements = model["elements"][1]

        for i, nodes in enumerate(elements):
            element = Element(i + 1, fetype=fetype, nodes=nodes)
            assert i + 1 == element.id
            assert fetype == element.fetype
            assert len(nodes) == element.count
            assert len(nodes) == len(element)
            assert nodes == element.nodes
            assert repr(element) == f"<Element object>\n  ID: {i+1:n}\n  Type: {fetype:s}\n  Nodes: {len(nodes):n}"
            assert str(element) == f"{fetype:s} Element ID {i+1:n}"

    def test_etype(self):
        element = Element(1001, "custom_hex27", list(range(1, 28)))
        assert element.id == 1001
        assert element.fetype == "CUSTOM_HEX27"
        assert len(element) == 27
        assert element.count == 27
        assert str(element) == "CUSTOM_HEX27 Element ID 1001"
        assert element.nodes == list(range(1, 28))


class TestElements:
    @pytest.mark.parametrize("model", models)
    def test_init_list(self, model):
        nds = model["nodes"]
        fetype = model["elements"][0]
        elenodes = model["elements"][1]
        els = []
        for eid in range(len(elenodes)):
            els.append([eid + 1, fetype, *elenodes[eid]])

        elements = Elements(els)
        assert elements.count == len(elenodes)
        assert list(elements.keys()) == list(range(1, len(els) + 1))


    @pytest.mark.parametrize("model", models)
    def test_init_list_Elements(self, model):
        nds = model["nodes"]
        fetype = model["elements"][0]
        elenodes = model["elements"][1]
        els = []
        for eid in range(len(elenodes)):
            els.append(Element(eid + 1, fetype, elenodes[eid]))

        elements = Elements(els)
        assert elements.count == len(elenodes)
        assert list(elements.keys()) == list(range(1, len(els) + 1))

    @pytest.mark.parametrize("model", models)
    def test_init_dict(self, model):
        nds = model["nodes"]
        fetype = model["elements"][0]
        elenodes = model["elements"][1]
        els = {}
        for eid in range(len(elenodes)):
            els.setdefault(eid + 1, {"eid": eid + 1, "fetype": fetype, "nodes": elenodes[eid]})

        elements = Elements(els)
        assert elements.count == len(elenodes)
        assert list(elements.keys()) == list(range(1, len(els) + 1))


    @pytest.mark.parametrize("model", models)
    def test_init_dict_Elements(self, model):
        nds = model["nodes"]
        fetype = model["elements"][0]
        elenodes = model["elements"][1]
        els = {}
        for eid in range(len(elenodes)):
            els.setdefault(eid + 1, Element(eid + 1, fetype, elenodes[eid]))

        elements = Elements(els)
        assert elements.count == len(elenodes)
        assert list(elements.keys()) == list(range(1, len(els) + 1))

    @pytest.mark.parametrize("model1", models)
    @pytest.mark.parametrize("model2", models)
    def test_init_multiple_types(self, model1, model2):
        nds = model1["nodes"]
        fetype = model1["elements"][0]
        elenodes = model1["elements"][1]

        els = []
        for eid in range(len(elenodes)):
            els.append(Element(eid + 1, fetype, elenodes[eid]))

        model1_nds = len(nds)
        model1_els = len(els)
        nds.extend(model2["nodes"])
        fetype = model2["elements"][0]
        elenodes = [[n + model1_nds for n in nd] for nd in model2["elements"][1]]

        for eid in range(len(elenodes)):
            els.append(Element(eid + model1_els + 1, fetype, elenodes[eid]))

        elements = Elements(els)
        assert elements.count == len(model1["elements"][1]) + len(model2["elements"][1])

    @pytest.mark.parametrize("model", models)
    def test_asdict(self, model):
        nds = model["nodes"]
        fetype = model["elements"][0]
        elenodes = model["elements"][1]
        els = {}
        for eid in range(len(elenodes)):
            els.setdefault(eid + 1, {"eid": eid + 1, "fetype": fetype, "nodes": elenodes[eid]})

        elements = Elements(els)
        assert elements.asdict() == els

    @pytest.mark.parametrize("model", models)
    def test_aslist(self, model):
        nds = model["nodes"]
        fetype = model["elements"][0]
        elenodes = model["elements"][1]
        els = []
        for eid in range(len(elenodes)):
            els.append([eid + 1, fetype, *elenodes[eid]])

        elements = Elements(els)
        assert elements.aslist() == els

    @pytest.mark.parametrize("model1", models)
    @pytest.mark.parametrize("model2", models)
    def test_add_(self, model1, model2):
        nds = model1["nodes"]
        fetype = model1["elements"][0]
        elenodes = model1["elements"][1]

        _els = {}
        els = {}
        for eid in range(len(elenodes)):
            # _els.setdefault(eid + 1, [fetype, elenodes[eid]])
            # els.setdefault(eid + 1, [fetype, elenodes[eid]])
            _els.setdefault(eid + 1, {"eid": eid + 1, "fetype": fetype, "nodes": elenodes[eid]})
            els.setdefault(eid + 1, {"eid": eid + 1, "fetype": fetype, "nodes": elenodes[eid]})

        elements1 = Elements(_els)

        nds = model2["nodes"]
        fetype = model2["elements"][0]
        elenodes = model2["elements"][1]

        _els = {}
        offset = len(els)
        for eid in range(len(elenodes)):
            # _els.setdefault(eid + 1 + offset, [fetype, elenodes[eid]])
            # els.setdefault(eid + 1 + offset, [fetype, elenodes[eid]])
            _els.setdefault(eid + 1 + offset, {"eid": eid + 1 + offset, "fetype": fetype, "nodes": elenodes[eid]})
            els.setdefault(eid + 1 + offset, {"eid": eid + 1 + offset, "fetype": fetype, "nodes": elenodes[eid]})

        elements2 = Elements(_els)

        elements = elements1 + elements2
        assert len(elements) == len(els)
        assert elements.asdict() == els

    @pytest.mark.parametrize("model1", models)
    @pytest.mark.parametrize("model2", models)
    def test_iadd_(self, model1, model2):
        nds = model1["nodes"]
        fetype = model1["elements"][0]
        elenodes = model1["elements"][1]

        _els = {}
        els = {}
        for eid in range(len(elenodes)):
            # _els.setdefault(eid + 1, [fetype, elenodes[eid]])
            # els.setdefault(eid + 1, [fetype, elenodes[eid]])
            _els.setdefault(eid + 1, {"eid": eid + 1, "fetype": fetype, "nodes": elenodes[eid]})
            els.setdefault(eid + 1, {"eid": eid + 1, "fetype": fetype, "nodes": elenodes[eid]})

        elements = Elements(_els)

        nds = model2["nodes"]
        fetype = model2["elements"][0]
        elenodes = model2["elements"][1]

        _els = {}
        offset = len(els)
        for eid in range(len(elenodes)):
            # _els.setdefault(eid + 1 + offset, [fetype, elenodes[eid]])
            # els.setdefault(eid + 1 + offset, [fetype, elenodes[eid]])
            _els.setdefault(eid + 1 + offset, {"eid": eid + 1 + offset, "fetype": fetype, "nodes": elenodes[eid]})
            els.setdefault(eid + 1 + offset, {"eid": eid + 1 + offset, "fetype": fetype, "nodes": elenodes[eid]})

        elements += Elements(_els)

        assert len(elements) == len(els)
        assert elements.asdict() == els


class TestMaterialISO:
    @pytest.mark.parametrize("material", materials[:-2])
    def test_init_array(self, material):
        name = material["name"]
        fetype = material["fetype"]
        E = material["E"]
        n = material["nu"]
        r = material["rho"]
        a = material["alpha"]
        if material["G"] is None:
            G = None
        else:
            G = material["G"]
        result = material["result"]

        if type(E) is list: E = np.array(E)
        if type(n) is list: n = np.array(n)
        if type(r) is list: r = np.array(r)
        if type(a) is list: a = np.array(a)
        if type(G) is list: G = np.array(G)

        if result == "pass":
            if G is not None:
                _G = G
            elif type(E) is np.ndarray:
                _G = E.copy()
                if type(n) is np.ndarray:
                    for i, t in enumerate(E[:,0]):
                        _G[i,1] = E[i,1] / (2. * (1. + np.interp(t, n[:,0], n[:,1])))
                else:
                    _G[:,1] = E[:,1] / (2. * (1. + n))
            else:
                if type(n) is np.ndarray:
                    _G = n.copy()
                    _G[:,1] = E / (2. * (1. + n[:,1]))
                else:
                    _G = E / (2. * (1. + n))

            # m = MaterialISO(name, E, n, r, a, G)
            m = Material.New(name, fetype, E, n, r, a, G)

            assert m.name == name
            if type(E) is np.ndarray:
                assert np.array_equal(m.young, E)
            else:
                assert m.young == E
            if type(n) is np.ndarray:
                assert np.array_equal(m.poisson, n)
            else:
                assert m.poisson == n
            if type(r) is np.ndarray:
                assert np.array_equal(m.density, r)
            else:
                assert m.density == r
            if type(a) is np.ndarray:
                assert np.array_equal(m.thermal_expansion, a)
            else:
                assert m.thermal_expansion == a
            if type(_G) is np.ndarray:
                print(f"orig: {_G}")
                print(f"new: {m.shear}")
                assert np.array_equal(m.shear, _G)
            else:
                assert m.shear == _G

            if type(E) is np.ndarray: E = E.tolist()
            if type(n) is np.ndarray: n = n.tolist()
            if type(r) is np.ndarray: r = r.tolist()
            if type(a) is np.ndarray: a = a.tolist()
            if type(_G) is np.ndarray: _G = _G.tolist()

            mdict = m.asdict()
            assert mdict["name"] == name
            assert mdict["fetype"] == fetype
            assert mdict["E"] == E
            assert mdict["nu"] == n
            assert mdict["rho"] == r
            assert mdict["alpha"] == a
            assert mdict["G"] == _G

            mlist = m.aslist()
            assert mlist[0] == name
            assert mlist[1] == fetype
            assert mlist[2] == E
            assert mlist[3] == n
            assert mlist[4] == r
            assert mlist[5] == a
            assert mlist[6] == _G


        else:
            with pytest.raises(result):
                # m = MaterialISO(name, E, n, r, a, G)
                m = Material.New(name, fetype, E, n, r, a, G)

    @pytest.mark.parametrize("material", materials)
    def test_init_list(self, material):
        name = material["name"]
        fetype = material["fetype"]
        E = material["E"]
        n = material["nu"]
        r = material["rho"]
        a = material["alpha"]
        if material["G"] is None:
            G = None
        else:
            G = material["G"]
        result = material["result"]

        if result == "pass":
            if G is not None:
                _G = G
            elif type(E) is list:
                _G = []
                if type(n) is list:
                    for i, (t, _e) in enumerate(E):
                        _G.append([t, _e / (2. * (1. + np.interp(t, np.array(n)[:,0], np.array(n)[:,1])))])
                else:
                    for i, (t, _e) in eumerate(E):
                        _G.append([t, _e / (2. * (1. + n))])
            else:
                if type(n) is list:
                    _G = []
                    for i, (t, _n) in enumerate(n):
                        _G.append([t, E / (2. * (1. + _n))])
                else:
                    _G = E / (2. * (1. + n))

            # m = MaterialISO(name, E, n, r, a, G)
            m = Material.New(name, fetype, E, n, r, a, G)

            assert m.name == name
            assert m.fetype == fetype
            if type(E) is list:
                assert m.young.tolist() == E
            else:
                assert m.young == E
            if type(n) is list:
                assert m.poisson.tolist() == n
            else:
                assert m.poisson == n
            if type(r) is list:
                assert m.density.tolist() == r
            else:
                assert m.density == r
            if type(a) is list:
                assert m.thermal_expansion.tolist() == a
            else:
                assert m.thermal_expansion == a
            if type(_G) is list:
                assert m.shear.tolist() == _G
            else:
                assert m.shear == _G

            mdict = m.asdict()
            assert mdict["name"] == name
            assert mdict["fetype"] == fetype
            assert mdict["E"] == E
            assert mdict["nu"] == n
            assert mdict["rho"] == r
            assert mdict["alpha"] == a
            assert mdict["G"] == _G

            mlist = m.aslist()
            assert mlist[0] == name
            assert mlist[1] == fetype
            assert mlist[2] == E
            assert mlist[3] == n
            assert mlist[4] == r
            assert mlist[5] == a
            assert mlist[6] == _G

        else:
            with pytest.raises(result):
                # m = MaterialISO(name, E, n, r, a, G)
                m = Material.New(name, fetype, E, n, r, a, G)

    @pytest.mark.parametrize("material", materials)
    def test_init_dict(self, material):
        name = material["name"]
        fetype = material["fetype"]
        E = material["E"]
        n = material["nu"]
        r = material["rho"]
        a = material["alpha"]
        if material["G"] is None:
            G = None
        else:
            G = material["G"]
        result = material["result"]

        if type(E) is list: E = {t: e for t, e in E}
        if type(n) is list: n = {t: e for t, e in n}
        if type(r) is list: r = {t: e for t, e in r}
        if type(a) is list: a = {t: e for t, e in a}
        if type(G) is list: G = {t: e for t, e in G}

        if result == "pass":
            if G is not None:
                _G = G
            elif type(E) is dict:
                _G = []
                if type(n) is dict:
                    for i, (t, _e) in enumerate(E.items()):
                        _G.append([t, _e / (2. * (1. + np.interp(t, np.array(list(n.keys())), np.array(list(n.values())))))])
                else:
                    for i, (t, _e) in eumerate(E.items()):
                        _G.append([t, _e / (2. * (1. + n))])
                _G = {t: g for t, g in _G}
            else:
                if type(n) is dict:
                    _G = []
                    for i, (t, _n) in enumerate(n.items()):
                        _G.append([t, E / (2. * (1. + _n))])
                    _G = {t: g for t, g in _G}
                else:
                    _G = E / (2. * (1. + n))

            # m = MaterialISO(name, E, n, r, a, G)
            m = Material.New(name, fetype, E, n, r, a, G)

            assert m.name == name
            if type(E) is dict:
                assert {t: e for t, e in m.young} == E
            else:
                assert m.young == E
            if type(n) is dict:
                assert {t: e for t, e in m.poisson} == n
            else:
                assert m.poisson == n
            if type(r) is dict:
                assert {t: e for t, e in m.density} == r
            else:
                assert m.density == r
            if type(a) is dict:
                assert {t: e for t, e in m.thermal_expansion} == a
            else:
                assert m.thermal_expansion == a
            if type(_G) is dict:
                assert {t: e for t, e in m.shear} == _G
            else:
                assert m.shear == _G

            if type(E) is dict: E = [list(row) for row in list(E.items())]
            if type(n) is dict: n = [list(row) for row in list(n.items())]
            if type(r) is dict: r = [list(row) for row in list(r.items())]
            if type(a) is dict: a = [list(row) for row in list(a.items())]
            if type(_G) is dict: _G = [list(row) for row in list(_G.items())]

            mdict = m.asdict()
            assert mdict["name"] == name
            assert mdict["fetype"] == fetype
            assert mdict["E"] == E
            assert mdict["nu"] == n
            assert mdict["rho"] == r
            assert mdict["alpha"] == a
            assert mdict["G"] == _G

            mlist = m.aslist()
            assert mlist[0] == name
            assert mlist[1] == fetype
            assert mlist[2] == E
            assert mlist[3] == n
            assert mlist[4] == r
            assert mlist[5] == a
            assert mlist[6] == _G

        else:
            with pytest.raises(result):
                # m = MaterialISO(name, E, n, r, a, G)
                m = Material.New(name, fetype, E, n, r, a, G)

class TestMaterial:
    @pytest.mark.parametrize("material", materials)
    def test_New_dict(self, material):
        if material["result"] == "pass":
            m = Material.New(material["name"], material["fetype"], material)
            assert m.name == material["name"]
            assert m.fetype == material["fetype"]
        else:
            with pytest.raises(material["result"]):
                m = Material.New(material["name"], material["fetype"], material)

    @pytest.mark.parametrize("material", materials)
    def test_New_list(self, material):
        if material["result"] == "pass":
            mat = []
            for key in ("E", "nu", "rho", "alpha", "G"):
                mat.append(material[key])
            m = Material.New(material["name"], material["fetype"], mat)
            assert m.name == material["name"]
            assert m.fetype == material["fetype"]
        else:
            with pytest.raises(material["result"]):
                m = Material.New(material["name"], material["fetype"], material)

    @pytest.mark.parametrize("material", materials)
    def test_New_expand_list(self, material):
        if material["result"] == "pass":
            mat = []
            for key in ("E", "nu", "rho", "alpha", "G"):
                mat.append(material[key])
            m = Material.New(material["name"], material["fetype"], *mat)
            assert m.name == material["name"]
            assert m.fetype == material["fetype"]
        else:
            with pytest.raises(material["result"]):
                m = Material.New(material["name"], material["fetype"], material)


class TestMaterials:
    def test_init_dict(self):
        mats = [[m["name"], m["fetype"], m] for m in materials if m["result"] == "pass"]
        m = Materials(mats)
        assert m.count == len(mats)

    def test_init_list(self):
        mats = [[m["name"], m["fetype"], m] for m in materials if m["result"] == "pass"]
        for i in range(len(mats)):
            _mats = []
            for key in ("E", "nu", "rho", "alpha", "G"):
                _mats.append(mats[i][2][key])
            mats[i][2] = _mats
        m = Materials(mats)
        assert m.count == len(mats)

    def test_dupl(self):
        mats = [["stahl", m["fetype"], m] for m in materials if m["result"] == "pass"]
        with pytest.raises(ValueError):
            m = Materials(mats)

    def test_type(self):
        mats = [[m["name"], "ISO_test", m] for m in materials if m["result"] == "pass"]
        with pytest.raises(ValueError):
            m = Materials(mats)

class TestProperty:
    @pytest.mark.parametrize("property", properties)
    def test_init_(self, property):
        if type(property) in (tuple, list):
            p = Property.New(*property)
            assert p.name == property[0]
        elif type(property) is dict:
            p = Property.New(**property)
            assert p.name == property["name"]

class TestProperties:
    def test_init_(self):
        ps = Properties(properties)
        assert ps.count == len(ps)
        for i in range(len(properties)):
            if type(properties[i]) is dict:
                name = properties[i]["name"]
                assert ps[name].asdict() == properties[i]
            elif type(properties[i]) is list:
                name = properties[i][0]
                assert ps[name].aslist() == properties[i]
            else:
                name = properties[i][0]
                assert ps[name].aslist() == list(properties[i])


class TestCLoad:
    @pytest.mark.parametrize("loadn", nodal_loads)
    def test_init_(self, loadn):
        # pdb.set_trace()
        if type(loadn) is dict:
            l = CLoad(**loadn)
            assert l.asdict() == loadn
        else:
            l = CLoad(*loadn[1:])
            assert l.aslist() == list(loadn)

    @pytest.mark.parametrize("loadn", nodal_loads_fail)
    def test_init_fail(self, loadn):
        result = loadn.pop("result")
        with pytest.raises(result):
            l = CLoad(**loadn)

class TestLoadsN:
    def test_init_CLoad(self):
        loadn = []
        nodes = []
        lpat = nodal_loads[0]["lpat"]
        for l in nodal_loads:
            if type(l) is dict:
                loadn.append(CLoad(**l))
                nodes.append(loadn[-1].node)
            else:
                loadn.append(CLoad(*l[1:]))
                nodes.append(loadn[-1].node)
        loads = LoadsN(lpat, loadn)
        nodes = list(set(nodes))
        assert loads.count == len(nodes)


class TestLoading:
    def test_init_(self):
        loadn = []
        nodes = []
        # pdb.set_trace()
        lpat = nodal_loads[0]["lpat"]
        for l in nodal_loads:
            if type(l) is dict:
                loadn.append(CLoad(**l))
                nodes.append(loadn[-1].node)
            else:
                loadn.append(CLoad(*l[1:]))
                nodes.append(loadn[-1].node)
        lpat = Loading(loadn)
        print(f"{lpat.nodal = }")
        # nodes = list(set(nodes))
        # assert loads.count == len(nodes)

    def test_init_list(self):
        loading = Loading()
        loading.nodal = nodal_loads_list
        loads_per_lpat = {}
        for nl in nodal_loads_list:
            if nl[1] not in loads_per_lpat.keys():
                loads_per_lpat.setdefault(nl[1], 0)
            loads_per_lpat[nl[1]] += 1
        assert len(loading.nodal) == len(loads_per_lpat)
        pdb.set_trace()
        for lpat in loading.nodal.keys():
            assert loads_per_lpat[lpat] == loading.nodal[lpat].count


