"""
tests/test_model.py
===================
Tests for DOFManager and Model.

Covers: add(), remove(), without(), finalize(), get(), find(),
        DOF assignment, reference validation, serialization.
"""

import pytest
import numpy as np

from node import Node, Support, NodalMass
from section import ElasticSection
from element import EulerBernoulliBeam
from material import ElasticMaterial
from model import Model, DOFManager


# ---------------------------------------------------------------------------
# Helpers — build a minimal valid model
# ---------------------------------------------------------------------------

def make_material():
    return ElasticMaterial(id="steel", E=210_000.0, rho=7.85e-3)


def make_section():
    return ElasticSection(id="sec", material_id="steel", A=100.0, I=10_000.0)


def make_simple_model():
    """Two nodes, one section, one element, one fixed support."""
    m = Model()
    m.add(Node(id="A", x=0.0, y=0.0))
    m.add(Node(id="B", x=5.0, y=0.0))
    m.add(make_section())
    m.add(EulerBernoulliBeam(id="e1", node_ids=("A", "B"), section_id="sec"))
    m.add(Support(id="s1", node_id="A", ux=True, uy=True, rz=True))
    return m


# ---------------------------------------------------------------------------
# Model — add()
# ---------------------------------------------------------------------------

class TestModelAdd:

    def test_add_node(self):
        m = Model()
        n = m.add(Node(id="A", x=0.0, y=0.0))
        assert "A" in m.nodes
        assert n.id == "A"

    def test_add_section(self):
        m = Model()
        m.add(make_section())
        assert "sec" in m.sections

    def test_add_element_requires_nodes_first(self):
        m = Model()
        m.add(make_section())
        with pytest.raises(ValueError, match="node"):
            m.add(EulerBernoulliBeam(id="e1", node_ids=("A", "B"), section_id="sec"))

    def test_add_element_requires_section_first(self):
        m = Model()
        m.add(Node(id="A", x=0.0, y=0.0))
        m.add(Node(id="B", x=5.0, y=0.0))
        with pytest.raises(ValueError, match="section"):
            m.add(EulerBernoulliBeam(id="e1", node_ids=("A", "B"), section_id="sec"))

    def test_add_element_valid(self):
        m = make_simple_model()
        assert "e1" in m.elements

    def test_add_support_requires_node(self):
        m = Model()
        with pytest.raises(ValueError, match="node"):
            m.add(Support(id="s1", node_id="X", ux=True))

    def test_add_nodal_mass_requires_node(self):
        m = Model()
        with pytest.raises(ValueError, match="node"):
            m.add(NodalMass(id="m1", node_id="X", mx=5.0, my=5.0, I_theta=1.0))

    def test_duplicate_node_id_raises(self):
        m = Model()
        m.add(Node(id="A", x=0.0, y=0.0))
        with pytest.raises(ValueError, match="already exists"):
            m.add(Node(id="A", x=1.0, y=0.0))

    def test_auto_id_node(self):
        m = Model()
        n = m.add(Node(id=None, x=0.0, y=0.0))
        assert n.id == 1
        n2 = m.add(Node(id=None, x=1.0, y=0.0))
        assert n2.id == 2

    def test_unsupported_type_raises(self):
        m = Model()
        with pytest.raises(TypeError, match="does not accept"):
            m.add("not_a_structural_object")

    def test_add_sets_dirty(self):
        m = make_simple_model()
        m.finalize()
        assert not m._dirty
        m.add(Node(id="C", x=10.0, y=0.0))
        assert m._dirty


# ---------------------------------------------------------------------------
# Model — remove()
# ---------------------------------------------------------------------------

class TestModelRemove:

    def test_remove_element(self):
        m = make_simple_model()
        m.remove("e1")
        assert "e1" not in m.elements

    def test_remove_node_referenced_by_element_raises(self):
        m = make_simple_model()
        with pytest.raises(ValueError, match="referenced by element"):
            m.remove("A")

    def test_remove_node_referenced_by_support_raises(self):
        m = make_simple_model()
        m.remove("e1")   # remove element first
        with pytest.raises(ValueError, match="referenced by support"):
            m.remove("A")

    def test_remove_node_free(self):
        m = make_simple_model()
        m.add(Node(id="C", x=10.0, y=0.0))
        m.remove("C")
        assert "C" not in m.nodes

    def test_remove_section_referenced_by_element_raises(self):
        m = make_simple_model()
        with pytest.raises(ValueError, match="referenced by element"):
            m.remove("sec")

    def test_remove_support(self):
        m = make_simple_model()
        m.remove("s1")
        assert "s1" not in m.supports

    def test_remove_unknown_id_raises(self):
        m = Model()
        with pytest.raises(KeyError):
            m.remove("nonexistent")


# ---------------------------------------------------------------------------
# Model — without()
# ---------------------------------------------------------------------------

class TestModelWithout:

    def test_without_returns_new_model(self):
        m = make_simple_model()
        m2 = m.without("e1")
        assert m2 is not m

    def test_without_removes_element(self):
        m = make_simple_model()
        m2 = m.without("e1")
        assert "e1" not in m2.elements

    def test_without_preserves_original(self):
        m = make_simple_model()
        m.without("e1")
        assert "e1" in m.elements

    def test_without_shares_nodes(self):
        m = make_simple_model()
        m2 = m.without("e1")
        assert m2.nodes is not m.nodes
        assert m2.nodes["A"] is m.nodes["A"]   # same object, different dict

    def test_without_unknown_element_raises(self):
        m = make_simple_model()
        with pytest.raises(KeyError):
            m.without("nonexistent")


# ---------------------------------------------------------------------------
# Model — finalize() and DOFManager
# ---------------------------------------------------------------------------

class TestFinalize:

    def test_finalize_returns_dof_manager(self):
        m = make_simple_model()
        dof_mgr = m.finalize()
        assert isinstance(dof_mgr, DOFManager)

    def test_finalize_idempotent(self):
        m = make_simple_model()
        dof1 = m.finalize()
        dof2 = m.finalize()
        assert dof1 is dof2   # same object returned

    def test_finalize_after_add_rebuilds(self):
        m = make_simple_model()
        dof1 = m.finalize()
        m.add(Node(id="C", x=10.0, y=0.0))
        dof2 = m.finalize()
        assert dof1 is not dof2

    def test_dof_count(self):
        m = make_simple_model()
        dof_mgr = m.finalize()
        assert dof_mgr.n_dofs == 6   # 2 nodes * 3 DOFs

    def test_node_dofs_assigned(self):
        m = make_simple_model()
        m.finalize()
        assert m.nodes["A"].dofs == [0, 1, 2]
        assert m.nodes["B"].dofs == [3, 4, 5]

    def test_elem_dof_map(self):
        m = make_simple_model()
        dof_mgr = m.finalize()
        assert dof_mgr.get_elem_dofs("e1") == [0, 1, 2, 3, 4, 5]

    def test_get_node_dofs_before_finalize_raises(self):
        dof_mgr = DOFManager()
        with pytest.raises(KeyError, match="finalize"):
            dof_mgr.get_node_dofs("A")

    def test_three_node_model_dof_count(self):
        m = Model()
        m.add(Node(id="A", x=0.0, y=0.0))
        m.add(Node(id="B", x=5.0, y=0.0))
        m.add(Node(id="C", x=10.0, y=0.0))
        m.add(make_section())
        m.add(EulerBernoulliBeam(id="e1", node_ids=("A", "B"), section_id="sec"))
        m.add(EulerBernoulliBeam(id="e2", node_ids=("B", "C"), section_id="sec"))
        dof_mgr = m.finalize()
        assert dof_mgr.n_dofs == 9


# ---------------------------------------------------------------------------
# DOFManager — build_U_from_state / extract_state_from_U
# ---------------------------------------------------------------------------

class TestDOFManagerStateConversion:

    def setup_method(self):
        self.m = make_simple_model()
        self.dof_mgr = self.m.finalize()

    def test_build_U_from_state(self):
        nodal_disp = {
            "A": np.array([0.0, 0.0, 0.0]),
            "B": np.array([0.01, 0.02, 0.001]),
        }
        U = self.dof_mgr.build_U_from_state(nodal_disp)
        assert U.shape == (6,)
        np.testing.assert_allclose(U[3:6], [0.01, 0.02, 0.001])

    def test_extract_state_from_U(self):
        U = np.array([0.0, 0.0, 0.0, 0.01, 0.02, 0.001])
        state = self.dof_mgr.extract_state_from_U(U)
        assert "A" in state
        assert "B" in state
        np.testing.assert_allclose(state["B"], [0.01, 0.02, 0.001])

    def test_round_trip(self):
        nodal_disp = {
            "A": np.array([0.0, 0.0, 0.0]),
            "B": np.array([1.0, 2.0, 3.0]),
        }
        U = self.dof_mgr.build_U_from_state(nodal_disp)
        recovered = self.dof_mgr.extract_state_from_U(U)
        np.testing.assert_allclose(recovered["B"], nodal_disp["B"])

    def test_build_U_ignores_unknown_nodes(self):
        """Nodes not in the current model are silently ignored (staged analysis)."""
        nodal_disp = {
            "A": np.array([0.0, 0.0, 0.0]),
            "B": np.array([1.0, 2.0, 3.0]),
            "GHOST": np.array([9.0, 9.0, 9.0]),  # not in model
        }
        U = self.dof_mgr.build_U_from_state(nodal_disp)
        assert U.shape == (6,)


# ---------------------------------------------------------------------------
# Model — get(), get_all(), find()
# ---------------------------------------------------------------------------

class TestModelAccess:

    def setup_method(self):
        self.m = make_simple_model()

    def test_get_node(self):
        n = self.m.get("A")
        assert isinstance(n, Node)
        assert n.id == "A"

    def test_get_element(self):
        e = self.m.get("e1")
        assert isinstance(e, EulerBernoulliBeam)

    def test_get_unknown_raises(self):
        with pytest.raises(KeyError):
            self.m.get("nonexistent")

    def test_get_all_nodes(self):
        nodes = self.m.get_all(Node)
        assert set(nodes.keys()) == {"A", "B"}

    def test_get_all_unknown_type_raises(self):
        with pytest.raises(TypeError):
            self.m.get_all(str)

    def test_find_by_predicate(self):
        result = self.m.find(lambda obj: isinstance(obj, Node) and obj.x > 2.0)
        assert "B" in result
        assert "A" not in result

    def test_find_no_match(self):
        result = self.m.find(lambda obj: False)
        assert result == {}


# ---------------------------------------------------------------------------
# Model — serialization
# ---------------------------------------------------------------------------

class TestModelSerialization:

    def test_to_dict_keys(self):
        m = make_simple_model()
        d = m.to_dict()
        assert "nodes" in d
        assert "elements" in d
        assert "sections" in d
        assert "supports" in d

    def test_round_trip(self):
        m = make_simple_model()
        d = m.to_dict()
        m2 = Model.from_dict(d)
        assert set(m2.nodes.keys()) == set(m.nodes.keys())
        assert set(m2.elements.keys()) == set(m.elements.keys())
        assert set(m2.sections.keys()) == set(m.sections.keys())
        assert set(m2.supports.keys()) == set(m.supports.keys())

    def test_round_trip_node_coordinates(self):
        m = make_simple_model()
        m2 = Model.from_dict(m.to_dict())
        assert m2.nodes["B"].x == 5.0
        assert m2.nodes["B"].y == 0.0

    def test_round_trip_finalize(self):
        m = make_simple_model()
        m2 = Model.from_dict(m.to_dict())
        dof_mgr = m2.finalize()
        assert dof_mgr.n_dofs == 6


# ---------------------------------------------------------------------------
# Model — summary and repr
# ---------------------------------------------------------------------------

class TestModelRepr:

    def test_summary_contains_counts(self):
        m = make_simple_model()
        s = m.summary()
        assert "2" in s   # 2 nodes
        assert "1" in s   # 1 element

    def test_repr(self):
        m = make_simple_model()
        r = repr(m)
        assert "Model" in r
