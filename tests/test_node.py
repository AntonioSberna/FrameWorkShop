"""
tests/test_node.py
==================
Tests for Node, Support, EndRelease, NodalMass.
"""

import pytest
from node import Node, Support, EndRelease, NodalMass


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class TestNode:

    def test_basic_attributes(self):
        n = Node(id="A", x=1.0, y=2.5)
        assert n.id == "A"
        assert n.x == 1.0
        assert n.y == 2.5
        assert n.dofs == []

    def test_dofs_default_empty(self):
        n = Node(id=1, x=0.0, y=0.0)
        assert n.dofs == []

    def test_dofs_can_be_set(self):
        n = Node(id="B", x=0.0, y=0.0, dofs=[0, 1, 2])
        assert n.dofs == [0, 1, 2]

    def test_integer_id(self):
        n = Node(id=42, x=3.0, y=4.0)
        assert n.id == 42

    def test_serialization_round_trip(self):
        n = Node(id="C", x=1.5, y=3.0, dofs=[3, 4, 5])
        n2 = Node.from_dict(n.to_dict())
        assert n2.id == n.id
        assert n2.x == n.x
        assert n2.y == n.y
        assert n2.dofs == n.dofs

    def test_serialization_without_dofs(self):
        n = Node(id="D", x=0.0, y=0.0)
        n2 = Node.from_dict(n.to_dict())
        assert n2.dofs == []

    def test_repr(self):
        n = Node(id="A1", x=1.0, y=2.0)
        assert "Node" in repr(n)
        assert "A1" in repr(n)


# ---------------------------------------------------------------------------
# Support
# ---------------------------------------------------------------------------

class TestSupport:

    def test_default_all_free(self):
        s = Support(id="s1", node_id="A")
        assert s.ux is False
        assert s.uy is False
        assert s.rz is False

    def test_fixed_support(self):
        s = Support(id="fix", node_id="A", ux=True, uy=True, rz=True)
        assert s.is_fully_fixed()

    def test_pin_support(self):
        s = Support(id="pin", node_id="B", ux=True, uy=True, rz=False)
        assert not s.is_fully_fixed()

    def test_constrained_dofs_fixed(self):
        s = Support(id="fix", node_id="A", ux=True, uy=True, rz=True)
        assert s.constrained_dofs() == ["ux", "uy", "rz"]

    def test_constrained_dofs_roller(self):
        s = Support(id="roll", node_id="C", ux=False, uy=True, rz=False)
        assert s.constrained_dofs() == ["uy"]

    def test_constrained_dofs_empty(self):
        s = Support(id="free", node_id="D")
        assert s.constrained_dofs() == []

    def test_serialization_round_trip(self):
        s = Support(id="fix", node_id="A", ux=True, uy=True, rz=True)
        s2 = Support.from_dict(s.to_dict())
        assert s2.id == s.id
        assert s2.node_id == s.node_id
        assert s2.ux == s.ux
        assert s2.uy == s.uy
        assert s2.rz == s.rz

    def test_repr(self):
        s = Support(id="s1", node_id="A", ux=True, uy=True)
        assert "Support" in repr(s)
        assert "A" in repr(s)


# ---------------------------------------------------------------------------
# EndRelease
# ---------------------------------------------------------------------------

class TestEndRelease:

    def test_valid_rz_release(self):
        r = EndRelease(dof="rz", end="j")
        assert r.dof == "rz"
        assert r.end == "j"
        assert r.stiffness == 1e-6

    def test_valid_end_i(self):
        r = EndRelease(dof="rz", end="i")
        assert r.end == "i"

    def test_invalid_dof_raises(self):
        with pytest.raises(ValueError, match="invalid dof"):
            EndRelease(dof="mz", end="j")

    def test_invalid_end_raises(self):
        with pytest.raises(ValueError, match="invalid end"):
            EndRelease(dof="rz", end="k")

    def test_zero_stiffness_raises(self):
        with pytest.raises(ValueError, match="positive"):
            EndRelease(dof="rz", end="j", stiffness=0.0)

    def test_negative_stiffness_raises(self):
        with pytest.raises(ValueError, match="positive"):
            EndRelease(dof="rz", end="j", stiffness=-1.0)

    def test_custom_stiffness(self):
        r = EndRelease(dof="uy", end="i", stiffness=1e-3)
        assert r.stiffness == 1e-3

    def test_serialization_round_trip(self):
        r = EndRelease(dof="rz", end="j", stiffness=1e-4)
        r2 = EndRelease.from_dict(r.to_dict())
        assert r2.dof == r.dof
        assert r2.end == r.end
        assert r2.stiffness == r.stiffness

    def test_repr(self):
        r = EndRelease(dof="rz", end="j")
        assert "EndRelease" in repr(r)
        assert "rz" in repr(r)


# ---------------------------------------------------------------------------
# NodalMass
# ---------------------------------------------------------------------------

class TestNodalMass:

    def test_basic_attributes(self):
        m = NodalMass(id="m1", node_id="A", mx=10.0, my=10.0, I_theta=5.0)
        assert m.mx == 10.0
        assert m.my == 10.0
        assert m.I_theta == 5.0

    def test_zero_mass_allowed(self):
        m = NodalMass(id="m0", node_id="A", mx=0.0, my=0.0, I_theta=0.0)
        assert m.mx == 0.0

    def test_negative_mass_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            NodalMass(id="bad", node_id="A", mx=-1.0, my=0.0, I_theta=0.0)

    def test_negative_I_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            NodalMass(id="bad", node_id="A", mx=0.0, my=0.0, I_theta=-1.0)

    def test_serialization_round_trip(self):
        m = NodalMass(id="m1", node_id="B", mx=5.0, my=5.0, I_theta=2.5)
        m2 = NodalMass.from_dict(m.to_dict())
        assert m2.id == m.id
        assert m2.node_id == m.node_id
        assert m2.mx == m.mx
        assert m2.I_theta == m.I_theta

    def test_repr(self):
        m = NodalMass(id="m1", node_id="A", mx=1.0, my=1.0, I_theta=0.5)
        assert "NodalMass" in repr(m)
        assert "A" in repr(m)
