"""
tests/test_load_case.py
=======================
Tests for LoadCase and DistributedLoad.
"""

import pytest
import numpy as np
from load_case import LoadCase, DistributedLoad


# ---------------------------------------------------------------------------
# DistributedLoad
# ---------------------------------------------------------------------------

class TestDistributedLoad:

    def test_default_attributes(self):
        dl = DistributedLoad(q=-5.0)
        assert dl.q == -5.0
        assert dl.direction == "y"
        assert dl.reference == "global"

    def test_invalid_direction_raises(self):
        with pytest.raises(ValueError, match="direction"):
            DistributedLoad(q=1.0, direction="z")

    def test_invalid_reference_raises(self):
        with pytest.raises(ValueError, match="reference"):
            DistributedLoad(q=1.0, reference="body")

    def test_scaled(self):
        dl = DistributedLoad(q=-5.0, direction="y", reference="global")
        dl2 = dl.scaled(2.0)
        assert dl2.q == pytest.approx(-10.0)
        assert dl2.direction == "y"
        assert dl is not dl2

    def test_scale_does_not_modify_original(self):
        dl = DistributedLoad(q=-5.0)
        dl.scaled(3.0)
        assert dl.q == -5.0

    def test_serialization_round_trip(self):
        dl = DistributedLoad(q=-3.0, direction="x", reference="local")
        dl2 = DistributedLoad.from_dict(dl.to_dict())
        assert dl2.q == dl.q
        assert dl2.direction == dl.direction
        assert dl2.reference == dl.reference


# ---------------------------------------------------------------------------
# LoadCase — construction and update interface
# ---------------------------------------------------------------------------

class TestLoadCaseConstruction:

    def test_empty_load_case(self):
        lc = LoadCase(id="lc1")
        assert lc.id == "lc1"
        assert lc.nodal_loads == {}
        assert lc.distributed_loads == {}

    def test_update_nodal_load(self):
        lc = LoadCase(id="lc1")
        lc.update_nodal_load("B", [0.0, -10.0, 0.0])
        np.testing.assert_allclose(lc.nodal_loads["B"], [0.0, -10.0, 0.0])

    def test_update_nodal_load_replaces_existing(self):
        lc = LoadCase(id="lc1")
        lc.update_nodal_load("B", [0.0, -10.0, 0.0])
        lc.update_nodal_load("B", [5.0, -20.0, 1.0])
        np.testing.assert_allclose(lc.nodal_loads["B"], [5.0, -20.0, 1.0])

    def test_update_nodal_load_wrong_shape_raises(self):
        lc = LoadCase(id="lc1")
        with pytest.raises(ValueError, match="3-component"):
            lc.update_nodal_load("B", [0.0, -10.0])

    def test_update_nodal_load_returns_self(self):
        lc = LoadCase(id="lc1")
        result = lc.update_nodal_load("B", [0.0, -10.0, 0.0])
        assert result is lc

    def test_update_distributed_load(self):
        lc = LoadCase(id="lc1")
        lc.update_distributed_load("e1", q=-5.0, reference="global")
        assert lc.distributed_loads["e1"].q == -5.0

    def test_update_distributed_load_returns_self(self):
        lc = LoadCase(id="lc1")
        result = lc.update_distributed_load("e1", q=-5.0)
        assert result is lc

    def test_chaining(self):
        lc = (
            LoadCase(id="wind")
            .update_nodal_load("A", [10.0, 0.0, 0.0])
            .update_nodal_load("B", [5.0, 0.0, 0.0])
            .update_distributed_load("e1", q=-2.0)
        )
        assert len(lc.nodal_loads) == 2
        assert len(lc.distributed_loads) == 1

    def test_remove_nodal_load(self):
        lc = LoadCase(id="lc1")
        lc.update_nodal_load("B", [0.0, -10.0, 0.0])
        lc.remove_nodal_load("B")
        assert "B" not in lc.nodal_loads

    def test_remove_nodal_load_noop_if_missing(self):
        lc = LoadCase(id="lc1")
        lc.remove_nodal_load("nonexistent")   # must not raise

    def test_remove_distributed_load(self):
        lc = LoadCase(id="lc1")
        lc.update_distributed_load("e1", q=-5.0)
        lc.remove_distributed_load("e1")
        assert "e1" not in lc.distributed_loads

    def test_clear(self):
        lc = LoadCase(id="lc1")
        lc.update_nodal_load("B", [0.0, -10.0, 0.0])
        lc.update_distributed_load("e1", q=-5.0)
        lc.clear()
        assert lc.nodal_loads == {}
        assert lc.distributed_loads == {}

    def test_clear_returns_self(self):
        lc = LoadCase(id="lc1")
        result = lc.clear()
        assert result is lc


# ---------------------------------------------------------------------------
# LoadCase — scale()
# ---------------------------------------------------------------------------

class TestLoadCaseScale:

    def test_scale_nodal_loads(self):
        lc = LoadCase(id="lc1")
        lc.update_nodal_load("B", [0.0, -10.0, 0.0])
        lc2 = lc.scale(2.0)
        np.testing.assert_allclose(lc2.nodal_loads["B"], [0.0, -20.0, 0.0])

    def test_scale_distributed_loads(self):
        lc = LoadCase(id="lc1")
        lc.update_distributed_load("e1", q=-5.0)
        lc2 = lc.scale(3.0)
        assert lc2.distributed_loads["e1"].q == pytest.approx(-15.0)

    def test_scale_does_not_modify_original(self):
        lc = LoadCase(id="lc1")
        lc.update_nodal_load("B", [0.0, -10.0, 0.0])
        lc.scale(5.0)
        np.testing.assert_allclose(lc.nodal_loads["B"], [0.0, -10.0, 0.0])

    def test_scale_returns_new_load_case(self):
        lc = LoadCase(id="lc1")
        lc2 = lc.scale(2.0)
        assert lc2 is not lc

    def test_scale_by_zero(self):
        lc = LoadCase(id="lc1")
        lc.update_nodal_load("B", [0.0, -10.0, 0.0])
        lc2 = lc.scale(0.0)
        np.testing.assert_allclose(lc2.nodal_loads["B"], [0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# LoadCase — combine()
# ---------------------------------------------------------------------------

class TestLoadCaseCombine:

    def test_combine_nodal_loads_same_node(self):
        lc1 = LoadCase(id="a").update_nodal_load("B", [10.0, 0.0, 0.0])
        lc2 = LoadCase(id="b").update_nodal_load("B", [0.0, -5.0, 0.0])
        lc = lc1.combine(lc2)
        np.testing.assert_allclose(lc.nodal_loads["B"], [10.0, -5.0, 0.0])

    def test_combine_nodal_loads_different_nodes(self):
        lc1 = LoadCase(id="a").update_nodal_load("A", [1.0, 0.0, 0.0])
        lc2 = LoadCase(id="b").update_nodal_load("B", [0.0, -2.0, 0.0])
        lc = lc1.combine(lc2)
        assert "A" in lc.nodal_loads
        assert "B" in lc.nodal_loads

    def test_combine_with_factors(self):
        lc1 = LoadCase(id="a").update_nodal_load("B", [10.0, 0.0, 0.0])
        lc2 = LoadCase(id="b").update_nodal_load("B", [10.0, 0.0, 0.0])
        lc = lc1.combine(lc2, factor_self=1.2, factor_other=0.5)
        np.testing.assert_allclose(lc.nodal_loads["B"], [17.0, 0.0, 0.0])

    def test_combine_distributed_incompatible_raises(self):
        lc1 = LoadCase(id="a")
        lc1.update_distributed_load("e1", q=-5.0, direction="y", reference="global")
        lc2 = LoadCase(id="b")
        lc2.update_distributed_load("e1", q=-3.0, direction="x", reference="global")
        with pytest.raises(ValueError, match="incompatible"):
            lc1.combine(lc2)

    def test_combine_does_not_modify_originals(self):
        lc1 = LoadCase(id="a").update_nodal_load("B", [10.0, 0.0, 0.0])
        lc2 = LoadCase(id="b").update_nodal_load("B", [5.0, 0.0, 0.0])
        lc1.combine(lc2)
        np.testing.assert_allclose(lc1.nodal_loads["B"], [10.0, 0.0, 0.0])
        np.testing.assert_allclose(lc2.nodal_loads["B"], [5.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# LoadCase — validate()
# ---------------------------------------------------------------------------

class TestLoadCaseValidate:

    def test_validate_passes_with_valid_ids(self):
        lc = LoadCase(id="lc1")
        lc.update_nodal_load("B", [0.0, -10.0, 0.0])
        lc.update_distributed_load("e1", q=-5.0)
        lc.validate(node_ids={"A", "B"}, elem_ids={"e1", "e2"})   # no exception

    def test_validate_unknown_node_raises(self):
        lc = LoadCase(id="lc1")
        lc.update_nodal_load("X", [0.0, -10.0, 0.0])
        with pytest.raises(ValueError, match="node"):
            lc.validate(node_ids={"A", "B"}, elem_ids=set())

    def test_validate_unknown_element_raises(self):
        lc = LoadCase(id="lc1")
        lc.update_distributed_load("e99", q=-5.0)
        with pytest.raises(ValueError, match="element"):
            lc.validate(node_ids=set(), elem_ids={"e1"})


# ---------------------------------------------------------------------------
# LoadCase — serialization
# ---------------------------------------------------------------------------

class TestLoadCaseSerialization:

    def test_round_trip(self):
        lc = LoadCase(id="wind")
        lc.update_nodal_load("B", [5.0, -10.0, 2.0])
        lc.update_distributed_load("e1", q=-3.0, reference="local")
        d = lc.to_dict()
        lc2 = LoadCase.from_dict(d)
        assert lc2.id == lc.id
        np.testing.assert_allclose(
            lc2.nodal_loads["B"], lc.nodal_loads["B"]
        )
        assert lc2.distributed_loads["e1"].q == lc.distributed_loads["e1"].q
        assert lc2.distributed_loads["e1"].reference == "local"

    def test_type_field_in_dict(self):
        lc = LoadCase(id="lc1")
        assert lc.to_dict()["type"] == "LoadCase"

    def test_empty_load_case_round_trip(self):
        lc = LoadCase(id="empty")
        lc2 = LoadCase.from_dict(lc.to_dict())
        assert lc2.nodal_loads == {}
        assert lc2.distributed_loads == {}
