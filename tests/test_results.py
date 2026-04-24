"""
tests/test_results.py
=====================
Tests for StoreOptions, SolverConfig, SolverCallbacks,
ModelState, StepResult, NodeResult, ElementResult, Results.
"""

import pytest
import numpy as np

from node import Node, Support
from section import ElasticSection
from element import EulerBernoulliBeam
from material import ElasticMaterial
from model import Model
from results import (
    StoreOptions,
    SolverConfig,
    SolverCallbacks,
    ModelState,
    NodeResult,
    ElementResult,
    StepResult,
    Results,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_dof_mgr():
    m = Model()
    m.add(Node(id="A", x=0.0, y=0.0))
    m.add(Node(id="B", x=5.0, y=0.0))
    m.add(ElasticSection(id="sec", material_id="steel", A=100.0, I=10_000.0))
    m.add(EulerBernoulliBeam(id="e1", node_ids=("A", "B"), section_id="sec"))
    m.add(Support(id="fix", node_id="A", ux=True, uy=True, rz=True))
    return m.finalize()


def make_step_result(step=1, load_factor=1.0, U=None):
    n = 6
    U = np.zeros(n) if U is None else np.asarray(U)
    reactions = {"A": np.array([-5.0, -10.0, -25.0])}
    elem_results = {
        "e1": ElementResult("e1", forces=np.ones(6) * step)
    }
    return StepResult(
        step=step,
        load_factor=load_factor,
        U=U,
        reactions=reactions,
        element_results=elem_results,
    )


# ---------------------------------------------------------------------------
# StoreOptions
# ---------------------------------------------------------------------------

class TestStoreOptions:

    def test_defaults(self):
        s = StoreOptions()
        assert s.iterations is False
        assert s.gauss_points is False
        assert s.fibers is False
        assert s.elements is None

    def test_stores_element_none_means_all(self):
        s = StoreOptions(elements=None)
        assert s.stores_element("e1")
        assert s.stores_element("any_id")

    def test_stores_element_explicit_list(self):
        s = StoreOptions(elements=["e1", "e3"])
        assert s.stores_element("e1")
        assert not s.stores_element("e2")

    def test_stores_element_empty_list(self):
        s = StoreOptions(elements=[])
        assert not s.stores_element("e1")


# ---------------------------------------------------------------------------
# SolverConfig
# ---------------------------------------------------------------------------

class TestSolverConfig:

    def test_defaults(self):
        c = SolverConfig()
        assert c.check_equilibrium is True
        assert c.max_iter == 50
        assert c.linear_solver == "spsolve"
        assert c.verbose is False
        assert c.checkpoint is None

    def test_custom_values(self):
        c = SolverConfig(max_iter=20, verbose=True, checkpoint="out.json")
        assert c.max_iter == 20
        assert c.verbose is True
        assert c.checkpoint == "out.json"

    def test_store_options_accessible(self):
        c = SolverConfig(store=StoreOptions(iterations=True))
        assert c.store.iterations is True

    def test_callbacks_accessible(self):
        called = []
        cb = SolverCallbacks(on_step=lambda r: called.append(r.step))
        c = SolverConfig(callbacks=cb)
        sr = make_step_result(step=3)
        c.callbacks.on_step(sr)
        assert called == [3]


# ---------------------------------------------------------------------------
# NodeResult
# ---------------------------------------------------------------------------

class TestNodeResult:

    def test_components(self):
        u = np.array([0.001, 0.005, 0.002])
        nr = NodeResult("A", u)
        assert nr.ux == pytest.approx(0.001)
        assert nr.uy == pytest.approx(0.005)
        assert nr.rz == pytest.approx(0.002)

    def test_u_returns_copy(self):
        u = np.array([1.0, 2.0, 3.0])
        nr = NodeResult("A", u)
        u_copy = nr.u
        u_copy[0] = 99.0
        assert nr.ux == pytest.approx(1.0)   # original unchanged

    def test_repr(self):
        nr = NodeResult("A", np.zeros(3))
        assert "NodeResult" in repr(nr)
        assert "A" in repr(nr)


# ---------------------------------------------------------------------------
# ElementResult
# ---------------------------------------------------------------------------

class TestElementResult:

    def test_get_forces_global(self):
        forces = np.ones(6) * 5.0
        er = ElementResult("e1", forces)
        np.testing.assert_allclose(er.get_forces(ref="global"), forces)

    def test_get_forces_local_raises(self):
        er = ElementResult("e1", np.ones(6))
        with pytest.raises(NotImplementedError):
            er.get_forces(ref="local")

    def test_element_state_none_by_default(self):
        er = ElementResult("e1", np.ones(6))
        assert er.element_state is None

    def test_repr(self):
        er = ElementResult("e1", np.ones(6))
        assert "ElementResult" in repr(er)


# ---------------------------------------------------------------------------
# StepResult
# ---------------------------------------------------------------------------

class TestStepResult:

    def test_attributes(self):
        sr = make_step_result(step=2, load_factor=0.5)
        assert sr.step == 2
        assert sr.load_factor == pytest.approx(0.5)

    def test_node_access(self):
        U = np.array([0.0, 0.0, 0.0, 0.001, 0.005, 0.002])
        sr = make_step_result(U=U)
        dof_mgr = make_dof_mgr()
        nr = sr.node("B", dof_mgr)
        assert nr.uy == pytest.approx(0.005)

    def test_empty_iterations_by_default(self):
        sr = make_step_result()
        assert sr.iterations == []

    def test_reactions_dict(self):
        sr = make_step_result()
        assert "A" in sr.reactions
        assert sr.reactions["A"].shape == (3,)

    def test_repr(self):
        sr = make_step_result(step=1)
        assert "StepResult" in repr(sr)


# ---------------------------------------------------------------------------
# Results — building and access
# ---------------------------------------------------------------------------

class TestResults:

    def setup_method(self):
        self.dof_mgr = make_dof_mgr()
        self.results = Results(dof_mgr=self.dof_mgr)

    def _add_steps(self, n=3):
        for i in range(1, n + 1):
            U = np.zeros(6)
            U[4] = i * 0.001   # tip displacement grows
            self.results.add_step(make_step_result(
                step=i, load_factor=i * 0.25, U=U
            ))

    def test_add_step_and_count(self):
        self._add_steps(3)
        assert self.results.n_steps == 3

    def test_getitem_step(self):
        self._add_steps(2)
        proxy = self.results["step_1"]
        assert proxy.step == 1

    def test_getitem_unknown_step_raises(self):
        with pytest.raises(KeyError, match="step_99"):
            self.results["step_99"]

    def test_add_step_after_finalize_raises(self):
        self.results.finalize()
        with pytest.raises(RuntimeError, match="immutable"):
            self.results.add_step(make_step_result())

    def test_last(self):
        self._add_steps(3)
        assert self.results.last.step == 3

    def test_last_empty_raises(self):
        with pytest.raises(RuntimeError, match="empty"):
            self.results.last

    def test_load_factors(self):
        self._add_steps(3)
        lf = self.results.load_factors
        assert lf == pytest.approx([0.25, 0.50, 0.75])

    def test_node_history(self):
        self._add_steps(3)
        uy_hist = self.results.node_history("B", "uy")
        assert uy_hist == pytest.approx([0.001, 0.002, 0.003])

    def test_node_history_invalid_component_raises(self):
        self._add_steps(1)
        with pytest.raises(ValueError, match="component"):
            self.results.node_history("B", "uz")

    def test_node_history_without_dof_mgr_raises(self):
        r = Results()   # no dof_mgr
        r.add_step(make_step_result())
        with pytest.raises(RuntimeError, match="DOFManager"):
            r.node_history("B", "uy")

    def test_reaction_history(self):
        self._add_steps(2)
        ry_hist = self.results.reaction_history("A", "Ry")
        assert len(ry_hist) == 2
        assert all(isinstance(v, float) for v in ry_hist)

    def test_reaction_history_invalid_component_raises(self):
        self._add_steps(1)
        with pytest.raises(ValueError, match="component"):
            self.results.reaction_history("A", "Fx")

    def test_steps_property(self):
        self._add_steps(3)
        steps = self.results.steps
        assert len(steps) == 3
        assert steps[0].step == 1

    def test_summary(self):
        self._add_steps(2)
        s = self.results.summary()
        assert "2 steps" in s


# ---------------------------------------------------------------------------
# Results — hierarchical access syntax
# ---------------------------------------------------------------------------

class TestResultsHierarchicalAccess:

    def setup_method(self):
        self.dof_mgr = make_dof_mgr()
        self.results = Results(dof_mgr=self.dof_mgr)
        U = np.array([0.0, 0.0, 0.0, 0.001, 0.005, 0.002])
        self.results.add_step(make_step_result(step=1, load_factor=1.0, U=U))

    def test_node_access_via_proxy(self):
        nr = self.results["step_1"]["A"]
        assert isinstance(nr, NodeResult)
        assert nr.ux == pytest.approx(0.0)
        assert nr.uy == pytest.approx(0.0)

    def test_node_B_via_proxy(self):
        nr = self.results["step_1"]["B"]
        assert nr.uy == pytest.approx(0.005)

    def test_element_access_via_proxy(self):
        er = self.results["step_1"]["e1"]
        assert isinstance(er, ElementResult)

    def test_unknown_id_raises(self):
        with pytest.raises(KeyError):
            self.results["step_1"]["nonexistent"]

    def test_load_factor_via_proxy(self):
        assert self.results["step_1"].load_factor == pytest.approx(1.0)

    def test_reactions_via_proxy(self):
        rxn = self.results["step_1"].reactions["A"]
        assert rxn.shape == (3,)

    def test_full_access_chain(self):
        """results['step_1']['B'].uy — the intended ergonomic access."""
        uy = self.results["step_1"]["B"].uy
        assert uy == pytest.approx(0.005)


# ---------------------------------------------------------------------------
# ModelState
# ---------------------------------------------------------------------------

class TestModelState:

    def test_from_step_result(self):
        dof_mgr = make_dof_mgr()
        U = np.array([0.0, 0.0, 0.0, 0.001, 0.005, 0.002])
        sr = make_step_result(step=2, load_factor=0.5, U=U)
        state = ModelState.from_step_result(sr, dof_mgr)
        assert state.step == 2
        assert state.load_factor == pytest.approx(0.5)
        np.testing.assert_allclose(
            state.nodal_displacements["B"], [0.001, 0.005, 0.002]
        )

    def test_to_dict(self):
        state = ModelState()
        state.load_factor = 1.0
        state.step = 3
        state.nodal_displacements["A"] = np.zeros(3)
        d = state.to_dict()
        assert d["load_factor"] == 1.0
        assert d["step"] == 3
        assert "A" in d["nodal_displacements"]

    def test_repr(self):
        state = ModelState()
        assert "ModelState" in repr(state)
