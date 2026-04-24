"""
tests/test_load_control.py
==========================
Tests for LoadControl (ABC), PlainLoadControl, DisplacementControl.

Linear Phase 0 tests: verify lambda sequence and initialize/update cycle.
Nonlinear preparation tests: verify constraint_equation() interface,
DOF resolution, and accumulated displacement tracking.
"""

import pytest
import numpy as np

from node import Node, Support
from section import ElasticSection
from element import EulerBernoulliBeam
from material import ElasticMaterial
from model import Model, DOFManager
from convergence import IterationData
from load_control import PlainLoadControl, DisplacementControl


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_dof_mgr():
    """Minimal two-node model to get a real DOFManager."""
    m = Model()
    m.add(Node(id="A", x=0.0, y=0.0))
    m.add(Node(id="B", x=5.0, y=0.0))
    m.add(ElasticSection(id="sec", material_id="steel", A=100.0, I=10_000.0))
    m.add(EulerBernoulliBeam(id="e1", node_ids=("A", "B"), section_id="sec"))
    m.add(Support(id="fix", node_id="A", ux=True, uy=True, rz=True))
    return m.finalize()


def make_iteration_data(U=None, load_factor=1.0):
    n = 6
    return IterationData(
        iteration=0,
        residual=np.zeros(n),
        delta_U=np.zeros(n),
        U_trial=np.zeros(n) if U is None else np.asarray(U),
        F_external=np.ones(n),
        load_factor=load_factor,
    )


# ---------------------------------------------------------------------------
# PlainLoadControl
# ---------------------------------------------------------------------------

class TestPlainLoadControl:

    def test_default_parameters(self):
        c = PlainLoadControl()
        assert c.delta_lambda == 1.0
        assert c.n_steps == 1

    def test_lambda_single_step(self):
        c = PlainLoadControl(delta_lambda=1.0)
        dof_mgr = make_dof_mgr()
        c.initialize(dof_mgr)
        assert c.get_lambda(step=1) == pytest.approx(1.0)

    def test_lambda_sequence(self):
        c = PlainLoadControl(delta_lambda=0.25, n_steps=4)
        dof_mgr = make_dof_mgr()
        c.initialize(dof_mgr)
        lambdas = [c.get_lambda(step=i) for i in range(1, 5)]
        assert lambdas == pytest.approx([0.25, 0.50, 0.75, 1.00])

    def test_initialize_resets_lambda(self):
        c = PlainLoadControl(delta_lambda=0.5)
        dof_mgr = make_dof_mgr()
        c.initialize(dof_mgr)
        c.get_lambda(step=3)
        c.initialize(dof_mgr)   # reset
        assert c._current_lambda == 0.0

    def test_update_is_noop(self):
        c = PlainLoadControl(delta_lambda=1.0)
        dof_mgr = make_dof_mgr()
        c.initialize(dof_mgr)
        c.get_lambda(step=1)
        data = make_iteration_data(load_factor=1.0)
        c.update(data)           # must not raise or change lambda unexpectedly
        assert c._current_lambda == pytest.approx(1.0)

    def test_constraint_equation_returns_none(self):
        c = PlainLoadControl(delta_lambda=1.0)
        dof_mgr = make_dof_mgr()
        c.initialize(dof_mgr)
        data = make_iteration_data()
        assert c.constraint_equation(data) is None

    def test_negative_delta_lambda_raises(self):
        with pytest.raises(ValueError, match="positive"):
            PlainLoadControl(delta_lambda=-0.1)

    def test_zero_delta_lambda_raises(self):
        with pytest.raises(ValueError, match="positive"):
            PlainLoadControl(delta_lambda=0.0)

    def test_zero_n_steps_raises(self):
        with pytest.raises(ValueError, match="n_steps"):
            PlainLoadControl(delta_lambda=1.0, n_steps=0)

    def test_serialization_round_trip(self):
        c = PlainLoadControl(delta_lambda=0.1, n_steps=10)
        d = c.to_dict()
        assert d["type"] == "PlainLoadControl"
        c2 = PlainLoadControl.from_dict(d)
        assert c2.delta_lambda == c.delta_lambda
        assert c2.n_steps == c.n_steps

    def test_repr(self):
        c = PlainLoadControl(delta_lambda=0.5, n_steps=2)
        assert "PlainLoadControl" in repr(c)
        assert "0.5" in repr(c)

    def test_linear_analysis_pattern(self):
        """
        Typical linear Phase 0 usage:
        single step, delta_lambda=1.0, lambda=1.0 after step 1.
        """
        c = PlainLoadControl(delta_lambda=1.0, n_steps=1)
        dof_mgr = make_dof_mgr()
        c.initialize(dof_mgr)
        lam = c.get_lambda(step=1)
        assert lam == pytest.approx(1.0)
        data = make_iteration_data(load_factor=1.0)
        c.update(data)
        # no further steps needed for linear analysis
        assert c.constraint_equation(data) is None

    def test_nonlinear_multi_step_pattern(self):
        """
        Simulated NL usage: 5 equal steps, lambda ramps from 0.2 to 1.0.
        """
        c = PlainLoadControl(delta_lambda=0.2, n_steps=5)
        dof_mgr = make_dof_mgr()
        c.initialize(dof_mgr)
        for step in range(1, 6):
            lam = c.get_lambda(step)
            assert lam == pytest.approx(step * 0.2, rel=1e-10)
            c.update(make_iteration_data(load_factor=lam))


# ---------------------------------------------------------------------------
# DisplacementControl
# ---------------------------------------------------------------------------

class TestDisplacementControl:

    def test_basic_attributes(self):
        c = DisplacementControl(node_id="B", direction="uy", delta_u=0.001)
        assert c.node_id == "B"
        assert c.direction == "uy"
        assert c.delta_u == 0.001

    def test_invalid_direction_raises(self):
        with pytest.raises(ValueError, match="direction"):
            DisplacementControl(node_id="B", direction="uz", delta_u=0.001)

    def test_zero_n_steps_raises(self):
        with pytest.raises(ValueError, match="n_steps"):
            DisplacementControl(node_id="B", direction="uy",
                                delta_u=0.001, n_steps=0)

    def test_initialize_resolves_controlled_dof(self):
        """After initialize(), controlled_dof must be the global DOF index."""
        c = DisplacementControl(node_id="B", direction="uy", delta_u=0.001)
        dof_mgr = make_dof_mgr()
        c.initialize(dof_mgr)
        # node B has dofs [3,4,5]; uy -> local index 1 -> global 4
        assert c.controlled_dof == 4

    def test_initialize_ux(self):
        c = DisplacementControl(node_id="B", direction="ux", delta_u=0.001)
        dof_mgr = make_dof_mgr()
        c.initialize(dof_mgr)
        assert c.controlled_dof == 3   # node B ux -> dof 3

    def test_initialize_rz(self):
        c = DisplacementControl(node_id="B", direction="rz", delta_u=0.001)
        dof_mgr = make_dof_mgr()
        c.initialize(dof_mgr)
        assert c.controlled_dof == 5   # node B rz -> dof 5

    def test_initialize_resets_accumulated(self):
        c = DisplacementControl(node_id="B", direction="uy", delta_u=0.01)
        dof_mgr = make_dof_mgr()
        c.initialize(dof_mgr)
        c._accumulated_u = 0.05   # simulate some history
        c.initialize(dof_mgr)     # re-initialize
        assert c._accumulated_u == 0.0

    def test_constraint_equation_before_initialize_returns_none(self):
        c = DisplacementControl(node_id="B", direction="uy", delta_u=0.001)
        data = make_iteration_data()
        assert c.constraint_equation(data) is None

    def test_constraint_equation_at_target(self):
        """If U_trial[controlled_dof] = delta_u, constraint = 0."""
        c = DisplacementControl(node_id="B", direction="uy", delta_u=0.005)
        dof_mgr = make_dof_mgr()
        c.initialize(dof_mgr)
        U = np.zeros(6)
        U[4] = 0.005   # exactly at target
        data = make_iteration_data(U=U)
        g = c.constraint_equation(data)
        assert g == pytest.approx(0.0, abs=1e-15)

    def test_constraint_equation_below_target(self):
        """If U_trial[controlled] < target, constraint < 0."""
        c = DisplacementControl(node_id="B", direction="uy", delta_u=0.005)
        dof_mgr = make_dof_mgr()
        c.initialize(dof_mgr)
        U = np.zeros(6)
        U[4] = 0.003   # below target
        data = make_iteration_data(U=U)
        assert c.constraint_equation(data) < 0.0

    def test_update_accumulates_displacement(self):
        """After each update(), accumulated_u grows by delta_u."""
        c = DisplacementControl(node_id="B", direction="uy", delta_u=0.01)
        dof_mgr = make_dof_mgr()
        c.initialize(dof_mgr)
        for step in range(1, 4):
            data = make_iteration_data(load_factor=float(step) * 0.1)
            c.update(data)
            assert c._accumulated_u == pytest.approx(step * 0.01, rel=1e-10)

    def test_constraint_equation_after_update(self):
        """After update(), target advances by delta_u."""
        c = DisplacementControl(node_id="B", direction="uy", delta_u=0.01)
        dof_mgr = make_dof_mgr()
        c.initialize(dof_mgr)
        # first step converged
        c.update(make_iteration_data(load_factor=0.5))
        # new target = 0.01 + 0.01 = 0.02
        U = np.zeros(6)
        U[4] = 0.02
        data = make_iteration_data(U=U)
        assert c.constraint_equation(data) == pytest.approx(0.0, abs=1e-15)

    def test_serialization_round_trip(self):
        c = DisplacementControl(
            node_id="B", direction="uy", delta_u=0.005, n_steps=20
        )
        d = c.to_dict()
        assert d["type"] == "DisplacementControl"
        c2 = DisplacementControl.from_dict(d)
        assert c2.node_id == c.node_id
        assert c2.direction == c.direction
        assert c2.delta_u == c.delta_u
        assert c2.n_steps == c.n_steps

    def test_repr(self):
        c = DisplacementControl(node_id="B", direction="uy", delta_u=0.005)
        assert "DisplacementControl" in repr(c)
        assert "B" in repr(c)
        assert "uy" in repr(c)

    def test_all_valid_directions(self):
        """All three directions must be accepted."""
        dof_mgr = make_dof_mgr()
        for direction, expected_local in [("ux", 3), ("uy", 4), ("rz", 5)]:
            c = DisplacementControl(node_id="B", direction=direction, delta_u=0.001)
            c.initialize(dof_mgr)
            assert c.controlled_dof == expected_local


# ---------------------------------------------------------------------------
# Interchangeability — same interface, different strategies
# ---------------------------------------------------------------------------

class TestLoadControlInterchangeability:
    """
    Verify that PlainLoadControl and DisplacementControl expose the
    same interface and can be used interchangeably by the solver.
    """

    def _simulate_steps(self, control, dof_mgr, n_steps=3):
        """Run a simulated step loop using the control object."""
        control.initialize(dof_mgr)
        lambdas = []
        for step in range(1, n_steps + 1):
            lam = control.get_lambda(step)
            lambdas.append(lam)
            data = make_iteration_data(load_factor=lam)
            control.update(data)
            # constraint_equation must not raise
            control.constraint_equation(data)
        return lambdas

    def test_plain_control_runs(self):
        dof_mgr = make_dof_mgr()
        c = PlainLoadControl(delta_lambda=0.5, n_steps=3)
        lambdas = self._simulate_steps(c, dof_mgr, n_steps=3)
        assert len(lambdas) == 3

    def test_displacement_control_runs(self):
        dof_mgr = make_dof_mgr()
        c = DisplacementControl(node_id="B", direction="uy",
                                delta_u=0.005, n_steps=3)
        lambdas = self._simulate_steps(c, dof_mgr, n_steps=3)
        assert len(lambdas) == 3

    def test_both_expose_constraint_equation(self):
        dof_mgr = make_dof_mgr()
        data = make_iteration_data()
        for c in [
            PlainLoadControl(delta_lambda=1.0),
            DisplacementControl(node_id="B", direction="uy", delta_u=0.01),
        ]:
            c.initialize(dof_mgr)
            result = c.constraint_equation(data)
            # must return None or float — never raise
            assert result is None or isinstance(result, float)
