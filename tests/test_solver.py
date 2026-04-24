"""
tests/test_solver.py
====================
Integration tests for LinearStaticSolver.

Verifies the full pipeline: model -> load case -> solve -> results.
Uses analytical benchmarks for validation.
"""

import pytest
import numpy as np

from node import Node, Support, NodalMass
from section import ElasticSection
from element import EulerBernoulliBeam
from material import ElasticMaterial
from model import Model
from load_case import LoadCase
from load_control import PlainLoadControl
from constraint import PlainConstraintHandler
from convergence import ForceResidual
from results import SolverConfig, StoreOptions, SolverCallbacks
from solver import LinearStaticSolver


# ---------------------------------------------------------------------------
# Shared material and helpers
# ---------------------------------------------------------------------------

E = 210_000.0
A = 100.0
I = 10_000.0
MAT = ElasticMaterial(id="steel", E=E, rho=7.85e-3)
MATERIALS = {"steel": MAT}


def make_cantilever(L=5.0):
    m = Model()
    m.add(Node(id="A", x=0.0, y=0.0))
    m.add(Node(id="B", x=L, y=0.0))
    m.add(ElasticSection(id="sec", material_id="steel", A=A, I=I))
    m.add(EulerBernoulliBeam(id="e1", node_ids=("A", "B"), section_id="sec"))
    m.add(Support(id="fix", node_id="A", ux=True, uy=True, rz=True))
    return m, L


def make_simply_supported(L=6.0):
    m = Model()
    m.add(Node(id="A", x=0.0, y=0.0))
    m.add(Node(id="B", x=L, y=0.0))
    m.add(ElasticSection(id="sec", material_id="steel", A=A, I=I))
    m.add(EulerBernoulliBeam(id="e1", node_ids=("A", "B"), section_id="sec"))
    m.add(Support(id="pin", node_id="A", ux=True, uy=True))
    m.add(Support(id="roll", node_id="B", uy=True))
    return m, L


def make_two_span(L=5.0):
    m = Model()
    m.add(Node(id="A", x=0.0, y=0.0))
    m.add(Node(id="B", x=L, y=0.0))
    m.add(Node(id="C", x=2*L, y=0.0))
    m.add(ElasticSection(id="sec", material_id="steel", A=A, I=I))
    m.add(EulerBernoulliBeam(id="e1", node_ids=("A", "B"), section_id="sec"))
    m.add(EulerBernoulliBeam(id="e2", node_ids=("B", "C"), section_id="sec"))
    m.add(Support(id="pin", node_id="A", ux=True, uy=True))
    m.add(Support(id="roll", node_id="C", uy=True))
    return m, L


# ---------------------------------------------------------------------------
# Basic solver functionality
# ---------------------------------------------------------------------------

class TestSolverBasic:

    def test_returns_results(self):
        m, L = make_cantilever()
        lc = LoadCase(id="lc").update_nodal_load("B", [0.0, 10.0, 0.0])
        solver = LinearStaticSolver()
        results = solver.solve(m, lc, materials=MATERIALS)
        assert results is not None
        assert results.n_steps == 1

    def test_results_finalized(self):
        m, L = make_cantilever()
        lc = LoadCase(id="lc").update_nodal_load("B", [0.0, 10.0, 0.0])
        solver = LinearStaticSolver()
        results = solver.solve(m, lc, materials=MATERIALS)
        assert results._finalized

    def test_step_key_accessible(self):
        m, L = make_cantilever()
        lc = LoadCase(id="lc").update_nodal_load("B", [0.0, 10.0, 0.0])
        solver = LinearStaticSolver()
        results = solver.solve(m, lc, materials=MATERIALS)
        assert "step_1" in results._steps

    def test_node_result_accessible(self):
        m, L = make_cantilever()
        lc = LoadCase(id="lc").update_nodal_load("B", [0.0, 10.0, 0.0])
        solver = LinearStaticSolver()
        results = solver.solve(m, lc, materials=MATERIALS)
        nr = results["step_1"]["B"]
        assert hasattr(nr, "uy")

    def test_fixed_node_zero_displacement(self):
        m, L = make_cantilever()
        lc = LoadCase(id="lc").update_nodal_load("B", [0.0, 10.0, 0.0])
        solver = LinearStaticSolver()
        results = solver.solve(m, lc, materials=MATERIALS)
        nr_A = results["step_1"]["A"]
        assert nr_A.ux == pytest.approx(0.0, abs=1e-12)
        assert nr_A.uy == pytest.approx(0.0, abs=1e-12)
        assert nr_A.rz == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Analytical benchmarks
# ---------------------------------------------------------------------------

class TestAnalyticalBenchmarks:

    def test_cantilever_tip_load(self):
        """v_tip = PL^3/(3EI), theta_tip = PL^2/(2EI)."""
        L = 5.0
        P = 10.0
        m, _ = make_cantilever(L)
        lc = LoadCase(id="tip").update_nodal_load("B", [0.0, P, 0.0])
        results = LinearStaticSolver().solve(m, lc, materials=MATERIALS)
        EI = E * I
        assert results["step_1"]["B"].uy == pytest.approx(P*L**3/(3*EI), rel=1e-8)
        assert results["step_1"]["B"].rz == pytest.approx(P*L**2/(2*EI), rel=1e-8)

    def test_cantilever_tip_moment(self):
        """theta_tip = ML/(EI), v_tip = ML^2/(2EI)."""
        L = 5.0
        Mz = 50.0
        m, _ = make_cantilever(L)
        lc = LoadCase(id="mom").update_nodal_load("B", [0.0, 0.0, Mz])
        results = LinearStaticSolver().solve(m, lc, materials=MATERIALS)
        EI = E * I
        assert results["step_1"]["B"].rz == pytest.approx(Mz*L/(EI), rel=1e-8)
        assert results["step_1"]["B"].uy == pytest.approx(Mz*L**2/(2*EI), rel=1e-8)

    def test_cantilever_axial_load(self):
        """u_tip = NL/(EA)."""
        L = 5.0
        N = 100.0
        m, _ = make_cantilever(L)
        lc = LoadCase(id="axial").update_nodal_load("B", [N, 0.0, 0.0])
        results = LinearStaticSolver().solve(m, lc, materials=MATERIALS)
        EA = E * A
        assert results["step_1"]["B"].ux == pytest.approx(N*L/(EA), rel=1e-8)

    def test_simply_supported_udl_end_rotation(self):
        """End rotation for SS beam + UDL: theta = wL^3/(24EI)."""
        L = 6.0
        w = -5.0
        m, _ = make_simply_supported(L)
        lc = LoadCase(id="udl").update_distributed_load("e1", q=w)
        results = LinearStaticSolver().solve(m, lc, materials=MATERIALS)
        EI = E * I
        theta_A = w * L**3 / (24 * EI)
        assert results["step_1"]["A"].rz == pytest.approx(theta_A, rel=1e-8)

    def test_reactions_equilibrium_cantilever(self):
        """Sum of external + reaction forces = 0."""
        L = 5.0
        P = 10.0
        m, _ = make_cantilever(L)
        lc = LoadCase(id="tip").update_nodal_load("B", [0.0, P, 0.0])
        results = LinearStaticSolver().solve(m, lc, materials=MATERIALS)
        rxn = results["step_1"].reactions["A"]
        assert rxn[1] == pytest.approx(-P, rel=1e-6)        # Ry = -P
        assert rxn[2] == pytest.approx(-P * L, rel=1e-6)    # Mrz = -PL

    def test_two_span_symmetry(self):
        """Equal UDL on two equal spans: rotation at B = 0 by symmetry."""
        L = 5.0
        w = -10.0
        m, _ = make_two_span(L)
        lc = (
            LoadCase(id="sym")
            .update_distributed_load("e1", q=w)
            .update_distributed_load("e2", q=w)
        )
        results = LinearStaticSolver().solve(m, lc, materials=MATERIALS)
        assert results["step_1"]["B"].rz == pytest.approx(0.0, abs=1e-10)
        assert results["step_1"]["B"].uy < 0.0   # downward


# ---------------------------------------------------------------------------
# Multi-step load control
# ---------------------------------------------------------------------------

class TestMultiStep:

    def test_n_steps_produces_correct_count(self):
        m, L = make_cantilever()
        lc = LoadCase(id="lc").update_nodal_load("B", [0.0, 10.0, 0.0])
        control = PlainLoadControl(delta_lambda=0.25, n_steps=4)
        results = LinearStaticSolver().solve(m, lc, control, materials=MATERIALS)
        assert results.n_steps == 4

    def test_load_factors_correct(self):
        m, L = make_cantilever()
        lc = LoadCase(id="lc").update_nodal_load("B", [0.0, 10.0, 0.0])
        control = PlainLoadControl(delta_lambda=0.5, n_steps=2)
        results = LinearStaticSolver().solve(m, lc, control, materials=MATERIALS)
        assert results.load_factors == pytest.approx([0.5, 1.0])

    def test_displacement_grows_linearly_with_lambda(self):
        """Linear problem: displacement proportional to lambda."""
        m, L = make_cantilever()
        P = 10.0
        lc = LoadCase(id="lc").update_nodal_load("B", [0.0, P, 0.0])
        control = PlainLoadControl(delta_lambda=0.5, n_steps=2)
        results = LinearStaticSolver().solve(m, lc, control, materials=MATERIALS)
        uy1 = results["step_1"]["B"].uy
        uy2 = results["step_2"]["B"].uy
        assert uy2 == pytest.approx(2 * uy1, rel=1e-10)

    def test_node_history_length(self):
        m, L = make_cantilever()
        lc = LoadCase(id="lc").update_nodal_load("B", [0.0, 10.0, 0.0])
        control = PlainLoadControl(delta_lambda=0.2, n_steps=5)
        results = LinearStaticSolver().solve(m, lc, control, materials=MATERIALS)
        hist = results.node_history("B", "uy")
        assert len(hist) == 5


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

class TestCallbacks:

    def test_on_step_called(self):
        m, L = make_cantilever()
        lc = LoadCase(id="lc").update_nodal_load("B", [0.0, 10.0, 0.0])
        steps_seen = []
        cb = SolverCallbacks(on_step=lambda r: steps_seen.append(r.step))
        config = SolverConfig(callbacks=cb)
        solver = LinearStaticSolver(config=config)
        solver.solve(m, lc, materials=MATERIALS)
        assert steps_seen == [1]

    def test_on_step_called_multi(self):
        m, L = make_cantilever()
        lc = LoadCase(id="lc").update_nodal_load("B", [0.0, 10.0, 0.0])
        steps_seen = []
        cb = SolverCallbacks(on_step=lambda r: steps_seen.append(r.step))
        config = SolverConfig(callbacks=cb)
        control = PlainLoadControl(delta_lambda=0.5, n_steps=2)
        solver = LinearStaticSolver(config=config)
        solver.solve(m, lc, control, materials=MATERIALS)
        assert steps_seen == [1, 2]

    def test_on_convergence_called(self):
        m, L = make_cantilever()
        lc = LoadCase(id="lc").update_nodal_load("B", [0.0, 10.0, 0.0])
        conv_called = []
        cb = SolverCallbacks(on_convergence=lambda d: conv_called.append(d.load_factor))
        config = SolverConfig(callbacks=cb)
        solver = LinearStaticSolver(config=config)
        solver.solve(m, lc, materials=MATERIALS)
        assert len(conv_called) == 1

    def test_store_iterations(self):
        m, L = make_cantilever()
        lc = LoadCase(id="lc").update_nodal_load("B", [0.0, 10.0, 0.0])
        config = SolverConfig(store=StoreOptions(iterations=True))
        solver = LinearStaticSolver(config=config)
        results = solver.solve(m, lc, materials=MATERIALS)
        assert len(results.last.iterations) == 1
