"""
tests/test_convergence.py
=========================
Tests for ConvergenceCriterion, ForceResidual, DisplacementIncrement,
EnergyConvergence, and IterationData.

Tests are structured to cover:
  - Linear elastic behavior (Phase 0): convergence in one iteration.
  - Simulated nonlinear iteration sequences (Phase 1 preparation):
    convergence over multiple iterations with decaying residuals.
  - Edge cases: zero load, zero displacement, reference energy.
"""

import pytest
import numpy as np
from convergence import (
    IterationData,
    ForceResidual,
    DisplacementIncrement,
    EnergyConvergence,
)


# ---------------------------------------------------------------------------
# Helpers — build IterationData snapshots
# ---------------------------------------------------------------------------

def make_data(
    residual=None,
    delta_U=None,
    U_trial=None,
    F_external=None,
    iteration=0,
    load_factor=1.0,
):
    n = 6
    return IterationData(
        iteration=iteration,
        residual=np.zeros(n) if residual is None else np.asarray(residual),
        delta_U=np.zeros(n) if delta_U is None else np.asarray(delta_U),
        U_trial=np.zeros(n) if U_trial is None else np.asarray(U_trial),
        F_external=np.ones(n) if F_external is None else np.asarray(F_external),
        load_factor=load_factor,
    )


def linear_converged_data():
    """Simulates a linear solve: delta_U = solution, residual ~ 0."""
    U = np.array([0.0, 0.0, 0.0, 0.001, 0.005, 0.002])
    F = np.array([0.0, 0.0, 0.0, 0.0, 10.0, 0.0])
    R = np.zeros(6)   # exact solution -> zero residual
    return make_data(residual=R, delta_U=U, U_trial=U, F_external=F)


def nl_iteration_sequence(n_iter=5, decay=0.1):
    """
    Simulate a converging Newton-Raphson sequence.
    Residual decays geometrically; displacement increment shrinks.
    """
    F = np.ones(6) * 100.0
    U_total = np.ones(6) * 0.01
    snapshots = []
    for i in range(n_iter):
        R = F * (decay ** (i + 1))
        dU = U_total * (decay ** i)
        U = U_total * (1 - decay ** i)
        snapshots.append(make_data(
            residual=R, delta_U=dU, U_trial=U,
            F_external=F, iteration=i
        ))
    return snapshots


# ---------------------------------------------------------------------------
# IterationData
# ---------------------------------------------------------------------------

class TestIterationData:

    def test_default_construction(self):
        data = IterationData()
        assert data.iteration == 0
        assert data.load_factor == 1.0

    def test_internal_energy(self):
        dU = np.array([1.0, 2.0, 0.0])
        R = np.array([3.0, 4.0, 0.0])
        data = make_data(delta_U=dU, residual=R)
        assert data.internal_energy == pytest.approx(1*3 + 2*4)

    def test_internal_energy_zero_residual(self):
        data = make_data(delta_U=np.ones(6), residual=np.zeros(6))
        assert data.internal_energy == pytest.approx(0.0)

    def test_reference_energy(self):
        dU = np.array([1.0, 0.0, 0.0])
        F = np.array([5.0, 0.0, 0.0])
        data = make_data(delta_U=dU, F_external=F)
        assert data.reference_energy == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# ConvergenceCriterion — base validation
# ---------------------------------------------------------------------------

class TestBaseCriterionValidation:

    def test_negative_tol_raises(self):
        with pytest.raises(ValueError, match="positive"):
            ForceResidual(tol=-1e-6)

    def test_zero_tol_raises(self):
        with pytest.raises(ValueError, match="positive"):
            ForceResidual(tol=0.0)

    def test_invalid_norm_raises(self):
        with pytest.raises(ValueError, match="norm"):
            ForceResidual(tol=1e-6, norm="L1")

    def test_repr(self):
        c = ForceResidual(tol=1e-6)
        assert "ForceResidual" in repr(c)
        assert "1e-06" in repr(c)


# ---------------------------------------------------------------------------
# ForceResidual
# ---------------------------------------------------------------------------

class TestForceResidual:

    def test_linear_solve_converges(self):
        """Linear solve -> zero residual -> always converged."""
        c = ForceResidual(tol=1e-6)
        data = linear_converged_data()
        assert c.is_converged(data)

    def test_large_residual_not_converged(self):
        c = ForceResidual(tol=1e-6)
        F = np.ones(6) * 100.0
        R = np.ones(6) * 10.0    # ratio = 0.1 >> tol
        data = make_data(residual=R, F_external=F)
        assert not c.is_converged(data)

    def test_residual_below_tol_converged(self):
        c = ForceResidual(tol=1e-4)
        F = np.ones(6) * 100.0
        R = np.ones(6) * 1e-5    # ratio = 1e-7 < tol
        data = make_data(residual=R, F_external=F)
        assert c.is_converged(data)

    def test_zero_load_uses_absolute_norm(self):
        """When F_ext ~ 0, criterion falls back to absolute residual norm."""
        c = ForceResidual(tol=1e-6)
        R = np.ones(6) * 1e-10
        F = np.zeros(6)
        data = make_data(residual=R, F_external=F)
        assert c.is_converged(data)

    def test_zero_load_large_residual_not_converged(self):
        c = ForceResidual(tol=1e-6)
        R = np.ones(6) * 1.0
        F = np.zeros(6)
        data = make_data(residual=R, F_external=F)
        assert not c.is_converged(data)

    def test_Linf_norm(self):
        c = ForceResidual(tol=1e-4, norm="Linf")
        F = np.ones(6) * 100.0
        R = np.zeros(6)
        R[0] = 1e-3   # max component
        data = make_data(residual=R, F_external=F)
        # ratio = 1e-3 / (100 * sqrt(6) in L2, but 1e-3/100 in Linf) = 1e-5 < tol
        assert c.is_converged(data)

    def test_nl_sequence_converges_eventually(self):
        """Simulated NL iteration: must converge before last snapshot."""
        c = ForceResidual(tol=1e-4)
        snapshots = nl_iteration_sequence(n_iter=6, decay=0.1)
        results = [c.is_converged(s) for s in snapshots]
        assert any(results)               # converges at some point
        assert results[-1]                # certainly converged at last iter
        assert not results[0]             # not converged at first iter

    def test_description_contains_key_info(self):
        c = ForceResidual(tol=1e-6)
        data = linear_converged_data()
        desc = c.description(data)
        assert "ForceResidual" in desc
        assert "CONVERGED" in desc

    def test_serialization_round_trip(self):
        c = ForceResidual(tol=1e-8, norm="Linf")
        c2 = ForceResidual.from_dict(c.to_dict())
        assert c2.tol == c.tol
        assert c2.norm == c.norm
        assert c2.to_dict()["type"] == "ForceResidual"


# ---------------------------------------------------------------------------
# DisplacementIncrement
# ---------------------------------------------------------------------------

class TestDisplacementIncrement:

    def test_linear_solve_converges(self):
        """Linear solve: delta_U = U_trial (first iter), ratio = 1."""
        # convergence requires small ratio -> not converged at ratio=1
        c = DisplacementIncrement(tol=1e-6)
        U = np.ones(6) * 0.01
        data = make_data(delta_U=U, U_trial=U)
        # ratio = ||U||/||U|| = 1.0 >> tol
        assert not c.is_converged(data)

    def test_small_increment_converges(self):
        c = DisplacementIncrement(tol=1e-4)
        U_trial = np.ones(6) * 1.0
        delta_U = np.ones(6) * 1e-6   # very small increment
        data = make_data(delta_U=delta_U, U_trial=U_trial)
        assert c.is_converged(data)

    def test_zero_U_trial_uses_absolute(self):
        c = DisplacementIncrement(tol=1e-6)
        delta_U = np.ones(6) * 1e-10
        U_trial = np.zeros(6)
        data = make_data(delta_U=delta_U, U_trial=U_trial)
        assert c.is_converged(data)

    def test_nl_sequence_converges(self):
        c = DisplacementIncrement(tol=1e-3)
        snapshots = nl_iteration_sequence(n_iter=6, decay=0.05)
        results = [c.is_converged(s) for s in snapshots]
        assert results[-1]

    def test_description_contains_key_info(self):
        c = DisplacementIncrement(tol=1e-6)
        U = np.ones(6) * 0.01
        dU = np.ones(6) * 1e-9
        data = make_data(delta_U=dU, U_trial=U)
        desc = c.description(data)
        assert "DisplacementIncrement" in desc
        assert "CONVERGED" in desc

    def test_serialization_round_trip(self):
        c = DisplacementIncrement(tol=1e-7, norm="Linf")
        c2 = DisplacementIncrement.from_dict(c.to_dict())
        assert c2.tol == c.tol
        assert c2.norm == c.norm
        assert c2.to_dict()["type"] == "DisplacementIncrement"


# ---------------------------------------------------------------------------
# EnergyConvergence
# ---------------------------------------------------------------------------

class TestEnergyConvergence:

    def test_not_converged_before_set_reference(self):
        """Must return False before set_reference() is called."""
        c = EnergyConvergence(tol=1e-8)
        data = make_data(delta_U=np.ones(6), residual=np.ones(6) * 1e-12)
        assert not c.is_converged(data)

    def test_converged_after_set_reference_small_energy(self):
        c = EnergyConvergence(tol=1e-6)
        # first iteration: large energy
        dU_0 = np.ones(6) * 0.01
        R_0 = np.ones(6) * 100.0
        data_0 = make_data(delta_U=dU_0, residual=R_0, F_external=R_0)
        c.set_reference(data_0)
        # later iteration: tiny energy
        dU_k = np.ones(6) * 1e-9
        R_k = np.ones(6) * 1e-9
        data_k = make_data(delta_U=dU_k, residual=R_k, F_external=R_0)
        assert c.is_converged(data_k)

    def test_not_converged_large_energy(self):
        c = EnergyConvergence(tol=1e-8)
        dU = np.ones(6) * 0.01
        R = np.ones(6) * 100.0
        data = make_data(delta_U=dU, residual=R, F_external=R)
        c.set_reference(data)
        assert not c.is_converged(data)   # same data -> ratio = 1

    def test_reset_clears_reference(self):
        c = EnergyConvergence(tol=1e-8)
        dU = np.ones(6) * 0.01
        R = np.ones(6) * 1e-12
        data = make_data(delta_U=dU, residual=R)
        c.set_reference(data)
        c.reset()
        assert not c.is_converged(data)

    def test_nl_sequence(self):
        """Energy criterion must converge over a decaying sequence."""
        c = EnergyConvergence(tol=1e-6)
        snapshots = nl_iteration_sequence(n_iter=8, decay=0.05)
        c.set_reference(snapshots[0])
        results = [c.is_converged(s) for s in snapshots]
        assert not results[0]
        assert results[-1]

    def test_description_before_reference(self):
        c = EnergyConvergence(tol=1e-8)
        data = make_data()
        desc = c.description(data)
        assert "EnergyConvergence" in desc

    def test_description_after_reference(self):
        c = EnergyConvergence(tol=1e-8)
        dU = np.ones(6) * 1e-10
        R = np.ones(6) * 1e-10
        data = make_data(delta_U=dU, residual=R, F_external=np.ones(6))
        c.set_reference(data)
        desc = c.description(data)
        assert "EnergyConvergence" in desc

    def test_serialization_round_trip(self):
        c = EnergyConvergence(tol=1e-10)
        c2 = EnergyConvergence.from_dict(c.to_dict())
        assert c2.tol == c.tol
        assert c2.to_dict()["type"] == "EnergyConvergence"

    def test_zero_reference_energy_handled(self):
        """If reference energy is zero, set_reference stores None -> not converged."""
        c = EnergyConvergence(tol=1e-8)
        data = make_data(delta_U=np.zeros(6), residual=np.zeros(6))
        c.set_reference(data)   # reference = 0 -> stored as None
        assert not c.is_converged(data)


# ---------------------------------------------------------------------------
# Cross-criterion comparison (nonlinear simulation)
# ---------------------------------------------------------------------------

class TestCriteriaComparison:
    """
    Verify that all three criteria can be used interchangeably
    in a simulated Newton-Raphson loop.
    """

    def _run_loop(self, criterion, max_iter=20):
        """Simulated NL loop: return iteration at convergence or None."""
        F = np.ones(6) * 1000.0
        U = np.zeros(6)
        if hasattr(criterion, "reset"):
            criterion.reset()

        for i in range(max_iter):
            R = F * (0.1 ** (i + 1))
            dU = np.ones(6) * 0.01 * (0.1 ** i)
            U = U + dU
            data = IterationData(
                iteration=i,
                residual=R,
                delta_U=dU,
                U_trial=U.copy(),
                F_external=F,
                load_factor=1.0,
            )
            if i == 0 and hasattr(criterion, "set_reference"):
                criterion.set_reference(data)
            if criterion.is_converged(data):
                return i
        return None

    def test_force_residual_converges(self):
        c = ForceResidual(tol=1e-4)
        iter_conv = self._run_loop(c)
        assert iter_conv is not None

    def test_displacement_increment_converges(self):
        c = DisplacementIncrement(tol=1e-4)
        iter_conv = self._run_loop(c)
        assert iter_conv is not None

    def test_energy_convergence_converges(self):
        c = EnergyConvergence(tol=1e-6)
        iter_conv = self._run_loop(c)
        assert iter_conv is not None

    def test_energy_stricter_than_force(self):
        """EnergyConvergence with tight tol requires more iterations than ForceResidual."""
        c_force = ForceResidual(tol=1e-3)
        c_energy = EnergyConvergence(tol=1e-8)
        i_force = self._run_loop(c_force)
        i_energy = self._run_loop(c_energy)
        assert i_energy >= i_force
