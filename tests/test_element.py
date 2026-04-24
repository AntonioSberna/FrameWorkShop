"""
tests/test_element.py
=====================
Tests for ElementState, Element (ABC), EulerBernoulliBeam.

Reference values for stiffness matrix and displacements are computed
analytically for a horizontal beam (alpha=0) and verified against
standard Euler-Bernoulli formulas.
"""

import pytest
import numpy as np
from dataclasses import dataclass

from material import ElasticMaterial, MaterialState
from section import ElasticSection, SectionState
from element import (
    ElementState,
    EulerBernoulliBeam,
    SectionForces,
    DeformedGeometry,
)


# ---------------------------------------------------------------------------
# Minimal Node stub — avoids importing node.py before it exists
# ---------------------------------------------------------------------------

@dataclass
class _Node:
    id: str | int
    x: float
    y: float
    dofs: list


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def make_horizontal_beam(L=5.0, n_gauss=2):
    """Horizontal beam from (0,0) to (L,0)."""
    mat = ElasticMaterial(id="steel", E=210_000.0, rho=7.85e-3)
    sec = ElasticSection(id="sec", material_id="steel", A=100.0, I=10_000.0)
    ni = _Node(id="i", x=0.0, y=0.0, dofs=[0, 1, 2])
    nj = _Node(id="j", x=L,   y=0.0, dofs=[3, 4, 5])
    elem = EulerBernoulliBeam(id="b1", node_ids=("i", "j"), section_id="sec",
                               n_gauss=n_gauss)
    return mat, sec, elem, [ni, nj], L


def make_inclined_beam(angle_deg=45.0, L=5.0):
    """Inclined beam at given angle."""
    import math
    alpha = math.radians(angle_deg)
    mat = ElasticMaterial(id="steel", E=210_000.0, rho=7.85e-3)
    sec = ElasticSection(id="sec", material_id="steel", A=100.0, I=10_000.0)
    ni = _Node(id="i", x=0.0,             y=0.0,             dofs=[0,1,2])
    nj = _Node(id="j", x=L*math.cos(alpha), y=L*math.sin(alpha), dofs=[3,4,5])
    elem = EulerBernoulliBeam(id="b2", node_ids=("i", "j"), section_id="sec")
    return mat, sec, elem, [ni, nj], L, alpha


# ---------------------------------------------------------------------------
# ElementState
# ---------------------------------------------------------------------------

class TestElementState:

    def test_initial_gauss_states_count(self):
        s = ElementState(n_gauss=3)
        assert len(s.gauss_states) == 3

    def test_commit_all_gauss_states(self):
        s = ElementState(n_gauss=2)
        s.gauss_states[0].eps_trial = 0.001
        s.gauss_states[1].eps_trial = 0.002
        s.commit()
        assert s.gauss_states[0].eps_committed == 0.001
        assert s.gauss_states[1].eps_committed == 0.002

    def test_revert_all_gauss_states(self):
        s = ElementState(n_gauss=2)
        s.gauss_states[0].eps_trial = 0.001
        s.commit()
        s.gauss_states[0].eps_trial = 0.999
        s.revert()
        assert s.gauss_states[0].eps_trial == 0.001

    def test_serialization_round_trip(self):
        s = ElementState(n_gauss=2)
        s.gauss_states[0].eps_trial = 0.005
        s.commit()
        s2 = ElementState.from_dict(s.to_dict())
        assert len(s2.gauss_states) == 2
        assert s2.gauss_states[0].eps_committed == 0.005


# ---------------------------------------------------------------------------
# EulerBernoulliBeam — geometry helpers
# ---------------------------------------------------------------------------

class TestBeamGeometry:

    def test_length_horizontal(self):
        _, _, elem, nodes, L = make_horizontal_beam()
        length, alpha = elem._get_length_and_angle(nodes)
        assert length == pytest.approx(L)
        assert alpha == pytest.approx(0.0)

    def test_length_inclined(self):
        mat, sec, elem, nodes, L, alpha = make_inclined_beam(45.0, 5.0)
        length, angle = elem._get_length_and_angle(nodes)
        assert length == pytest.approx(L)
        assert angle == pytest.approx(alpha)

    def test_rotation_matrix_horizontal(self):
        _, _, elem, nodes, L = make_horizontal_beam()
        T = elem._rotation_matrix(0.0)
        np.testing.assert_allclose(T, np.eye(6), atol=1e-12)

    def test_rotation_matrix_orthogonal(self):
        _, _, elem, nodes, L, alpha = make_inclined_beam(30.0)
        T = elem._rotation_matrix(alpha)
        np.testing.assert_allclose(T @ T.T, np.eye(6), atol=1e-12)

    def test_zero_length_raises(self):
        ni = _Node("i", 0.0, 0.0, [0,1,2])
        nj = _Node("j", 0.0, 0.0, [3,4,5])
        elem = EulerBernoulliBeam("b", ("i","j"), "s")
        with pytest.raises(ValueError, match="zero length"):
            elem._get_length_and_angle([ni, nj])


# ---------------------------------------------------------------------------
# EulerBernoulliBeam — stiffness matrix
# ---------------------------------------------------------------------------

class TestBeamStiffness:

    def test_stiffness_shape(self):
        mat, sec, elem, nodes, L = make_horizontal_beam()
        K = elem.get_local_stiffness(nodes, sec, mat)
        assert K.shape == (6, 6)

    def test_stiffness_symmetric(self):
        mat, sec, elem, nodes, L = make_horizontal_beam()
        K = elem.get_local_stiffness(nodes, sec, mat)
        np.testing.assert_allclose(K, K.T, atol=1e-10)

    def test_stiffness_singular(self):
        """K must be singular (rigid body modes not constrained)."""
        mat, sec, elem, nodes, L = make_horizontal_beam()
        K = elem.get_local_stiffness(nodes, sec, mat)
        rank = np.linalg.matrix_rank(K, tol=1e-6)
        assert rank == 3   # 6 DOFs - 3 rigid body modes

    def test_axial_stiffness_value(self):
        """K[0,0] for horizontal beam = EA/L."""
        mat, sec, elem, nodes, L = make_horizontal_beam()
        K = elem.get_local_stiffness(nodes, sec, mat)
        EA = mat.E * sec.A
        assert K[0, 0] == pytest.approx(EA / L, rel=1e-10)

    def test_bending_stiffness_value(self):
        """K[1,1] for horizontal beam = 12*EI/L^3."""
        mat, sec, elem, nodes, L = make_horizontal_beam()
        K = elem.get_local_stiffness(nodes, sec, mat)
        EI = mat.E * sec.I
        assert K[1, 1] == pytest.approx(12 * EI / L**3, rel=1e-10)

    def test_stiffness_rotational_invariance(self):
        """
        K_global rotated back to local must equal K_local.
        """
        mat, sec, elem, nodes, L, alpha = make_inclined_beam(30.0)
        K_global = elem.get_local_stiffness(nodes, sec, mat)
        T = elem._rotation_matrix(alpha)
        K_recovered = T @ K_global @ T.T
        EA, EI = sec.get_stiffness(mat)
        K_local_ref = elem._local_stiffness(EA, EI, L)
        np.testing.assert_allclose(K_recovered, K_local_ref, atol=1e-6)

    def test_deformed_geometry_raises_not_implemented(self):
        mat, sec, elem, nodes, L = make_horizontal_beam()
        geom = DeformedGeometry({"i": (0.0, 0.0), "j": (L, 0.0)})
        with pytest.raises(NotImplementedError):
            elem.get_local_stiffness(nodes, sec, mat, geometry=geom)


# ---------------------------------------------------------------------------
# EulerBernoulliBeam — mass matrix
# ---------------------------------------------------------------------------

class TestBeamMass:

    def test_mass_shape(self):
        mat, sec, elem, nodes, L = make_horizontal_beam()
        M = elem.get_local_mass(nodes, sec, mat)
        assert M.shape == (6, 6)

    def test_mass_symmetric(self):
        mat, sec, elem, nodes, L = make_horizontal_beam()
        M = elem.get_local_mass(nodes, sec, mat)
        np.testing.assert_allclose(M, M.T, atol=1e-12)

    def test_mass_positive_definite(self):
        mat, sec, elem, nodes, L = make_horizontal_beam()
        M = elem.get_local_mass(nodes, sec, mat)
        assert np.all(np.linalg.eigvalsh(M) > 0)


# ---------------------------------------------------------------------------
# EulerBernoulliBeam — cantilever benchmark
# ---------------------------------------------------------------------------

class TestCantileverBenchmark:
    """
    Single cantilever beam: node i fixed, node j free with tip load P.
    Analytical solution:
      v_tip   = P * L^3 / (3 * EI)
      theta_tip = P * L^2 / (2 * EI)
    """

    def setup_method(self):
        self.L = 5.0
        self.mat = ElasticMaterial(id="steel", E=210_000.0, rho=7.85e-3)
        self.sec = ElasticSection(id="sec", material_id="steel",
                                   A=100.0, I=10_000.0)
        ni = _Node("i", 0.0, 0.0, [0, 1, 2])
        nj = _Node("j", self.L, 0.0, [3, 4, 5])
        self.nodes = [ni, nj]
        self.elem = EulerBernoulliBeam("b", ("i","j"), "sec")

    def _solve_cantilever(self, P):
        """
        Solve the clamped-free system by condensation.
        Fixed DOFs: 0,1,2 (node i). Free DOFs: 3,4,5 (node j).
        """
        K = self.elem.get_local_stiffness(self.nodes, self.sec, self.mat)
        # extract free DOF sub-matrix (3x3)
        free = [3, 4, 5]
        K_ff = K[np.ix_(free, free)]
        F_f = np.array([0.0, P, 0.0])   # tip load in y direction
        U_f = np.linalg.solve(K_ff, F_f)
        # full displacement vector
        U = np.zeros(6)
        U[free] = U_f
        return U

    def test_tip_displacement_v(self):
        P = 10.0  # kN
        U = self._solve_cantilever(P)
        EI = self.mat.E * self.sec.I
        v_tip_analytical = P * self.L**3 / (3 * EI)
        assert U[4] == pytest.approx(v_tip_analytical, rel=1e-10)

    def test_tip_rotation_theta(self):
        P = 10.0
        U = self._solve_cantilever(P)
        EI = self.mat.E * self.sec.I
        theta_tip_analytical = P * self.L**2 / (2 * EI)
        assert U[5] == pytest.approx(theta_tip_analytical, rel=1e-10)

    def test_update_and_gauss_strains(self):
        """After update(), Gauss point strains must be non-zero for P > 0."""
        P = 10.0
        U_global = self._solve_cantilever(P)
        self.elem.update(self.nodes, self.sec, self.mat, U_global)
        # at least one Gauss point should have non-zero curvature
        kappas = [gs.kappa_trial for gs in self.elem.state.gauss_states]
        assert any(abs(k) > 1e-15 for k in kappas)

    def test_commit_revert_cycle(self):
        """commit() then revert() with a new trial must restore previous state."""
        P = 10.0
        U = self._solve_cantilever(P)
        self.elem.update(self.nodes, self.sec, self.mat, U)
        self.elem.commit()
        kappa_committed = self.elem.state.gauss_states[0].kappa_committed

        # simulate a diverging iteration
        U2 = U * 5.0
        self.elem.update(self.nodes, self.sec, self.mat, U2)
        self.elem.revert()
        assert self.elem.state.gauss_states[0].kappa_trial == pytest.approx(
            kappa_committed, rel=1e-10
        )


# ---------------------------------------------------------------------------
# EulerBernoulliBeam — section forces
# ---------------------------------------------------------------------------

class TestSectionForces:

    def test_section_forces_type(self):
        mat, sec, elem, nodes, L = make_horizontal_beam()
        P = 10.0
        U = np.zeros(6)
        U[4] = P * L**3 / (3 * mat.E * sec.I)
        U[5] = P * L**2 / (2 * mat.E * sec.I)
        elem.update(nodes, sec, mat, U)
        sf = elem.get_section_forces(0.5, nodes, sec, mat)
        assert isinstance(sf, SectionForces)
        assert sf.xi == 0.5


# ---------------------------------------------------------------------------
# EulerBernoulliBeam — Gauss points
# ---------------------------------------------------------------------------

class TestGaussPoints:

    def test_gauss_1_point(self):
        xi, w = EulerBernoulliBeam._gauss_points(1)
        assert len(xi) == 1
        assert sum(w) == pytest.approx(1.0)

    def test_gauss_2_points(self):
        xi, w = EulerBernoulliBeam._gauss_points(2)
        assert len(xi) == 2
        assert sum(w) == pytest.approx(1.0)

    def test_gauss_3_points(self):
        xi, w = EulerBernoulliBeam._gauss_points(3)
        assert len(xi) == 3
        assert sum(w) == pytest.approx(1.0)

    def test_gauss_invalid_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            EulerBernoulliBeam._gauss_points(5)


# ---------------------------------------------------------------------------
# EulerBernoulliBeam — serialization
# ---------------------------------------------------------------------------

class TestBeamSerialization:

    def test_round_trip(self):
        mat, sec, elem, nodes, L = make_horizontal_beam()
        d = elem.to_dict()
        assert d["type"] == "EulerBernoulliBeam"
        elem2 = EulerBernoulliBeam.from_dict(d)
        assert elem2.id == elem.id
        assert elem2.node_ids == elem.node_ids
        assert elem2.section_id == elem.section_id
        assert elem2.n_gauss == elem.n_gauss
