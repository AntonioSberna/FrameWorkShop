"""
tests/test_assembler.py
=======================
Tests for Assembler: assemble_K, assemble_M, assemble_F.

Analytical benchmarks used:
  - Cantilever beam: tip displacement v = PL^3/(3EI), theta = PL^2/(2EI)
  - Simply supported beam: midspan UDL w, deflection = 5wL^4/(384EI)
  - Two-element beam: verifies DOF assembly across elements
"""

import pytest
import numpy as np
from scipy.sparse.linalg import spsolve

from node import Node, Support, NodalMass
from section import ElasticSection
from element import EulerBernoulliBeam
from material import ElasticMaterial
from model import Model, DOFManager
from constraint import PlainConstraintHandler
from load_case import LoadCase
from assembler import Assembler


# ---------------------------------------------------------------------------
# Shared setup helpers
# ---------------------------------------------------------------------------

MAT = ElasticMaterial(id="steel", E=210_000.0, rho=7.85e-3)
MATERIALS = {"steel": MAT}


def make_cantilever(L=5.0):
    m = Model()
    m.add(Node(id="A", x=0.0, y=0.0))
    m.add(Node(id="B", x=L,   y=0.0))
    m.add(ElasticSection(id="sec", material_id="steel", A=100.0, I=10_000.0))
    m.add(EulerBernoulliBeam(id="e1", node_ids=("A", "B"), section_id="sec"))
    m.add(Support(id="fix", node_id="A", ux=True, uy=True, rz=True))
    dof_mgr = m.finalize()
    return m, dof_mgr


def make_simply_supported(L=6.0):
    m = Model()
    m.add(Node(id="A", x=0.0, y=0.0))
    m.add(Node(id="B", x=L,   y=0.0))
    m.add(ElasticSection(id="sec", material_id="steel", A=100.0, I=10_000.0))
    m.add(EulerBernoulliBeam(id="e1", node_ids=("A", "B"), section_id="sec"))
    m.add(Support(id="pin",  node_id="A", ux=True, uy=True))
    m.add(Support(id="roll", node_id="B", uy=True))
    dof_mgr = m.finalize()
    return m, dof_mgr


def make_two_span(L=5.0):
    """Two equal spans A-B-C, pinned at A and C, internal node B free."""
    m = Model()
    m.add(Node(id="A", x=0.0,  y=0.0))
    m.add(Node(id="B", x=L,    y=0.0))
    m.add(Node(id="C", x=2*L,  y=0.0))
    m.add(ElasticSection(id="sec", material_id="steel", A=100.0, I=10_000.0))
    m.add(EulerBernoulliBeam(id="e1", node_ids=("A", "B"), section_id="sec"))
    m.add(EulerBernoulliBeam(id="e2", node_ids=("B", "C"), section_id="sec"))
    m.add(Support(id="pin",  node_id="A", ux=True, uy=True))
    m.add(Support(id="roll", node_id="C", uy=True))
    dof_mgr = m.finalize()
    return m, dof_mgr


def solve(model, dof_mgr, load_case):
    """Full solve pipeline: assemble, constrain, solve."""
    asm = Assembler()
    handler = PlainConstraintHandler()
    handler.initialize(model, dof_mgr)

    K = asm.assemble_K(model, dof_mgr, MATERIALS)
    F = asm.assemble_F(load_case, model, dof_mgr, MATERIALS)
    K_mod, F_mod = handler.apply(K, F)
    U = spsolve(K_mod, F_mod)
    return U, K, F


# ---------------------------------------------------------------------------
# assemble_K
# ---------------------------------------------------------------------------

class TestAssembleK:

    def test_shape(self):
        m, dof_mgr = make_cantilever()
        asm = Assembler()
        K = asm.assemble_K(m, dof_mgr, MATERIALS)
        assert K.shape == (6, 6)

    def test_symmetric(self):
        m, dof_mgr = make_cantilever()
        asm = Assembler()
        K = asm.assemble_K(m, dof_mgr, MATERIALS)
        np.testing.assert_allclose(
            K.toarray(), K.toarray().T, atol=1e-10
        )

    def test_two_element_shape(self):
        m, dof_mgr = make_two_span()
        asm = Assembler()
        K = asm.assemble_K(m, dof_mgr, MATERIALS)
        assert K.shape == (9, 9)   # 3 nodes * 3 DOFs

    def test_two_element_symmetric(self):
        m, dof_mgr = make_two_span()
        asm = Assembler()
        K = asm.assemble_K(m, dof_mgr, MATERIALS)
        np.testing.assert_allclose(
            K.toarray(), K.toarray().T, atol=1e-10
        )

    def test_missing_material_raises(self):
        m, dof_mgr = make_cantilever()
        asm = Assembler()
        with pytest.raises(ValueError, match="Material"):
            asm.assemble_K(m, dof_mgr, materials={})


# ---------------------------------------------------------------------------
# assemble_M
# ---------------------------------------------------------------------------

class TestAssembleM:

    def test_shape(self):
        m, dof_mgr = make_cantilever()
        asm = Assembler()
        M = asm.assemble_M(m, dof_mgr, MATERIALS)
        assert M.shape == (6, 6)

    def test_symmetric(self):
        m, dof_mgr = make_cantilever()
        asm = Assembler()
        M = asm.assemble_M(m, dof_mgr, MATERIALS)
        np.testing.assert_allclose(
            M.toarray(), M.toarray().T, atol=1e-12
        )

    def test_positive_definite(self):
        m, dof_mgr = make_cantilever()
        asm = Assembler()
        M = asm.assemble_M(m, dof_mgr, MATERIALS)
        eigvals = np.linalg.eigvalsh(M.toarray())
        assert np.all(eigvals > 0)

    def test_nodal_mass_added_to_diagonal(self):
        m, dof_mgr = make_cantilever()
        m.add(NodalMass(id="nm1", node_id="B", mx=5.0, my=5.0, I_theta=1.0))
        dof_mgr = m.finalize()
        asm = Assembler()
        M = asm.assemble_M(m, dof_mgr, MATERIALS)
        M_arr = M.toarray()
        # DOFs for node B are [3,4,5]
        # mx added to M[3,3], my to M[4,4], I_theta to M[5,5]
        M_no_nm = Assembler().assemble_M(
            *make_cantilever(), MATERIALS
        )
        assert M_arr[3, 3] == pytest.approx(M_no_nm.toarray()[3, 3] + 5.0)
        assert M_arr[4, 4] == pytest.approx(M_no_nm.toarray()[4, 4] + 5.0)
        assert M_arr[5, 5] == pytest.approx(M_no_nm.toarray()[5, 5] + 1.0)


# ---------------------------------------------------------------------------
# assemble_F — nodal loads
# ---------------------------------------------------------------------------

class TestAssembleF_Nodal:

    def test_nodal_load_scattered_correctly(self):
        m, dof_mgr = make_cantilever()
        asm = Assembler()
        lc = LoadCase(id="lc").update_nodal_load("B", [0.0, -10.0, 0.0])
        F = asm.assemble_F(lc, m, dof_mgr, MATERIALS)
        # node B has dofs [3,4,5]
        assert F[4] == pytest.approx(-10.0)
        assert F[3] == pytest.approx(0.0)
        assert F[5] == pytest.approx(0.0)

    def test_multiple_nodal_loads(self):
        m, dof_mgr = make_two_span()
        asm = Assembler()
        lc = (
            LoadCase(id="lc")
            .update_nodal_load("A", [1.0, 0.0, 0.0])
            .update_nodal_load("C", [0.0, -5.0, 0.0])
        )
        F = asm.assemble_F(lc, m, dof_mgr, MATERIALS)
        assert F[0] == pytest.approx(1.0)    # node A ux
        assert F[7] == pytest.approx(-5.0)   # node C uy (dofs 6,7,8)

    def test_empty_load_case_gives_zero_vector(self):
        m, dof_mgr = make_cantilever()
        asm = Assembler()
        F = asm.assemble_F(LoadCase(id="empty"), m, dof_mgr, MATERIALS)
        np.testing.assert_allclose(F, 0.0)


# ---------------------------------------------------------------------------
# assemble_F — distributed loads
# ---------------------------------------------------------------------------

class TestAssembleF_Distributed:

    def test_udl_global_y_equivalent_nodal_forces(self):
        """
        UDL q on horizontal beam: equivalent nodal forces must satisfy
        F_i = F_j = qL/2  and  M_i = -M_j = qL^2/12.
        """
        L = 6.0
        q = -10.0   # downward
        m, dof_mgr = make_simply_supported(L)
        asm = Assembler()
        lc = LoadCase(id="udl").update_distributed_load(
            "e1", q=q, direction="y", reference="global"
        )
        F = asm.assemble_F(lc, m, dof_mgr, MATERIALS)
        # node A dofs [0,1,2], node B dofs [3,4,5]
        assert F[1] == pytest.approx(q * L / 2, rel=1e-10)   # Fy at A
        assert F[4] == pytest.approx(q * L / 2, rel=1e-10)   # Fy at B
        assert F[2] == pytest.approx( q * L**2 / 12, rel=1e-10)  # Mz at A
        assert F[5] == pytest.approx(-q * L**2 / 12, rel=1e-10)  # Mz at B

    def test_udl_symmetric_forces(self):
        """For a horizontal beam UDL, shear forces at i and j are equal."""
        m, dof_mgr = make_simply_supported(6.0)
        asm = Assembler()
        lc = LoadCase(id="udl").update_distributed_load("e1", q=-5.0)
        F = asm.assemble_F(lc, m, dof_mgr, MATERIALS)
        assert F[1] == pytest.approx(F[4], rel=1e-10)


# ---------------------------------------------------------------------------
# Full solve benchmarks
# ---------------------------------------------------------------------------

class TestFullSolve:

    def test_cantilever_tip_load(self):
        """v_tip = PL^3 / (3EI)"""
        L = 5.0
        P = 10.0
        m, dof_mgr = make_cantilever(L)
        lc = LoadCase(id="tip").update_nodal_load("B", [0.0, P, 0.0])
        U, K, F = solve(m, dof_mgr, lc)
        EI = MAT.E * 10_000.0
        v_tip = P * L**3 / (3 * EI)
        assert U[4] == pytest.approx(v_tip, rel=1e-8)

    def test_cantilever_tip_moment(self):
        """theta_tip = ML / (EI) for tip moment M."""
        L = 5.0
        Mz = 50.0
        m, dof_mgr = make_cantilever(L)
        lc = LoadCase(id="mom").update_nodal_load("B", [0.0, 0.0, Mz])
        U, K, F = solve(m, dof_mgr, lc)
        EI = MAT.E * 10_000.0
        theta_tip = Mz * L / EI
        assert U[5] == pytest.approx(theta_tip, rel=1e-8)

    def test_simply_supported_udl(self):
        """
        Simply supported beam, UDL w.
        Max deflection at midspan: 5wL^4 / (384EI).
        Single element cannot capture midspan deflection exactly, but
        end rotations must match: theta = wL^3 / (24EI).
        """
        L = 6.0
        w = -5.0   # downward
        m, dof_mgr = make_simply_supported(L)
        lc = LoadCase(id="udl").update_distributed_load("e1", q=w)
        U, K, F = solve(m, dof_mgr, lc)
        EI = MAT.E * 10_000.0
        # end rotation at A (DOF 2): theta_A = wL^3/(24EI)
        theta_A = w * L**3 / (24 * EI)
        assert U[2] == pytest.approx(theta_A, rel=1e-8)

    def test_two_span_symmetry(self):
        """
        Two equal spans with equal UDL. By symmetry:
          - vertical displacement of B must be downward (max deflection)
          - rotation at B must be zero
        """
        L = 5.0
        w = -10.0
        m, dof_mgr = make_two_span(L)
        lc = (
            LoadCase(id="sym")
            .update_distributed_load("e1", q=w)
            .update_distributed_load("e2", q=w)
        )
        U, K, F = solve(m, dof_mgr, lc)
        # node B has dofs [3,4,5]
        assert U[4] < 0.0          # downward displacement
        assert U[5] == pytest.approx(0.0, abs=1e-10)   # zero rotation by symmetry

    def test_reactions_global_equilibrium(self):
        """Sum of external forces + reactions must be zero."""
        L = 6.0
        P = 10.0
        m, dof_mgr = make_simply_supported(L)
        lc = LoadCase(id="pt").update_nodal_load("B", [0.0, -P, 0.0])
        U, K, F = solve(m, dof_mgr, lc)
        handler = PlainConstraintHandler()
        handler.initialize(m, dof_mgr)
        reactions = handler.get_reactions(U, K, F)
        # sum of all vertical forces (external + reactions) = 0
        total_fy = -P   # applied
        for r in reactions.values():
            total_fy += r[1]   # reaction Ry (upward = positive)
        assert total_fy == pytest.approx(0.0, abs=1e-8)
