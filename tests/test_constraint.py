"""
tests/test_constraint.py
========================
Tests for ConstraintHandler (ABC), PlainConstraintHandler.

Verifies DOF elimination, symmetry preservation, reaction computation,
and integration with a simple cantilever model.
"""

import pytest
import numpy as np
from scipy.sparse import csr_matrix

from node import Node, Support
from section import ElasticSection
from element import EulerBernoulliBeam
from material import ElasticMaterial
from model import Model, DOFManager
from constraint import PlainConstraintHandler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_cantilever():
    """
    Single beam, node A fixed, node B free.
    Returns (model, dof_mgr, handler).
    """
    m = Model()
    m.add(Node(id="A", x=0.0, y=0.0))
    m.add(Node(id="B", x=5.0, y=0.0))
    m.add(ElasticSection(id="sec", material_id="steel", A=100.0, I=10_000.0))
    m.add(EulerBernoulliBeam(id="e1", node_ids=("A", "B"), section_id="sec"))
    m.add(Support(id="fix", node_id="A", ux=True, uy=True, rz=True))
    dof_mgr = m.finalize()
    return m, dof_mgr


def make_simply_supported():
    """
    Single beam, node A pinned (ux, uy), node B roller (uy only).
    Returns (model, dof_mgr).
    """
    m = Model()
    m.add(Node(id="A", x=0.0, y=0.0))
    m.add(Node(id="B", x=6.0, y=0.0))
    m.add(ElasticSection(id="sec", material_id="steel", A=100.0, I=10_000.0))
    m.add(EulerBernoulliBeam(id="e1", node_ids=("A", "B"), section_id="sec"))
    m.add(Support(id="pin", node_id="A", ux=True, uy=True, rz=False))
    m.add(Support(id="roll", node_id="B", ux=False, uy=True, rz=False))
    dof_mgr = m.finalize()
    return m, dof_mgr


def assemble_K(model, dof_mgr):
    """Minimal assembler for testing — assembles K without Assembler class."""
    mat = ElasticMaterial(id="steel", E=210_000.0, rho=7.85e-3)
    from scipy.sparse import lil_matrix
    K = lil_matrix((dof_mgr.n_dofs, dof_mgr.n_dofs))
    for elem_id, elem in model.elements.items():
        nodes = [model.get(nid) for nid in elem.node_ids]
        sec = model.sections[elem.section_id]
        K_e = elem.get_local_stiffness(nodes, sec, mat)
        dofs = dof_mgr.get_elem_dofs(elem_id)
        for i, gi in enumerate(dofs):
            for j, gj in enumerate(dofs):
                K[gi, gj] += K_e[i, j]
    return K.tocsr()


# ---------------------------------------------------------------------------
# PlainConstraintHandler — initialize()
# ---------------------------------------------------------------------------

class TestInitialize:

    def test_constrained_dofs_cantilever(self):
        m, dof_mgr = make_cantilever()
        h = PlainConstraintHandler()
        h.initialize(m, dof_mgr)
        # node A has dofs [0,1,2], all three constrained
        assert sorted(h._constrained_dofs) == [0, 1, 2]

    def test_constrained_dofs_simply_supported(self):
        m, dof_mgr = make_simply_supported()
        h = PlainConstraintHandler()
        h.initialize(m, dof_mgr)
        # node A: dofs [0,1] (ux,uy); node B: dof [4] (uy)
        assert sorted(h._constrained_dofs) == [0, 1, 4]

    def test_node_constrained_dofs_mapping(self):
        m, dof_mgr = make_cantilever()
        h = PlainConstraintHandler()
        h.initialize(m, dof_mgr)
        assert "A" in h._node_constrained_dofs
        assert sorted(h._node_constrained_dofs["A"]) == [0, 1, 2]

    def test_not_initialized_raises_on_apply(self):
        h = PlainConstraintHandler()
        K = csr_matrix(np.eye(6))
        F = np.zeros(6)
        with pytest.raises(RuntimeError, match="initialize"):
            h.apply(K, F)

    def test_not_initialized_raises_on_reactions(self):
        h = PlainConstraintHandler()
        with pytest.raises(RuntimeError, match="initialize"):
            h.get_reactions(np.zeros(6), csr_matrix(np.eye(6)), np.zeros(6))

    def test_global_to_local_mapping(self):
        m, dof_mgr = make_cantilever()
        h = PlainConstraintHandler()
        h.initialize(m, dof_mgr)
        # global DOF 0 -> local 0 (ux), 1 -> 1 (uy), 2 -> 2 (rz)
        assert h._global_to_local[0] == 0
        assert h._global_to_local[1] == 1
        assert h._global_to_local[2] == 2


# ---------------------------------------------------------------------------
# PlainConstraintHandler — apply()
# ---------------------------------------------------------------------------

class TestApply:

    def setup_method(self):
        self.m, self.dof_mgr = make_cantilever()
        self.h = PlainConstraintHandler()
        self.h.initialize(self.m, self.dof_mgr)
        self.K = assemble_K(self.m, self.dof_mgr)
        self.F = np.zeros(self.dof_mgr.n_dofs)
        self.F[4] = 10.0   # tip load in y direction

    def test_apply_returns_new_objects(self):
        K_mod, F_mod = self.h.apply(self.K, self.F)
        assert K_mod is not self.K
        assert F_mod is not self.F

    def test_constrained_diagonal_is_one(self):
        K_mod, F_mod = self.h.apply(self.K, self.F)
        K_dense = K_mod.toarray()
        for d in [0, 1, 2]:
            assert K_dense[d, d] == pytest.approx(1.0)

    def test_constrained_row_is_zero_except_diagonal(self):
        K_mod, F_mod = self.h.apply(self.K, self.F)
        K_dense = K_mod.toarray()
        for d in [0, 1, 2]:
            row = K_dense[d, :].copy()
            row[d] = 0.0
            np.testing.assert_allclose(row, 0.0, atol=1e-12)

    def test_constrained_column_is_zero_except_diagonal(self):
        K_mod, F_mod = self.h.apply(self.K, self.F)
        K_dense = K_mod.toarray()
        for d in [0, 1, 2]:
            col = K_dense[:, d].copy()
            col[d] = 0.0
            np.testing.assert_allclose(col, 0.0, atol=1e-12)

    def test_constrained_rhs_is_zero(self):
        K_mod, F_mod = self.h.apply(self.K, self.F)
        for d in [0, 1, 2]:
            assert F_mod[d] == 0.0

    def test_free_rhs_preserved(self):
        K_mod, F_mod = self.h.apply(self.K, self.F)
        assert F_mod[4] == pytest.approx(10.0)

    def test_modified_K_is_symmetric(self):
        K_mod, F_mod = self.h.apply(self.K, self.F)
        K_dense = K_mod.toarray()
        np.testing.assert_allclose(K_dense, K_dense.T, atol=1e-10)

    def test_modified_system_is_solvable(self):
        K_mod, F_mod = self.h.apply(self.K, self.F)
        from scipy.sparse.linalg import spsolve
        U = spsolve(K_mod, F_mod)
        assert np.all(np.isfinite(U))

    def test_constrained_dofs_zero_in_solution(self):
        K_mod, F_mod = self.h.apply(self.K, self.F)
        from scipy.sparse.linalg import spsolve
        U = spsolve(K_mod, F_mod)
        for d in [0, 1, 2]:
            assert U[d] == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# PlainConstraintHandler — get_reactions()
# ---------------------------------------------------------------------------

class TestReactions:

    def _solve_cantilever(self, P=10.0):
        m, dof_mgr = make_cantilever()
        h = PlainConstraintHandler()
        h.initialize(m, dof_mgr)
        K = assemble_K(m, dof_mgr)
        F = np.zeros(dof_mgr.n_dofs)
        F[4] = P
        K_mod, F_mod = h.apply(K, F)
        from scipy.sparse.linalg import spsolve
        U = spsolve(K_mod, F_mod)
        return h, U, K, F, P

    def test_reactions_returns_dict(self):
        h, U, K, F, P = self._solve_cantilever()
        reactions = h.get_reactions(U, K, F)
        assert isinstance(reactions, dict)
        assert "A" in reactions

    def test_reactions_shape(self):
        h, U, K, F, P = self._solve_cantilever()
        reactions = h.get_reactions(U, K, F)
        assert reactions["A"].shape == (3,)

    def test_vertical_reaction_equilibrium(self):
        """Ry at A must equal -P (upward reaction)."""
        P = 10.0
        h, U, K, F, P = self._solve_cantilever(P)
        reactions = h.get_reactions(U, K, F)
        # Ry is component index 1
        assert reactions["A"][1] == pytest.approx(-P, rel=1e-6)

    def test_horizontal_reaction_zero(self):
        """No horizontal load -> Rx = 0."""
        h, U, K, F, P = self._solve_cantilever()
        reactions = h.get_reactions(U, K, F)
        assert reactions["A"][0] == pytest.approx(0.0, abs=1e-8)

    def test_moment_reaction(self):
        """Fixed end moment = P * L."""
        L = 5.0
        P = 10.0
        h, U, K, F, P = self._solve_cantilever(P)
        reactions = h.get_reactions(U, K, F)
        # Mrz at A = -P*L (hogging moment — negative by sign convention)
        assert reactions["A"][2] == pytest.approx(-P * L, rel=1e-6)

    def test_simply_supported_reactions(self):
        """
        Simply supported beam, midspan load P.
        Reactions: Ra = Rb = P/2.
        """
        P = 20.0
        L = 6.0
        m, dof_mgr = make_simply_supported()
        h = PlainConstraintHandler()
        h.initialize(m, dof_mgr)
        K = assemble_K(m, dof_mgr)
        F = np.zeros(dof_mgr.n_dofs)
        # midspan load not straightforward for single element —
        # apply P/2 at each node as equivalent nodal loads
        F[1] = P / 2   # uy at node A
        F[4] = P / 2   # uy at node B
        K_mod, F_mod = h.apply(K, F)
        from scipy.sparse.linalg import spsolve
        U = spsolve(K_mod, F_mod)
        reactions = h.get_reactions(U, K, F)
        # both supports carry equal load
        assert reactions["A"][1] == pytest.approx(-P / 2, rel=1e-6)
        assert reactions["B"][1] == pytest.approx(-P / 2, rel=1e-6)


# ---------------------------------------------------------------------------
# PlainConstraintHandler — serialization
# ---------------------------------------------------------------------------

class TestConstraintSerialization:

    def test_round_trip(self):
        h = PlainConstraintHandler()
        d = h.to_dict()
        assert d["type"] == "PlainConstraintHandler"
        h2 = PlainConstraintHandler.from_dict(d)
        assert isinstance(h2, PlainConstraintHandler)
