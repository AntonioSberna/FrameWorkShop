"""
constraint.py
=============
ConstraintHandler (ABC), PlainConstraintHandler.

The ConstraintHandler is responsible for:
  1. Reading Constraint objects from the Model and translating them into
     global DOF indices (via DOFManager).
  2. Modifying K and F to enforce boundary conditions.
  3. Computing constraint reactions after the solution.

PlainConstraintHandler uses direct elimination (penalty-free):
  - Rows and columns of constrained DOFs are zeroed out.
  - The diagonal is set to 1.0 and the RHS to the prescribed displacement
    (zero for fixed constraints).

This is the simplest and most numerically stable approach for fixed constraints.
Lagrange multiplier and penalty methods are deferred to Phase 4.

Reactions are computed as:
    R = K_free * U_full - F_external
evaluated only at the constrained DOF indices.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy import ndarray
from scipy.sparse import lil_matrix, csr_matrix

from model import Model, DOFManager
from node import Constraint


# ---------------------------------------------------------------------------
# ConstraintHandler (ABC)
# ---------------------------------------------------------------------------

class ConstraintHandler(ABC):
    """
    Abstract interface for enforcing boundary conditions.

    Workflow (called by the solver):
        handler.initialize(model, dof_mgr)
        K_mod, F_mod = handler.apply(K, F)
        U = solve(K_mod, F_mod)
        reactions = handler.get_reactions(U, K_original, F)
    """

    @abstractmethod
    def initialize(self, model: Model, dof_mgr: DOFManager) -> None:
        """
        Read constraints from the model and build internal DOF index maps.

        Called once per analysis stage (DOF numbering may change between
        stages). Must be called before apply() or get_reactions().
        """
        ...

    @abstractmethod
    def apply(
        self, K: csr_matrix, F: ndarray
    ) -> tuple[csr_matrix, ndarray]:
        """
        Return a modified (K, F) pair with boundary conditions enforced.

        The original K and F are not modified in place — new objects
        are returned. The solver uses the modified pair for the linear solve.

        Parameters
        ----------
        K : global stiffness matrix (n_dofs x n_dofs), csr format
        F : global force vector (n_dofs,)

        Returns
        -------
        K_mod, F_mod : modified stiffness and force
        """
        ...

    @abstractmethod
    def get_reactions(
        self,
        U: ndarray,
        K_free: csr_matrix,
        F: ndarray,
    ) -> dict[str | int, ndarray]:
        """
        Compute constraint reactions after the linear solve.

        Parameters
        ----------
        U      : converged global displacement vector (n_dofs,)
        K_free : original (unmodified) stiffness matrix
        F      : external force vector (before constraint application)

        Returns
        -------
        dict : node_id -> [Rx, Ry, Mrz]  for each constrained node
        """
        ...

    @abstractmethod
    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dictionary."""
        ...


# ---------------------------------------------------------------------------
# PlainConstraintHandler
# ---------------------------------------------------------------------------

class PlainConstraintHandler(ConstraintHandler):
    """
    Enforce boundary conditions by direct DOF elimination.

    For each constrained DOF d:
      - K[d, :] = 0
      - K[:, d] = 0
      - K[d, d] = 1
      - F[d]    = prescribed displacement (0 for fixed constraints)

    This zeroing approach preserves matrix symmetry and is numerically
    clean. It is equivalent to static condensation for zero-prescribed
    displacements.

    Reactions are recovered as:
        R[d] = (K_original @ U)[d] - F[d]

    Only fixed (zero-displacement) constraints are handled in Phase 0.
    Non-zero prescribed displacements are deferred to Phase 4.
    """

    def __init__(self) -> None:
        # constrained_dofs : list of global DOF indices to constrain
        self._constrained_dofs: list[int] = []
        # node_constrained_dofs : node_id -> list of constrained DOF indices
        self._node_constrained_dofs: dict[str | int, list[int]] = {}
        self._initialized: bool = False

    def initialize(self, model: Model, dof_mgr: DOFManager) -> None:
        """
        Translate Constraint objects into lists of global DOF indices.

        DOF name to local index mapping:
            ux -> 0,  uy -> 1,  rz -> 2
        """
        self._constrained_dofs = []
        self._node_constrained_dofs = {}

        dof_name_to_local = {"ux": 0, "uy": 1, "rz": 2}

        for sup in model.constraints.values():
            node_dofs = dof_mgr.get_node_dofs(sup.node_id)
            constrained = []
            for dof_name in sup.constrained_dofs():
                local_idx = dof_name_to_local[dof_name]
                global_idx = node_dofs[local_idx]
                self._constrained_dofs.append(global_idx)
                constrained.append(global_idx)
            if constrained:
                self._node_constrained_dofs[sup.node_id] = constrained

        self._initialized = True

    def apply(
        self, K: csr_matrix, F: ndarray
    ) -> tuple[csr_matrix, ndarray]:
        """
        Apply boundary conditions by zeroing rows/columns and setting
        diagonal to 1, RHS to 0 (fixed constraint).

        Works on a copy — original K and F are not modified.
        """
        if not self._initialized:
            raise RuntimeError(
                "PlainConstraintHandler.apply() called before initialize(). "
                "Call handler.initialize(model, dof_mgr) first."
            )

        # work in lil format for efficient row/column access
        K_mod = lil_matrix(K)
        F_mod = F.copy()

        for d in self._constrained_dofs:
            # zero the row and column
            K_mod[d, :] = 0.0
            K_mod[:, d] = 0.0
            # unit diagonal — prescribed displacement = 0
            K_mod[d, d] = 1.0
            F_mod[d] = 0.0

        return K_mod.tocsr(), F_mod

    def get_reactions(
        self,
        U: ndarray,
        K_free: csr_matrix,
        F: ndarray,
    ) -> dict[str | int, ndarray]:
        """
        Compute constraint reactions.

        R = K_free @ U - F
        evaluated at constrained DOF indices only.

        Returns node_id -> [Rx, Ry, Mrz] for each constrained node.
        Only the constrained components are non-zero — free components
        are set to zero.
        """
        if not self._initialized:
            raise RuntimeError(
                "PlainConstraintHandler.get_reactions() called before "
                "initialize(). Call handler.initialize(model, dof_mgr) first."
            )

        # full residual vector
        residual = K_free @ U - F

        reactions = {}
        for node_id, constrained in self._node_constrained_dofs.items():
            # find all DOFs for this node (3 total)
            # by scanning which of node_dofs are in constrained list
            r = np.zeros(3)
            # reconstruct node dof base from constrained list
            # (we know constrained contains global indices)
            for global_idx in constrained:
                # find local position (0=ux, 1=uy, 2=rz) from the
                # offset within the node block
                # node block starts at min(constrained for this node) - offset
                # simpler: store the (local_idx, global_idx) pairs
                pass
            reactions[node_id] = r

        # rebuild properly using stored mapping
        reactions = self._compute_reactions(residual)
        return reactions

    def _compute_reactions(
        self, residual: ndarray
    ) -> dict[str | int, ndarray]:
        """
        Extract reaction components from the full residual vector.

        For each constrained node, reads the reaction at each constrained
        global DOF and places it in the correct local position [ux,uy,rz].
        """
        reactions = {}
        for node_id, constrained_global in self._node_constrained_dofs.items():
            r = np.zeros(3)
            for global_idx in constrained_global:
                # find local index (0,1,2) from the node's DOF block
                # node DOF block = [base, base+1, base+2]
                # base = min of the 3 node dofs = constrained_global[0]
                # but we need the full node dof list — store it at init
                local_idx = self._global_to_local[global_idx]
                r[local_idx] = residual[global_idx]
            reactions[node_id] = r
        return reactions

    def initialize(self, model: Model, dof_mgr: DOFManager) -> None:
        """
        Translate Constraint objects into lists of global DOF indices.

        Also builds _global_to_local mapping for reaction extraction.
        """
        self._constrained_dofs = []
        self._node_constrained_dofs = {}
        self._global_to_local: dict[int, int] = {}

        dof_name_to_local = {"ux": 0, "uy": 1, "rz": 2}

        for sup in model.constraints.values():
            node_dofs = dof_mgr.get_node_dofs(sup.node_id)
            constrained = []
            for dof_name in sup.constrained_dofs():
                local_idx = dof_name_to_local[dof_name]
                global_idx = node_dofs[local_idx]
                self._constrained_dofs.append(global_idx)
                constrained.append(global_idx)
                self._global_to_local[global_idx] = local_idx
            if constrained:
                self._node_constrained_dofs[sup.node_id] = constrained

        self._initialized = True

    def to_dict(self) -> dict:
        return {"type": "PlainConstraintHandler"}

    @classmethod
    def from_dict(cls, data: dict) -> PlainConstraintHandler:
        return cls()

    def __repr__(self) -> str:
        n = len(self._constrained_dofs)
        return f"PlainConstraintHandler(constrained_dofs={n})"
