"""
assembler.py
============
Assembler.

Responsible for building the global stiffness matrix K, mass matrix M,
and force vector F from individual element contributions.

Key principles:
  - The Assembler resolves node and section IDs into objects BEFORE calling
    element methods. Elements never access the Model directly.
  - Uses lil_matrix for assembly (efficient row insertion), then converts
    to csr for the linear solve.
  - geometry=None -> linear analysis (elements use initial coordinates).
  - geometry provided -> geometric nonlinearity (Phase 2, not yet active).

Distributed loads:
  - Equivalent nodal forces are computed by each element via
    get_equivalent_nodal_forces_distributed(), a helper defined here
    that follows the standard Euler-Bernoulli fixed-end force formulas.
  - The element handles the sign convention internally.
  - reference="local"  -> load acts in local element axes.
  - reference="global" -> load is transformed to local before integration.
"""

from __future__ import annotations

import numpy as np
from numpy import ndarray
from scipy.sparse import lil_matrix, csr_matrix

from model import Model, DOFManager
from element import Element, EulerBernoulliBeam, DeformedGeometry
from load_case import LoadCase, DistributedLoad
from material import Material, ElasticMaterial


# ---------------------------------------------------------------------------
# Assembler
# ---------------------------------------------------------------------------

class Assembler:
    """
    Builds global K, M, and F matrices from element contributions.

    The Assembler is stateless — all information is passed as parameters.
    It can be called multiple times during an analysis (e.g. once per
    Newton-Raphson iteration for nonlinear problems).
    """

    # ------------------------------------------------------------------
    # Stiffness matrix
    # ------------------------------------------------------------------

    def assemble_K(
        self,
        model: Model,
        dof_mgr: DOFManager,
        geometry: DeformedGeometry | None = None,
    ) -> csr_matrix:
        """
        Assemble the global stiffness matrix.

        Parameters
        ----------
        model    : finalized Model (materials resolved from model.materials)
        dof_mgr  : DOFManager from model.finalize()
        geometry : None for linear; DeformedGeometry for Phase 2

        Returns
        -------
        K : (n_dofs, n_dofs) sparse matrix in csr format
        """
        n = dof_mgr.n_dofs
        K = lil_matrix((n, n))

        for elem_id, elem in model.elements.items():
            nodes = [model.get(nid) for nid in elem.node_ids]
            section = model.sections[elem.section_id]
            material = self._resolve_material(section, model.materials)

            K_e = elem.get_local_stiffness(nodes, section, material, geometry)
            dofs = dof_mgr.get_elem_dofs(elem_id)
            self._scatter(K, K_e, dofs)

        return K.tocsr()

    # ------------------------------------------------------------------
    # Mass matrix
    # ------------------------------------------------------------------

    def assemble_M(
        self,
        model: Model,
        dof_mgr: DOFManager,
    ) -> csr_matrix:
        """
        Assemble the global consistent mass matrix.

        Two-pass assembly:
          Pass 1 — distributed element mass (consistent mass matrix).
          Pass 2 — concentrated nodal masses (added to diagonal).

        Parameters
        ----------
        model    : finalized Model (materials resolved from model.materials)
        dof_mgr  : DOFManager from model.finalize()

        Returns
        -------
        M : (n_dofs, n_dofs) sparse matrix in csr format
        """
        n = dof_mgr.n_dofs
        M = lil_matrix((n, n))

        # Pass 1: element consistent mass matrices
        for elem_id, elem in model.elements.items():
            nodes = [model.get(nid) for nid in elem.node_ids]
            section = model.sections[elem.section_id]
            material = self._resolve_material(section, model.materials)

            M_e = elem.get_local_mass(nodes, section, material)
            dofs = dof_mgr.get_elem_dofs(elem_id)
            self._scatter(M, M_e, dofs)

        # Pass 2: concentrated nodal masses (diagonal only)
        for nm in model.nodal_masses.values():
            dofs = dof_mgr.get_node_dofs(nm.node_id)
            M[dofs[0], dofs[0]] += nm.mx
            M[dofs[1], dofs[1]] += nm.my
            M[dofs[2], dofs[2]] += nm.I_theta

        return M.tocsr()

    # ------------------------------------------------------------------
    # Force vector
    # ------------------------------------------------------------------

    def assemble_F(
        self,
        load_case: LoadCase,
        model: Model,
        dof_mgr: DOFManager,
    ) -> ndarray:
        """
        Assemble the global external force vector.

        Handles:
          - Nodal loads (direct scatter to global DOFs).
          - Uniformly distributed loads (converted to equivalent nodal
            forces via fixed-end force formulas, then scattered).

        Parameters
        ----------
        load_case : LoadCase with spatial load description
        model     : finalized Model (materials resolved from model.materials)
        dof_mgr   : DOFManager from model.finalize()

        Returns
        -------
        F : (n_dofs,) force vector
        """
        n = dof_mgr.n_dofs
        F = np.zeros(n)

        # --- nodal loads ---
        for node_id, load in load_case.nodal_loads.items():
            dofs = dof_mgr.get_node_dofs(node_id)
            F[dofs] += load

        # --- distributed loads -> equivalent nodal forces ---
        for elem_id, dl in load_case.distributed_loads.items():
            elem = model.elements[elem_id]
            nodes = [model.get(nid) for nid in elem.node_ids]
            section = model.sections[elem.section_id]
            material = self._resolve_material(section, model.materials)

            F_eq = self._equivalent_nodal_forces(dl, nodes, elem, material)
            dofs = dof_mgr.get_elem_dofs(elem_id)
            F[dofs] += F_eq

        return F

    # ------------------------------------------------------------------
    # Equivalent nodal forces for distributed loads
    # ------------------------------------------------------------------

    def _equivalent_nodal_forces(
        self,
        dl: DistributedLoad,
        nodes: list,
        elem: Element,
        material: Material,
    ) -> ndarray:
        """
        Compute the 6-component equivalent nodal force vector for a
        uniformly distributed load on a beam element.

        Fixed-end forces for a uniform load q (force/length) acting
        transversely on a beam of length L:

          F_i = q*L/2        (shear at node i)
          M_i = q*L^2/12     (moment at node i)
          F_j = q*L/2        (shear at node j)
          M_j = -q*L^2/12    (moment at node j)

        For an axial load q (force/length) along the element:
          N_i = q*L/2
          N_j = q*L/2

        If reference="global", the load direction is first transformed
        to the local element frame before applying the formulas.

        DOF order (local): [u_i, v_i, theta_i, u_j, v_j, theta_j]
        """
        if not isinstance(elem, EulerBernoulliBeam):
            raise NotImplementedError(
                f"Equivalent nodal forces for distributed loads are only "
                f"implemented for EulerBernoulliBeam (got "
                f"{type(elem).__name__})."
            )

        L, alpha = elem._get_length_and_angle(nodes)
        q = dl.q

        # transform load to local frame if needed
        if dl.reference == "global":
            q_local = self._global_load_to_local(q, dl.direction, alpha)
            q_axial = q_local[0]
            q_transverse = q_local[1]
        else:
            # load already in local frame
            q_axial = q if dl.direction == "x" else 0.0
            q_transverse = q if dl.direction == "y" else 0.0

        # fixed-end forces in local coordinates
        F_local = np.zeros(6)

        # transverse load (y direction)
        if q_transverse != 0.0:
            F_local[1] += q_transverse * L / 2        # V at node i
            F_local[2] += q_transverse * L**2 / 12    # M at node i
            F_local[4] += q_transverse * L / 2        # V at node j
            F_local[5] -= q_transverse * L**2 / 12    # M at node j

        # axial load (x direction)
        if q_axial != 0.0:
            F_local[0] += q_axial * L / 2             # N at node i
            F_local[3] += q_axial * L / 2             # N at node j

        # transform to global coordinates
        T = elem._rotation_matrix(alpha)
        return T.T @ F_local

    def _global_load_to_local(
        self, q: float, direction: str, alpha: float
    ) -> ndarray:
        """
        Decompose a global load q (in direction 'x' or 'y') into
        local [q_x_local, q_y_local] components.

        alpha : element angle with respect to global x axis
        """
        c = np.cos(alpha)
        s = np.sin(alpha)

        if direction == "y":
            # global (0, q) -> local
            q_x_local = -q * s    # projection onto local x
            q_y_local =  q * c    # projection onto local y
        else:
            # global (q, 0) -> local
            q_x_local =  q * c
            q_y_local = -q * s

        return np.array([q_x_local, q_y_local])

    # ------------------------------------------------------------------
    # Scatter helper
    # ------------------------------------------------------------------

    @staticmethod
    def _scatter(
        K: lil_matrix, K_e: ndarray, dofs: list[int]
    ) -> None:
        """Add element matrix K_e into global K at positions dofs."""
        for i, gi in enumerate(dofs):
            for j, gj in enumerate(dofs):
                K[gi, gj] += K_e[i, j]

    # ------------------------------------------------------------------
    # Material resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_material(section, materials: dict) -> Material:
        """
        Return the Material object for the given section.

        Looks up section.material_id in the materials dict.
        Raises a clear error if not found.
        """
        mat_id = section.material_id
        if mat_id not in materials:
            raise ValueError(
                f"Material '{mat_id}' referenced by section '{section.id}' "
                f"not found. Pass a materials dict to the Assembler methods."
            )
        return materials[mat_id]

    # ------------------------------------------------------------------
    # Condition number check (educational)
    # ------------------------------------------------------------------

    @staticmethod
    def check_conditioning(K: csr_matrix, threshold: float = 1e12) -> None:
        """
        Estimate the condition number of K and warn if it exceeds threshold.

        Uses the ratio of largest to smallest singular value via
        scipy.sparse.linalg. For large systems this is an approximation.

        Called optionally by the solver (SolverConfig.check_equilibrium).
        """
        from scipy.sparse.linalg import eigsh
        try:
            n = K.shape[0]
            k = min(6, n - 1)
            largest = eigsh(K, k=k, which="LM", return_eigenvectors=False)
            smallest = eigsh(K, k=k, which="SM", return_eigenvectors=False,
                             sigma=0.0)
            cond = abs(largest.max()) / (abs(smallest.min()) + 1e-300)
            if cond > threshold:
                print(
                    f"[Assembler] WARNING: condition number of K is {cond:.2e} "
                    f"(threshold {threshold:.0e}). "
                    f"The system may be ill-conditioned."
                )
        except Exception:
            pass   # conditioning check is advisory — never block the solve
