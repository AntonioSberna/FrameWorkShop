"""
element.py
==========
ElementState, Element (ABC), EulerBernoulliBeam.

The element is the core computational object of the framework.
It knows its geometry (node IDs) and section (section ID), but never
holds direct references to Node or Section objects — those are resolved
externally by the Assembler and passed as parameters.

Local coordinate system convention (FIXED, never change):
  - local x axis : from node i to node j
  - local y axis : right-hand rule with respect to the section frame
  - positive moment : tension in fibers with y > 0

DOF order (local and global): [u_i, v_i, theta_i, u_j, v_j, theta_j]
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from numpy import ndarray

from section import Section, SectionState, ElasticSection
from material import Material


# ---------------------------------------------------------------------------
# DeformedGeometry (forward declaration — full impl in Phase 2)
# ---------------------------------------------------------------------------

class DeformedGeometry:
    """
    Current nodal coordinates after deformation.

    In Phase 0 (linear) this is never instantiated — elements receive
    geometry=None and use initial node coordinates.
    In Phase 2 (geometric nonlinearity) the solver updates this object
    at every iteration and passes it to the Assembler.
    """

    def __init__(self, coords: dict) -> None:
        # node_id -> (x_current, y_current)
        self.coords: dict[str | int, tuple[float, float]] = coords

    def get(self, node_id: str | int) -> tuple[float, float]:
        return self.coords[node_id]


# ---------------------------------------------------------------------------
# SectionForces
# ---------------------------------------------------------------------------

@dataclass
class SectionForces:
    """
    Internal forces at a cross-section.

    N     : axial force
    V     : shear force
    M     : bending moment
    xi    : dimensionless coordinate along element [0, 1]
    """
    N: float
    V: float
    M: float
    xi: float


# ---------------------------------------------------------------------------
# ElementState
# ---------------------------------------------------------------------------

class ElementState:
    """
    Internal state of an element.

    Aggregates one SectionState per Gauss point.
    commit() / revert() delegate to all Gauss point states.

    In Phase 0 (linear elastic) the section states are not strictly
    needed for the solution, but they are populated so that the
    post-processing API is uniform across linear and nonlinear analyses.
    """

    def __init__(self, n_gauss: int) -> None:
        self.gauss_states: list[SectionState] = [
            SectionState() for _ in range(n_gauss)
        ]

    def commit(self) -> None:
        """Commit all Gauss point states."""
        for gs in self.gauss_states:
            gs.commit()

    def revert(self) -> None:
        """Revert all Gauss point states."""
        for gs in self.gauss_states:
            gs.revert()

    def to_dict(self) -> dict:
        return {
            "gauss_states": [gs.to_dict() for gs in self.gauss_states],
        }

    @classmethod
    def from_dict(cls, data: dict) -> ElementState:
        from section import SectionState
        n = len(data["gauss_states"])
        obj = cls(n_gauss=n)
        obj.gauss_states = [
            SectionState.from_dict(d) for d in data["gauss_states"]
        ]
        return obj


# ---------------------------------------------------------------------------
# Element (ABC)
# ---------------------------------------------------------------------------

class Element(ABC):
    """
    Abstract interface for all elements.

    Principles enforced here:
      - node_ids and section_id are stored as IDs, never as objects.
      - Node and Section objects are always passed as parameters by the
        Assembler — the element never fetches them from a registry.
      - update() modifies only the trial state.
      - commit() / revert() are concrete and delegate to self.state.

    Subclasses must set self.state in __init__ before calling super().__init__
    or immediately after, since commit/revert are already concrete here.
    """

    def __init__(
        self,
        id: str | int,
        node_ids: tuple[str | int, str | int],
        section_id: str | int,
        n_gauss: int = 2,
    ) -> None:
        self.id = id
        self.node_ids = node_ids      # (id_i, id_j) — reference by ID
        self.section_id = section_id  # reference by ID
        self.n_gauss = n_gauss
        self.state = ElementState(n_gauss)

    # ------------------------------------------------------------------
    # Stiffness and mass
    # ------------------------------------------------------------------

    @abstractmethod
    def get_local_stiffness(
        self,
        nodes: list,
        section: Section,
        material: Material,
        geometry: DeformedGeometry | None = None,
    ) -> ndarray:
        """
        Return the 6x6 local stiffness matrix.

        Parameters
        ----------
        nodes    : [node_i, node_j] resolved by the Assembler
        section  : Section object resolved by the Assembler
        material : Material object resolved by the Assembler
        geometry : None for linear analysis (use initial coords);
                   DeformedGeometry for geometric nonlinearity (Phase 2)

        Returns
        -------
        K_local : (6, 6) ndarray
        """
        ...

    @abstractmethod
    def get_local_mass(
        self,
        nodes: list,
        section: Section,
        material: Material,
    ) -> ndarray:
        """
        Return the 6x6 local consistent mass matrix.

        Parameters
        ----------
        nodes    : [node_i, node_j] resolved by the Assembler
        section  : Section object resolved by the Assembler
        material : Material object resolved by the Assembler

        Returns
        -------
        M_local : (6, 6) ndarray
        """
        ...

    # ------------------------------------------------------------------
    # Internal forces and section forces
    # ------------------------------------------------------------------

    @abstractmethod
    def get_internal_forces(
        self,
        nodes: list,
        section: Section,
        material: Material,
    ) -> ndarray:
        """
        Return the 6-component vector of internal nodal forces
        in global coordinates.

        Parameters
        ----------
        nodes    : [node_i, node_j]
        section  : Section object
        material : Material object

        Returns
        -------
        F_int : (6,) ndarray in global coordinates
        """
        ...

    @abstractmethod
    def get_section_forces(
        self,
        xi: float,
        nodes: list,
        section: Section,
        material: Material,
    ) -> SectionForces:
        """
        Return internal forces at a point along the element.

        Parameters
        ----------
        xi       : dimensionless coordinate in [0, 1]
        nodes    : [node_i, node_j]
        section  : Section object
        material : Material object

        Returns
        -------
        SectionForces(N, V, M, xi)
        """
        ...

    # ------------------------------------------------------------------
    # Equivalent nodal forces (thermal, imposed strains, prestress)
    # ------------------------------------------------------------------

    @abstractmethod
    def get_equivalent_forces(
        self,
        nodes: list,
        section: Section,
        material: Material,
        imposed_strain,
    ) -> ndarray:
        """
        Return equivalent nodal forces for imposed strains.

        The sign convention is handled by the element:
            F_eq = -integral(B^T * E * eps_0) dV
        The negative sign is intentional (thermal loads, prestress).

        Parameters
        ----------
        nodes          : [node_i, node_j]
        section        : Section object
        material       : Material object
        imposed_strain : strain field to impose (type depends on subclass)

        Returns
        -------
        F_eq : (6,) ndarray in global coordinates
        """
        ...

    # ------------------------------------------------------------------
    # State update
    # ------------------------------------------------------------------

    @abstractmethod
    def update(
        self,
        nodes: list,
        section: Section,
        material: Material,
        U_global: ndarray,
    ) -> None:
        """
        Update the trial state from the global displacement vector.

        Only modifies self.state (trial values).
        Never modifies committed values — use commit() for that.

        Parameters
        ----------
        nodes    : [node_i, node_j]
        section  : Section object
        material : Material object
        U_global : full global displacement vector (n_dofs,)
        """
        ...

    def commit(self) -> None:
        """Commit trial state. Called by solver after convergence."""
        self.state.commit()

    def revert(self) -> None:
        """Revert to committed state. Called by solver after divergence."""
        self.state.revert()

    # ------------------------------------------------------------------
    # Geometry helpers (shared by all beam elements)
    # ------------------------------------------------------------------

    @staticmethod
    def _get_length_and_angle(nodes: list) -> tuple[float, float]:
        """
        Return element length L and angle alpha with respect to global x.

        Parameters
        ----------
        nodes : [node_i, node_j]  (or DeformedGeometry coords)

        Returns
        -------
        (L, alpha) : length and angle in radians
        """
        ni, nj = nodes
        dx = nj.x - ni.x
        dy = nj.y - ni.y
        L = np.sqrt(dx**2 + dy**2) # 2D Pitagora theorem
        if L < 1e-14:
            raise ValueError(
                f"Element has zero length: nodes {ni.id!r} and {nj.id!r} "
                f"are at the same position."
            )
        alpha = np.arctan2(dy, dx)
        return L, alpha

    @staticmethod
    def _rotation_matrix(alpha: float) -> ndarray:
        """
        Build the 6x6 rotation matrix from local to global coordinates.

        DOF order: [u_i, v_i, theta_i, u_j, v_j, theta_j]
        """
        c = np.cos(alpha)
        s = np.sin(alpha)
        T = np.zeros((6, 6))
        for k in [0, 3]:   # node i block, node j block
            T[k,   k  ] =  c
            T[k,   k+1] =  s
            T[k+1, k  ] = -s
            T[k+1, k+1] =  c
            T[k+2, k+2] =  1.0
        return T

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    @abstractmethod
    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dictionary."""
        ...

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict) -> Element:
        """Reconstruct instance from dictionary."""
        ...

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(id={self.id!r}, nodes={self.node_ids}, "
            f"section_id={self.section_id!r})"
        )


# ---------------------------------------------------------------------------
# EulerBernoulliBeam
# ---------------------------------------------------------------------------

class EulerBernoulliBeam(Element):
    """
    2D Euler-Bernoulli beam element (linear elastic, Phase 0).

    Assumptions:
      - Plane sections remain plane (no shear deformation).
      - Small displacements and rotations (linear geometry).
      - Prismatic cross-section (constant EA, EI along the element).

    The local stiffness matrix is the standard 6x6 Euler-Bernoulli matrix.
    The rotation matrix transforms between local and global coordinates.

    For geometric nonlinearity (Phase 2) this class will be replaced by
    NonlinearBeam, which accounts for updated geometry and geometric stiffness.

    DOF order (local and global): [u_i, v_i, theta_i, u_j, v_j, theta_j]
    """

    def __init__(
        self,
        id: str | int,
        node_ids: tuple[str | int, str | int],
        section_id: str | int,
        n_gauss: int = 2,
    ) -> None:
        super().__init__(
            id=id,
            node_ids=node_ids,
            section_id=section_id,
            n_gauss=n_gauss,
        )

    # ------------------------------------------------------------------
    # Core stiffness
    # ------------------------------------------------------------------

    def _local_stiffness(self, EA: float, EI: float, L: float) -> ndarray:
        """
        Build the 6x6 local stiffness matrix.

        Partitioned as:
          axial block  (u_i, u_j)            : EA/L * [[1,-1],[-1,1]]
          bending block (v_i,th_i, v_j,th_j) : EI/L^3 * classic 4x4
        """
        K = np.zeros((6, 6))

        # axial (DOFs 0, 3)
        K[0, 0] =  EA / L
        K[0, 3] = -EA / L
        K[3, 0] = -EA / L
        K[3, 3] =  EA / L

        # bending (DOFs 1, 2, 4, 5)
        b = [1, 2, 4, 5]
        c = EI / L**3
        Kb = c * np.array([
            [ 12,    6*L,  -12,   6*L],
            [  6*L,  4*L**2, -6*L,  2*L**2],
            [-12,   -6*L,   12,  -6*L],
            [  6*L,  2*L**2, -6*L,  4*L**2],
        ])
        for i, gi in enumerate(b):
            for j, gj in enumerate(b):
                K[gi, gj] = Kb[i, j]

        return K

    def get_local_stiffness(
        self,
        nodes: list,
        section: Section,
        material: Material,
        geometry: DeformedGeometry | None = None,
    ) -> ndarray:
        """
        Return the 6x6 stiffness matrix in GLOBAL coordinates.

        geometry=None -> use initial node coordinates (linear analysis).
        geometry provided -> use deformed coordinates (Phase 2, not yet active).
        """
        if geometry is None:
            L, alpha = self._get_length_and_angle(nodes)
        else:
            # Phase 2: use deformed coordinates
            # placeholder — raises until implemented
            raise NotImplementedError(
                "Deformed geometry not yet supported in EulerBernoulliBeam. "
                "Use NonlinearBeam for geometric nonlinearity (Phase 2)."
            )

        EA, EI = section.get_stiffness(material)
        K_local = self._local_stiffness(EA, EI, L)
        T = self._rotation_matrix(alpha)
        # K_global = T^T * K_local * T
        return T.T @ K_local @ T

    def get_local_mass(
        self,
        nodes: list,
        section: Section,
        material: Material,
    ) -> ndarray:
        """
        Return the 6x6 consistent mass matrix in GLOBAL coordinates.
        """
        L, alpha = self._get_length_and_angle(nodes)
        M_local = section.get_local_mass(material, L)
        T = self._rotation_matrix(alpha)
        return T.T @ M_local @ T

    # ------------------------------------------------------------------
    # Internal forces
    # ------------------------------------------------------------------

    def get_internal_forces(
        self,
        nodes: list,
        section: Section,
        material: Material,
    ) -> ndarray:
        """
        Return the 6-component vector of internal nodal forces
        in global coordinates, computed from the current trial state.

        F_int = K_global * u_element

        where u_element is extracted from the trial displacements stored
        in the element state (updated by update()).
        """
        L, alpha = self._get_length_and_angle(nodes)
        EA, EI = section.get_stiffness(material)
        K_local = self._local_stiffness(EA, EI, L)
        T = self._rotation_matrix(alpha)

        # local displacements from trial state
        # state stores [eps, kappa] per Gauss point — for internal forces
        # we reconstruct from the stored nodal displacements in trial
        # (stored by update())
        u_local = self._trial_local_displacements(nodes, T)
        f_local = K_local @ u_local
        return T.T @ f_local

    def get_section_forces(
        self,
        xi: float,
        nodes: list,
        section: Section,
        material: Material,
    ) -> SectionForces:
        """
        Return internal forces at xi in [0, 1] along the element.

        For a linear elastic Euler-Bernoulli beam:
          N(xi) = const  (no distributed axial load)
          V(xi) = const  (no distributed transverse load)
          M(xi) = linear interpolation between M_i and M_j
        """
        L, alpha = self._get_length_and_angle(nodes)
        EA, EI = section.get_stiffness(material)
        K_local = self._local_stiffness(EA, EI, L)
        T = self._rotation_matrix(alpha)

        u_local = self._trial_local_displacements(nodes, T)
        f_local = K_local @ u_local  # [N_i, V_i, M_i, N_j, V_j, M_j]

        # sign convention: f_local[0..2] are forces on node i end (local)
        N = f_local[3]           # axial at node j (equilibrium)
        V = f_local[1]           # shear at node i
        M = (1 - xi) * (-f_local[2]) + xi * f_local[5]  # linear M diagram

        return SectionForces(N=N, V=V, M=M, xi=xi)

    def get_equivalent_forces(
        self,
        nodes: list,
        section: Section,
        material: Material,
        imposed_strain,
    ) -> ndarray:
        """
        Equivalent nodal forces for imposed strains (thermal, prestress).

        F_eq = -T^T * K_local * eps_0_nodal

        Not used in Phase 0 (no imposed strains). Raises NotImplementedError
        until a concrete imposed_strain type is defined.
        """
        raise NotImplementedError(
            "Equivalent forces for imposed strains are not yet implemented "
            "in Phase 0. Will be added with thermal load support."
        )

    # ------------------------------------------------------------------
    # State update
    # ------------------------------------------------------------------

    def update(
        self,
        nodes: list,
        section: Section,
        material: Material,
        U_global: ndarray,
    ) -> None:
        """
        Update trial state (eps, kappa) at each Gauss point.

        Extracts local displacements from U_global, computes strains
        at Gauss points using standard B matrix, stores in section states.
        """
        L, alpha = self._get_length_and_angle(nodes)
        T = self._rotation_matrix(alpha)
        u_local = self._extract_local_displacements(nodes, T, U_global)

        # store nodal local displacements for later use by internal forces
        self._u_local_trial = u_local

        # update strain at each Gauss point
        xi_gauss, _ = self._gauss_points(self.n_gauss)
        for k, xi in enumerate(xi_gauss):
            eps, kappa = self._strain_at(xi, u_local, L)
            self.state.gauss_states[k].eps_trial = eps
            self.state.gauss_states[k].kappa_trial = kappa

            # update material state through section
            # for ElasticSection: single material point
            # for FiberSection (Phase 1): one point per fiber
            sig, _ = material.compute(eps, self.state.gauss_states[k])
            self.state.gauss_states[k].eps_trial = eps
            self.state.gauss_states[k].kappa_trial = kappa

    def commit(self) -> None:
        """Commit trial state and cached local displacements."""
        self.state.commit()
        self._u_local_committed = getattr(self, "_u_local_trial", np.zeros(6))

    def revert(self) -> None:
        """Revert to committed state."""
        self.state.revert()
        self._u_local_trial = getattr(self, "_u_local_committed", np.zeros(6))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _trial_local_displacements(self, nodes: list, T: ndarray) -> ndarray:
        """Return cached local trial displacements (set by update())."""
        return getattr(self, "_u_local_trial", np.zeros(6))

    def _extract_local_displacements(
        self,
        nodes: list,
        T: ndarray,
        U_global: ndarray,
    ) -> ndarray:
        """
        Extract the 6-component local displacement vector from U_global.

        Requires node.dofs to be populated by DOFManager.finalize().
        """
        ni, nj = nodes
        dofs = ni.dofs + nj.dofs   # [d0,d1,d2, d3,d4,d5]
        u_global_elem = U_global[dofs]
        return T @ u_global_elem

    def _strain_at(
        self, xi: float, u_local: ndarray, L: float
    ) -> tuple[float, float]:
        """
        Compute axial strain eps and curvature kappa at xi in [0, 1].

        eps   = (u_j - u_i) / L          (constant along element)
        kappa = B_bending(xi) @ u_bending (linear for EB beam)
        """
        u_i, v_i, th_i, u_j, v_j, th_j = u_local

        # axial strain (constant)
        eps = (u_j - u_i) / L

        # curvature from second derivative of cubic Hermite shape functions
        # N1'' = 12*xi/L^2 - 6/L^2
        # N2'' = 6*xi/L - 4/L
        # N3'' = -12*xi/L^2 + 6/L^2
        # N4'' = 6*xi/L - 2/L
        kappa = (
            (12 * xi / L**2 - 6 / L**2) * v_i
            + (6 * xi / L - 4 / L) * th_i
            + (-12 * xi / L**2 + 6 / L**2) * v_j
            + (6 * xi / L - 2 / L) * th_j
        )
        return eps, kappa

    @staticmethod
    def _gauss_points(n: int) -> tuple[list[float], list[float]]:
        """
        Return Gauss point coordinates and weights on [0, 1].

        n=1 : midpoint rule
        n=2 : 2-point Gauss-Legendre
        n=3 : 3-point Gauss-Legendre
        """
        if n == 1:
            return [0.5], [1.0]
        elif n == 2:
            c = 1 / (2 * np.sqrt(3))
            return [0.5 - c, 0.5 + c], [0.5, 0.5]
        elif n == 3:
            c = np.sqrt(3 / 5)
            return (
                [0.5 - c / 2, 0.5, 0.5 + c / 2],
                [5 / 18, 4 / 9, 5 / 18],
            )
        else:
            raise ValueError(f"Unsupported number of Gauss points: {n}. Use 1, 2, or 3.")

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "type": "EulerBernoulliBeam",
            "id": self.id,
            "node_ids": list(self.node_ids),
            "section_id": self.section_id,
            "n_gauss": self.n_gauss,
            "state": self.state.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> EulerBernoulliBeam:
        obj = cls(
            id=data["id"],
            node_ids=tuple(data["node_ids"]),
            section_id=data["section_id"],
            n_gauss=data.get("n_gauss", 2),
        )
        obj.state = ElementState.from_dict(data["state"])
        return obj
