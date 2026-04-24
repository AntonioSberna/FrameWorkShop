"""
section.py
==========
SectionState, Section (ABC), ElasticSection.

A section aggregates material properties into cross-section quantities
(axial stiffness EA, bending stiffness EI) used by the element.

Sections are NOT registered in the Model — they are retrieved on demand
by iterating elements (collect on demand pattern).

The number of Gauss points is a property of the Element, not the Section.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy import ndarray

from material import Material, MaterialState


# ---------------------------------------------------------------------------
# SectionState
# ---------------------------------------------------------------------------

class SectionState:
    """
    Internal state of a cross-section at a single integration point.

    For ElasticSection there are no fiber states — the section behaves
    as a single material point with two strain components (axial, curvature).
    For FiberSection (Phase 1) this will hold a list of MaterialState objects,
    one per fiber.

    Two strain components:
      eps  : axial strain  (average over the cross-section)
      kappa: curvature     (bending strain gradient)

    commit() / revert() delegate to fiber states when present.
    """

    def __init__(self) -> None:
        # axial strain — committed and trial
        self.eps_committed: float = 0.0
        self.eps_trial: float = 0.0

        # curvature — committed and trial
        self.kappa_committed: float = 0.0
        self.kappa_trial: float = 0.0

        # fiber states — populated by FiberSection (Phase 1)
        self.fiber_states: list[MaterialState] = []

    def commit(self) -> None:
        """Trial -> committed. Delegates to fiber states if present."""
        self.eps_committed = self.eps_trial
        self.kappa_committed = self.kappa_trial
        for fs in self.fiber_states:
            fs.commit()

    def revert(self) -> None:
        """Committed -> trial. Delegates to fiber states if present."""
        self.eps_trial = self.eps_committed
        self.kappa_trial = self.kappa_committed
        for fs in self.fiber_states:
            fs.revert()

    def to_dict(self) -> dict:
        return {
            "eps_committed": self.eps_committed,
            "eps_trial": self.eps_trial,
            "kappa_committed": self.kappa_committed,
            "kappa_trial": self.kappa_trial,
            "fiber_states": [fs.to_dict() for fs in self.fiber_states],
        }

    @classmethod
    def from_dict(cls, data: dict) -> SectionState:
        from material import MaterialState
        s = cls()
        s.eps_committed = data["eps_committed"]
        s.eps_trial = data["eps_trial"]
        s.kappa_committed = data["kappa_committed"]
        s.kappa_trial = data["kappa_trial"]
        s.fiber_states = [MaterialState.from_dict(d) for d in data["fiber_states"]]
        return s


# ---------------------------------------------------------------------------
# Section (ABC)
# ---------------------------------------------------------------------------

class Section(ABC):
    """
    Abstract interface for all cross-sections.

    A section knows its geometry (A, I) and holds a reference to its
    material by ID — never a direct object reference (principle 1).

    The material object must be passed explicitly to methods that need it
    (get_stiffness, get_local_mass) — the section never fetches it from
    a registry on its own.

    The `unit` field is optional and never read internally by the framework.
    """

    def __init__(
        self,
        id: str | int,
        material_id: str | int,
        unit: str | None = None,
    ) -> None:
        self.id = id
        self.material_id = material_id  # reference by ID, never direct object
        self.unit = unit

    @abstractmethod
    def get_stiffness(self, material: Material) -> tuple[float, float]:
        """
        Return (EA, EI) — axial and bending stiffness.

        Parameters
        ----------
        material : material object resolved externally by the caller

        Returns
        -------
        (EA, EI) : axial stiffness and bending stiffness
        """
        ...

    @abstractmethod
    def get_local_mass(self, material: Material, L: float) -> ndarray:
        """
        Return the consistent local mass matrix (6x6) for a beam element.

        Parameters
        ----------
        material : material object resolved externally by the caller
        L        : element length

        Returns
        -------
        6x6 consistent mass matrix in local coordinates
        """
        ...

    @abstractmethod
    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dictionary."""
        ...

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict) -> Section:
        """Reconstruct instance from dictionary."""
        ...

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(id={self.id!r}, material_id={self.material_id!r})"
        )


# ---------------------------------------------------------------------------
# ElasticSection
# ---------------------------------------------------------------------------

class ElasticSection(Section):
    """
    Prismatic elastic cross-section defined by area A and moment of inertia I.

    EA = E * A  (axial stiffness)
    EI = E * I  (bending stiffness)

    The consistent mass matrix follows the standard Euler-Bernoulli
    formulation: m = rho * A * L * M_consistent, where M_consistent
    is the 6x6 matrix from cubic Hermite shape functions.

    DOF order: [u_i, v_i, theta_i, u_j, v_j, theta_j]

    Example
    -------
    >>> from material import ElasticMaterial
    >>> mat = ElasticMaterial(id="steel", E=210_000, rho=7.85e-3)
    >>> sec = ElasticSection(id="IPE300", material_id="steel", A=53.8, I=8360)
    >>> EA, EI = sec.get_stiffness(mat)
    >>> EA 11298000.0
    >>> EI 1755600000.0
    """

    def __init__(
        self,
        id: str | int,
        material_id: str | int,
        A: float,
        I: float,
        unit: str | None = None,
    ) -> None:
        super().__init__(id=id, material_id=material_id, unit=unit)
        if A <= 0:
            raise ValueError(
                f"Section '{id}': area A must be positive (got A={A})."
            )
        if I <= 0:
            raise ValueError(
                f"Section '{id}': moment of inertia I must be positive "
                f"(got I={I})."
            )
        self.A = A
        self.I = I

    def get_stiffness(self, material: Material) -> tuple[float, float]:
        """Return (EA, EI)."""
        EA = material.E * self.A
        EI = material.E * self.I
        return EA, EI

    def get_local_mass(self, material: Material, L: float) -> ndarray:
        raise NotImplementedError("To be implemented in next phase (dynamic analysis)")


    def to_dict(self) -> dict:
        return {
            "type": "ElasticSection",
            "id": self.id,
            "material_id": self.material_id,
            "A": self.A,
            "I": self.I,
            "unit": self.unit,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ElasticSection:
        return cls(
            id=data["id"],
            material_id=data["material_id"],
            A=data["A"],
            I=data["I"],
            unit=data.get("unit"),
        )

    def __repr__(self) -> str:
        return (
            f"ElasticSection(id={self.id!r}, material_id={self.material_id!r}, "
            f"A={self.A}, I={self.I})"
        )
