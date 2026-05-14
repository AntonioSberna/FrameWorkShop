"""
fem_material.py
===============
MaterialState, Material (ABC), ElasticMaterial.

Two distinct time levels:
  - committed : converged state at the end of the last step
  - trial     : provisional state of the current iteration

commit() -> trial becomes committed  (called by solver after convergence)
revert() -> trial reverts to committed  (called by solver after divergence)

All attributes are readable (educational purpose).
Writing only through update() / commit() / revert().
"""
# add new comment

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# MaterialState
# ---------------------------------------------------------------------------

@dataclass
class MaterialState:
    """
    Internal state of a material point.

    *_committed attributes : converged values (start of current step).
    *_trial attributes     : provisional values (current iteration).

    For ElasticMaterial the plastic fields remain zero — they are present
    for interface uniformity with EPMaterial (Phase 1).
    """

    # --- committed ---
    eps_committed: float = 0.0    # converged strain
    sig_committed: float = 0.0    # converged stress
    eps_p_committed: float = 0.0  # converged plastic strain
    alpha_committed: float = 0.0  # converged hardening variable

    # --- trial ---
    eps_trial: float = 0.0        # trial strain
    sig_trial: float = 0.0        # trial stress
    eps_p_trial: float = 0.0      # trial plastic strain
    alpha_trial: float = 0.0      # trial hardening variable

    def commit(self) -> None:
        """Trial -> committed. Called after step convergence."""
        self.eps_committed = self.eps_trial
        self.sig_committed = self.sig_trial
        self.eps_p_committed = self.eps_p_trial
        self.alpha_committed = self.alpha_trial

    def revert(self) -> None:
        """Committed -> trial. Called after divergence (undoes the iteration)."""
        # It can be useful if we wanna implement a dynamic steps
        self.eps_trial = self.eps_committed
        self.sig_trial = self.sig_committed
        self.eps_p_trial = self.eps_p_committed
        self.alpha_trial = self.alpha_committed

    def to_dict(self) -> dict:
        return {
            "eps_committed": self.eps_committed,
            "sig_committed": self.sig_committed,
            "eps_p_committed": self.eps_p_committed,
            "alpha_committed": self.alpha_committed,
            "eps_trial": self.eps_trial,
            "sig_trial": self.sig_trial,
            "eps_p_trial": self.eps_p_trial,
            "alpha_trial": self.alpha_trial,
        }

    @classmethod
    def from_dict(cls, data: dict) -> MaterialState:
        return cls(**data)


# ---------------------------------------------------------------------------
# Material (ABC)
# ---------------------------------------------------------------------------

class Material(ABC):
    """
    Abstract interface for all materials.

    Each material must:
      - implement compute(), which receives the trial strain and the current
        state, and returns (sigma, E_tangent) without modifying the state.
        The state is updated externally (by SectionState / ElementState).
      - implement to_dict() / from_dict() for serialization without pickle.

    The `rho` field (density) is used by Section to compute the mass matrix for elements.
    The `unit` field is optional and never read internally by the framework.
    """

    def __init__(
        self,
        id: str | int,
        rho: float = 0.0,
        unit: str | None = None,
    ) -> None:
        self.id = id
        self.rho = rho      # density — used by Section.get_local_mass()
        self.unit = unit    # e.g. "kN/cm²" — for post-processing only

    @abstractmethod
    def compute(
        self,
        eps: float,
        state: MaterialState,
    ) -> tuple[float, float]:
        """
        Compute stress and tangent modulus given trial strain.

        Parameters
        ----------
        eps   : trial strain (total, not incremental)
        state : current (committed) state — not modified by this method

        Returns
        -------
        (sigma, E_tan) : trial stress and tangent modulus
        """
        ...

    @abstractmethod
    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dictionary."""
        ...

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict) -> Material:
        """Reconstruct instance from dictionary."""
        ...

    def run_strain_history(
        self,
        eps_history: list[float],
    ) -> tuple[list[float], list[float]]:
        """
        Drive the material through a strain history and return (sigmas, E_tans).
        Each step is committed — state evolves correctly for path-dependent materials.
        """
        state = MaterialState()
        sigmas, E_tans = [], []
        for eps in eps_history:
            sigma, E_tan = self.compute(eps, state)
            state.eps_trial = eps
            state.sig_trial = sigma
            state.commit()
            sigmas.append(sigma)
            E_tans.append(E_tan)
        return sigmas, E_tans

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.id!r})"


# ---------------------------------------------------------------------------
# ElasticMaterial
# ---------------------------------------------------------------------------

class ElasticMaterial(Material):
    """
    Linear elastic material. E and rho are constant.

    compute() is O(1) — does not depend on state.

    Example
    -------
    >>> mat = ElasticMaterial(e.g., id=1, E=210_000, rho=7.85e-3)
    >>> state = MaterialState()
    >>> sigma, E_tan = mat.compute(eps=0.001, state=state)
    >>> sigma
    >>> E_tan
    """

    def __init__(
        self,
        id: str | int,
        E: float,
        rho: float = 0.0,
        unit: str | None = None,
    ) -> None:
        super().__init__(id=id, rho=rho, unit=unit)
        if E <= 0:
            raise ValueError(
                f"Material '{id}': elastic modulus E must be positive "
                f"(got E={E})."
            )
        self.E = E

    def compute(
        self,
        eps: float,
        state: MaterialState,
    ) -> tuple[float, float]:
        """
        sigma = E * eps  (Hooke's law).
        E_tan = E  (constant — linear material).
        State is neither read nor modified.
        """
        sigma = self.E * eps
        return sigma, self.E

    def to_dict(self) -> dict:
        return {
            "type": "ElasticMaterial",
            "id": self.id,
            "E": self.E,
            "rho": self.rho,
            "unit": self.unit,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ElasticMaterial:
        return cls(
            id=data["id"],
            E=data["E"],
            rho=data.get("rho", 0.0),
            unit=data.get("unit"),
        )

    def __repr__(self) -> str:
        return f"ElasticMaterial(id={self.id!r}, E={self.E})"
