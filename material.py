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

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


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
    hstv_committed: dict = field(default_factory = dict) # converged history variables


    # --- trial ---
    eps_trial: float = 0.0        # trial strain
    sig_trial: float = 0.0        # trial stress
    hstv_trial: dict = field(default_factory = dict)    # trial history variables


    def commit(self) -> None:
        """Trial -> committed. Called after step convergence."""
        self.eps_committed = self.eps_trial
        self.sig_committed = self.sig_trial
        self.hstv_committed = self.hstv_trial.copy()

    def revert(self) -> None:
        """Committed -> trial. Called after divergence (undoes the iteration)."""
        # It can be useful if we wanna implement a dynamic steps
        self.eps_trial = self.eps_committed
        self.sig_trial = self.sig_committed
        self.hstv_trial = self.hstv_committed.copy()


    def to_dict(self) -> dict:
        return {
            "eps_committed": self.eps_committed,
            "sig_committed": self.sig_committed,
            "hstv_committed": self.hstv_committed,
            "eps_trial": self.eps_trial,
            "sig_trial": self.sig_trial,
            "hstv_trial": self.hstv_trial,
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
    

# ---------------------------------------------------------------------------
# bilinear material with isotropic hardening
# ---------------------------------------------------------------------------
class steel01(Material):

    def __init__(
            self,
            id: str | int,
            fy: float,                   # yield stress
            E0: float,                   # initial stiffness
            b: float,                    # hardening ratio (Eh/E0)
            a1: float = 0.07,            # coefficient for isotropic hardening
            a2: float = 2,               # coefficient for isotropic hardening
            a3: float = 0.07,            # coefficient for isotropic hardening
            a4: float = 2,               # coefficient for isotropic hardening
            rho: float = 0.0,
            unit: str | None = None
    ) -> None:
        
        super().__init__(id=id, rho=rho, unit=unit)
        self.fy = fy
        self.E0 = E0
        self.b  = b
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.a4 = a4
        # -------------------------------------------------------------------
        # calculate fixed material properties
        # -------------------------------------------------------------------
        self.Eh = b*E0
        self.epsy = fy/E0

    
    def compute(
            self,
            eps: float,
            state: MaterialState
    ) -> tuple[float,float]:
        
        # -------------------------------------------------------------------
        # retrieve history variables
        # -------------------------------------------------------------------
        epsP = state.eps_committed                      # strain at previous converged step
        sigP = state.sig_committed                      # stress at previous converged step
        epsmin = state.hstv_committed.get("epsmin",0.0) # max eps in compression
        epsmax = state.hstv_committed.get("epsmax",0.0) # max eps in tension
        # -------------------------------------------------------------------
        # calculate current strain increment
        # -------------------------------------------------------------------
        deps = eps - epsP
        # -------------------------------------------------------------------
        # isotropic hardening
        # -------------------------------------------------------------------
        hc = max(
            self.fy * self.a1 * (epsmax/self.epsy - self.a2),
            0.0 )
        ht = max(
            self.fy * self.a3 * (abs(epsmin/self.epsy) - self.a4),
            0.0 )
        # -------------------------------------------------------------------
        # bilinear model
        # -------------------------------------------------------------------
        c1 = self.Eh * eps
        c2 = (self.fy + hc)*(1 - self.b)
        c3 = (self.fy + ht)*(1 - self.b)

        c = sigP + self.E0*deps

        sig = max(
            c1 - c2,
            min( (c1+c3), c ) )
        
        Et = self.Eh
        if abs(sig - c) < 1e-10:
            Et = self.E0

        # -------------------------------------------------------------------
        # update history variables
        # -------------------------------------------------------------------
        epsmin = min(eps, epsmin)
        epsmax = max(eps, epsmax)
        # -------------------------------------------------------------------
        # update trial state
        # -------------------------------------------------------------------
        state.eps_trial = eps
        state.sig_trial = sig
        state.hstv_trial = {
            "epsmin": epsmin,
            "epsmax": epsmax }
        
        return sig, Et
    

    def to_dict(self) -> dict:
        return {
            "type": "steel01",
            "id": self.id,
            "fy": self.fy,
            "E0": self.E0,
            "b": self.b,
            "a1": self.a1,
            "a2": self.a2,
            "a3": self.a3,
            "a4": self.a4,
            "rho": self.rho,
            "unit": self.unit,
        }

    @classmethod
    def from_dict(cls, data: dict) -> steel01:
        return cls(
            id=data["id"],
            fy=data["fy"],
            E0=data["E0"],
            b=data["b"],
            a1=data.get("a1", 0.07),
            a2=data.get("a2", 2.0),
            a3=data.get("a3", 0.07),
            a4=data.get("a4", 2.0),
            rho=data.get("rho", 0.0),
            unit=data.get("unit"),
        )
    
    def __repr__(self) -> str:
        return f"steel01 (id={self.id!r}, fy={self.fy}, E0={self.E0}, b={self.b}, a1={self.a1}, a2={self.a2}, a3={self.a3}, a4={self.a4})"
