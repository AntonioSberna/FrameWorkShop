"""
load_control.py
===============
LoadControl (ABC), PlainLoadControl, DisplacementControl.

LoadControl governs how the load factor lambda grows between steps and
how it updates between iterations within a step.

Separation of concerns:
  - LoadControl  : knows lambda and how to advance it.
  - Solver       : owns the Newton-Raphson loop.
  - ConvergenceCriterion : decides when an iteration has converged.

These three objects are completely independent and interchangeable.

PlainLoadControl:
  Lambda grows by a fixed delta_lambda each step.
  No constraint equation — straightforward proportional loading.
  Suitable for linear analysis (Phase 0) and nonlinear problems
  that do not exhibit limit points or snap-through.

DisplacementControl:
  Controls the displacement at a specific DOF (node + direction).
  Lambda is back-calculated so that the controlled displacement
  advances by delta_u each step.
  Suitable for softening response past a peak load.
  (Predisposed for Phase 2 — constraint_equation() is implemented
   but the solver does not use it yet in Phase 0.)

ArcLengthControl:
  Deferred to Phase 2. Interface predisposed via constraint_equation().
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy import ndarray

from convergence import IterationData
from model import DOFManager


# ---------------------------------------------------------------------------
# LoadControl (ABC)
# ---------------------------------------------------------------------------

class LoadControl(ABC):
    """
    Abstract interface for load stepping strategies.

    The solver calls these methods in this order each step:

        control.initialize(dof_mgr)          # once per stage
        lambda_ = control.get_lambda(step)   # at start of each step
        ...Newton-Raphson loop...
        control.update(data)                 # after convergence

    constraint_equation() returns None for simple load control (no
    extra equation added to the system). Arc-length returns a float.
    """

    @abstractmethod
    def initialize(self, dof_mgr: DOFManager) -> None:
        """
        Called once per analysis stage, after model.finalize().

        DOF numbering may change between stages (different topology),
        so any DOF-dependent setup must happen here, not in __init__.
        """
        ...

    @abstractmethod
    def get_lambda(self, step: int) -> float:
        """
        Return the load factor lambda for the given step.

        For PlainLoadControl: lambda = step * delta_lambda.
        For DisplacementControl: lambda is back-calculated.

        Parameters
        ----------
        step : 1-based step counter
        """
        ...

    @abstractmethod
    def update(self, data: IterationData) -> None:
        """
        Update internal state after a converged step.

        Called by the solver after commit(). Used by DisplacementControl
        to track the accumulated controlled displacement.
        """
        ...

    def constraint_equation(self, data: IterationData) -> float | None:
        """
        Return the arc-length constraint equation value, or None.

        None   -> no extra constraint (PlainLoadControl, DisplacementControl
                  in its simple form).
        float  -> value of the constraint g(U, lambda) = 0 that the
                  arc-length method must satisfy (Phase 2).

        Default implementation returns None — subclasses override as needed.
        """
        return None

    @abstractmethod
    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dictionary."""
        ...

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict) -> LoadControl:
        """Reconstruct instance from dictionary."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# ---------------------------------------------------------------------------
# PlainLoadControl
# ---------------------------------------------------------------------------

class PlainLoadControl(LoadControl):
    """
    Proportional load stepping with a fixed lambda increment.

    lambda(step) = step * delta_lambda

    This is the default control strategy for linear analysis (Phase 0).
    For a linear problem, a single step with delta_lambda=1.0 applies
    the full load and the solver converges in one iteration.

    For nonlinear problems (Phase 1+), multiple steps allow tracking
    the load-displacement response incrementally.

    Parameters
    ----------
    delta_lambda : load factor increment per step (default 1.0)
    n_steps      : total number of steps (informational — the solver
                   decides when to stop, not the control object)

    Example
    -------
    # Linear analysis: single step, full load
    control = PlainLoadControl(delta_lambda=1.0)

    # Nonlinear: 10 equal load steps
    control = PlainLoadControl(delta_lambda=0.1, n_steps=10)
    """

    def __init__(
        self,
        delta_lambda: float = 1.0,
        n_steps: int = 1,
    ) -> None:
        if delta_lambda <= 0:
            raise ValueError(
                f"PlainLoadControl: delta_lambda must be positive "
                f"(got {delta_lambda})."
            )
        if n_steps < 1:
            raise ValueError(
                f"PlainLoadControl: n_steps must be >= 1 (got {n_steps})."
            )
        self.delta_lambda = delta_lambda
        self.n_steps = n_steps
        self._current_lambda: float = 0.0

    def initialize(self, dof_mgr: DOFManager) -> None:
        """Reset lambda to zero. Called at the start of each stage."""
        self._current_lambda = 0.0

    def get_lambda(self, step: int) -> float:
        """
        Return lambda for the given step (1-based).

        lambda = step * delta_lambda
        """
        self._current_lambda = step * self.delta_lambda
        return self._current_lambda

    def update(self, data: IterationData) -> None:
        """No internal state to update for plain load control."""
        pass

    def to_dict(self) -> dict:
        return {
            "type": "PlainLoadControl",
            "delta_lambda": self.delta_lambda,
            "n_steps": self.n_steps,
        }

    @classmethod
    def from_dict(cls, data: dict) -> PlainLoadControl:
        return cls(
            delta_lambda=data.get("delta_lambda", 1.0),
            n_steps=data.get("n_steps", 1),
        )

    def __repr__(self) -> str:
        return (
            f"PlainLoadControl(delta_lambda={self.delta_lambda}, "
            f"n_steps={self.n_steps})"
        )


# ---------------------------------------------------------------------------
# DisplacementControl
# ---------------------------------------------------------------------------

class DisplacementControl(LoadControl):
    """
    Controls the displacement at a specific node DOF.

    Instead of prescribing lambda directly, the solver drives the
    displacement of a chosen DOF (the "controlled DOF") by delta_u
    each step, and back-calculates the corresponding lambda.

    This allows tracing the response past a load peak (softening),
    where PlainLoadControl would diverge.

    Parameters
    ----------
    node_id   : ID of the controlled node
    direction : "ux" | "uy" | "rz"
    delta_u   : displacement increment per step
    n_steps   : total number of steps

    Usage
    -----
    After initialize(), the controlled global DOF index is available
    as self.controlled_dof.

    Phase 0 note:
    The constraint equation is predisposed but the solver does not
    use it yet. In Phase 0 this behaves like PlainLoadControl with
    lambda = step * delta_lambda, where delta_lambda is computed from
    the stiffness at the controlled DOF.

    Phase 2 note:
    The full back-calculation of lambda requires the tangent stiffness
    matrix, which will be passed by the NonlinearStaticSolver.
    """

    _DOF_NAME_TO_LOCAL = {"ux": 0, "uy": 1, "rz": 2}

    def __init__(
        self,
        node_id: str | int,
        direction: str,
        delta_u: float,
        n_steps: int = 10,
    ) -> None:
        if direction not in self._DOF_NAME_TO_LOCAL:
            raise ValueError(
                f"DisplacementControl: invalid direction '{direction}'. "
                f"Must be 'ux', 'uy', or 'rz'."
            )
        if n_steps < 1:
            raise ValueError(
                f"DisplacementControl: n_steps must be >= 1 (got {n_steps})."
            )
        self.node_id = node_id
        self.direction = direction
        self.delta_u = delta_u
        self.n_steps = n_steps

        self.controlled_dof: int | None = None   # set by initialize()
        self._accumulated_u: float = 0.0
        self._current_lambda: float = 0.0

    def initialize(self, dof_mgr: DOFManager) -> None:
        """
        Resolve the controlled node + direction to a global DOF index.

        Must be called after model.finalize() so that DOF indices are
        available. Called again at the start of each stage.
        """
        node_dofs = dof_mgr.get_node_dofs(self.node_id)
        local_idx = self._DOF_NAME_TO_LOCAL[self.direction]
        self.controlled_dof = node_dofs[local_idx]
        self._accumulated_u = 0.0
        self._current_lambda = 0.0

    def get_lambda(self, step: int) -> float:
        """
        Return the current lambda.

        In Phase 0 this returns step * (delta_u / reference_stiffness),
        which is a placeholder. The full back-calculation happens in
        NonlinearStaticSolver (Phase 2).

        For now, returns the lambda stored by the last update() call,
        or step * 0.1 as a safe default before the first update.
        """
        # Phase 0 placeholder: lambda grows linearly with step
        # Full implementation in Phase 2 (NonlinearStaticSolver)
        return self._current_lambda if step > 1 else 0.0

    def update(self, data: IterationData) -> None:
        """
        Update accumulated displacement and back-calculate lambda.

        Called after each converged step. In Phase 2, lambda will be
        back-calculated from K_tangent and the controlled DOF response.

        TODO (Phase 2): implement proper lambda back-calculation:
            lambda = (K_tangent[free, controlled] * delta_u) / F_ref[controlled]
        """
        if self.controlled_dof is not None:
            self._accumulated_u += self.delta_u
        # lambda update deferred to Phase 2
        self._current_lambda = data.load_factor

    def constraint_equation(self, data: IterationData) -> float | None:
        """
        Displacement constraint: u_controlled - target = 0.

        Returns the violation of the constraint at the current trial state.
        Used by the solver to enforce the controlled displacement exactly.

        g = U_trial[controlled_dof] - (step * delta_u) = 0

        Returns None if controlled_dof is not yet initialized.

        TODO (Phase 2): this will drive the lambda back-calculation in
        the augmented system [K, dF/dlambda; g_U, g_lambda] * [dU, dlambda].
        """
        if self.controlled_dof is None:
            return None
        u_current = data.U_trial[self.controlled_dof]
        target = self._accumulated_u + self.delta_u
        return float(u_current - target)

    def to_dict(self) -> dict:
        return {
            "type": "DisplacementControl",
            "node_id": self.node_id,
            "direction": self.direction,
            "delta_u": self.delta_u,
            "n_steps": self.n_steps,
        }

    @classmethod
    def from_dict(cls, data: dict) -> DisplacementControl:
        return cls(
            node_id=data["node_id"],
            direction=data["direction"],
            delta_u=data["delta_u"],
            n_steps=data.get("n_steps", 10),
        )

    def __repr__(self) -> str:
        return (
            f"DisplacementControl(node_id={self.node_id!r}, "
            f"direction={self.direction!r}, delta_u={self.delta_u})"
        )
