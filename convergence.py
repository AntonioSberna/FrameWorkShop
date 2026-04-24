"""
convergence.py
==============
ConvergenceCriterion (ABC), ForceResidual, DisplacementIncrement,
EnergyConvergence.

Two separate convergence levels exist in the framework:
  - Global convergence  : checks whether the Newton-Raphson iteration
                          has converged at the structural level.
  - Section convergence : checks state determination at the fiber level
                          (Phase 1, FiberSection).

Both levels use the same interface — the solver instantiates them
separately and calls is_converged() independently.

IterationData carries all quantities needed by any criterion, so that
switching criterion never requires changes to the solver.

Norms supported: "L2" (Euclidean) and "Linf" (maximum absolute value).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from numpy import ndarray


# ---------------------------------------------------------------------------
# IterationData
# ---------------------------------------------------------------------------

@dataclass
class IterationData:
    """
    Snapshot of a single Newton-Raphson iteration.

    Passed by the solver to ConvergenceCriterion.is_converged().
    All fields needed by any criterion are included here so that
    switching criterion never requires changes to the solver loop.

    Fields
    ------
    iteration       : iteration counter (0-based)
    residual        : out-of-balance force vector R = F_ext - F_int
    delta_U         : displacement increment from this iteration
    U_trial         : total trial displacement vector
    load_factor     : current lambda (for nonlinear load control)
    F_external      : scaled external force vector (lambda * F_ref)
    internal_energy : 0.5 * delta_U^T * residual  (energy criterion)

    Notes
    -----
    For linear analysis (Phase 0) the system converges in one iteration.
    residual will be near machine epsilon after the first solve.
    delta_U and U_trial are identical in the linear case.
    """

    iteration: int = 0
    residual: ndarray = field(default_factory=lambda: np.zeros(1))
    delta_U: ndarray = field(default_factory=lambda: np.zeros(1))
    U_trial: ndarray = field(default_factory=lambda: np.zeros(1))
    load_factor: float = 1.0
    F_external: ndarray = field(default_factory=lambda: np.zeros(1))

    @property
    def internal_energy(self) -> float:
        """
        Incremental internal energy: delta_U^T * residual.

        Dimensionally consistent convergence measure — combines both
        force and displacement errors in a single scalar.
        Used by EnergyConvergence.
        """
        return float(np.dot(self.delta_U, self.residual))

    @property
    def reference_energy(self) -> float:
        """
        Reference energy for normalization: delta_U_0^T * F_external.

        In Newton-Raphson this is computed at the first iteration (iter=0)
        and cached by the solver. Here it is approximated as
        delta_U^T * F_external which is exact at iter=0.
        """
        return float(np.dot(self.delta_U, self.F_external))


# ---------------------------------------------------------------------------
# ConvergenceCriterion (ABC)
# ---------------------------------------------------------------------------

class ConvergenceCriterion(ABC):
    """
    Abstract interface for convergence checks.

    Each criterion:
      - Stores a tolerance and a norm type.
      - Implements is_converged(data) -> bool.
      - Implements description() -> str for logging.
      - Implements to_dict() / from_dict() for serialization.

    Two instances are used by the solver:
      solver.convergence         -> global Newton-Raphson convergence
      solver.section_convergence -> fiber state determination (Phase 1)
    """

    def __init__(self, tol: float, norm: str = "L2") -> None:
        if tol <= 0:
            raise ValueError(
                f"{self.__class__.__name__}: tolerance must be positive "
                f"(got tol={tol})."
            )
        if norm not in ("L2", "Linf"):
            raise ValueError(
                f"{self.__class__.__name__}: norm must be 'L2' or 'Linf' "
                f"(got norm={norm!r})."
            )
        self.tol = tol
        self.norm = norm

    def _compute_norm(self, v: ndarray) -> float:
        """Compute the selected vector norm."""
        if self.norm == "L2":
            return float(np.linalg.norm(v))
        else:   # Linf
            return float(np.max(np.abs(v)))

    @abstractmethod
    def is_converged(self, data: IterationData) -> bool:
        """Return True if the convergence criterion is satisfied."""
        ...

    @abstractmethod
    def description(self, data: IterationData) -> str:
        """
        Return a human-readable string describing the current state.
        Used by the solver for logging and educational output.
        """
        ...

    @abstractmethod
    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dictionary."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(tol={self.tol}, norm={self.norm!r})"


# ---------------------------------------------------------------------------
# ForceResidual
# ---------------------------------------------------------------------------

class ForceResidual(ConvergenceCriterion):
    """
    Convergence based on the out-of-balance force norm.

    Converged when:
        ||R|| / ||F_ext|| < tol

    where R = F_external - F_internal is the residual and
    ||F_ext|| is the norm of the external force vector (reference).

    If ||F_ext|| is near zero (unloaded structure), the absolute norm
    ||R|| is used instead to avoid division by zero.

    This is the most common criterion in structural FEM and the
    default choice for Phase 0 (linear elastic).

    For linear problems the residual is zero after the first iteration
    (to machine precision), so convergence is always achieved at iter=0.
    """

    def __init__(self, tol: float = 1e-6, norm: str = "L2") -> None:
        super().__init__(tol=tol, norm=norm)

    def is_converged(self, data: IterationData) -> bool:
        r_norm = self._compute_norm(data.residual)
        f_norm = self._compute_norm(data.F_external)

        if f_norm < 1e-14:
            # unloaded or near-zero load: use absolute norm
            return r_norm < self.tol

        return (r_norm / f_norm) < self.tol

    def description(self, data: IterationData) -> str:
        r_norm = self._compute_norm(data.residual)
        f_norm = self._compute_norm(data.F_external)
        ratio = r_norm / f_norm if f_norm > 1e-14 else r_norm
        return (
            f"ForceResidual [{self.norm}] "
            f"iter={data.iteration}  "
            f"||R||={r_norm:.3e}  "
            f"||F||={f_norm:.3e}  "
            f"ratio={ratio:.3e}  "
            f"tol={self.tol:.3e}  "
            f"{'CONVERGED' if ratio < self.tol else 'not converged'}"
        )

    def to_dict(self) -> dict:
        return {
            "type": "ForceResidual",
            "tol": self.tol,
            "norm": self.norm,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ForceResidual:
        return cls(tol=data["tol"], norm=data.get("norm", "L2"))


# ---------------------------------------------------------------------------
# DisplacementIncrement
# ---------------------------------------------------------------------------

class DisplacementIncrement(ConvergenceCriterion):
    """
    Convergence based on the displacement increment norm.

    Converged when:
        ||delta_U|| / ||U_trial|| < tol

    where delta_U is the displacement increment of this iteration and
    U_trial is the total trial displacement vector.

    If ||U_trial|| is near zero (first iteration from rest), the
    absolute norm ||delta_U|| is used instead.

    This criterion is useful for nonlinear problems where force residuals
    can be misleading (e.g. near limit points). It will be the primary
    criterion for ArcLengthControl (Phase 2).

    For linear problems this is equivalent to ForceResidual since
    delta_U = U_trial at the first (and only) iteration.
    """

    def __init__(self, tol: float = 1e-6, norm: str = "L2") -> None:
        super().__init__(tol=tol, norm=norm)

    def is_converged(self, data: IterationData) -> bool:
        du_norm = self._compute_norm(data.delta_U)
        u_norm = self._compute_norm(data.U_trial)

        if u_norm < 1e-14:
            return du_norm < self.tol

        return (du_norm / u_norm) < self.tol

    def description(self, data: IterationData) -> str:
        du_norm = self._compute_norm(data.delta_U)
        u_norm = self._compute_norm(data.U_trial)
        ratio = du_norm / u_norm if u_norm > 1e-14 else du_norm
        return (
            f"DisplacementIncrement [{self.norm}] "
            f"iter={data.iteration}  "
            f"||dU||={du_norm:.3e}  "
            f"||U||={u_norm:.3e}  "
            f"ratio={ratio:.3e}  "
            f"tol={self.tol:.3e}  "
            f"{'CONVERGED' if ratio < self.tol else 'not converged'}"
        )

    def to_dict(self) -> dict:
        return {
            "type": "DisplacementIncrement",
            "tol": self.tol,
            "norm": self.norm,
        }

    @classmethod
    def from_dict(cls, data: dict) -> DisplacementIncrement:
        return cls(tol=data["tol"], norm=data.get("norm", "L2"))


# ---------------------------------------------------------------------------
# EnergyConvergence
# ---------------------------------------------------------------------------

class EnergyConvergence(ConvergenceCriterion):
    """
    Convergence based on incremental internal energy.

    Converged when:
        |delta_U^T * R| / |delta_U_0^T * F_ext| < tol

    where delta_U_0 is the displacement increment at the first iteration.

    This criterion is dimensionally consistent (units of energy = force *
    length) and combines both force and displacement errors in a single
    scalar. It is particularly effective near limit points where either
    force or displacement alone can give misleading convergence signals.

    The reference energy (denominator) must be set externally by the solver
    at the first iteration via set_reference(). Before set_reference() is
    called, is_converged() returns False.

    Pedagogical note: this criterion is sensitive to the choice of units —
    mixing N with m vs kN with mm gives very different energy values, even
    though the physics is identical. This is a good teaching moment about
    unit consistency in FEM.
    """

    def __init__(self, tol: float = 1e-8, norm: str = "L2") -> None:
        # norm is inherited but not used for scalars — kept for interface
        # uniformity. The energy is always a scalar absolute value.
        super().__init__(tol=tol, norm=norm)
        self._reference_energy: float | None = None

    def set_reference(self, data: IterationData) -> None:
        """
        Set the reference energy from the first iteration.

        Must be called by the solver at iteration 0 before is_converged()
        is meaningful.
        """
        ref = abs(data.internal_energy)
        self._reference_energy = ref if ref > 1e-14 else None

    def reset(self) -> None:
        """Reset reference energy. Called by solver at each new step."""
        self._reference_energy = None

    def is_converged(self, data: IterationData) -> bool:
        if self._reference_energy is None:
            # reference not yet set — cannot be converged
            return False

        energy = abs(data.internal_energy)
        return (energy / self._reference_energy) < self.tol

    def description(self, data: IterationData) -> str:
        energy = abs(data.internal_energy)
        ref = self._reference_energy or float("nan")
        ratio = energy / ref if self._reference_energy else float("nan")
        return (
            f"EnergyConvergence "
            f"iter={data.iteration}  "
            f"|dU·R|={energy:.3e}  "
            f"ref={ref:.3e}  "
            f"ratio={ratio:.3e}  "
            f"tol={self.tol:.3e}  "
            f"{'CONVERGED' if (self._reference_energy and ratio < self.tol) else 'not converged'}"
        )

    def to_dict(self) -> dict:
        return {
            "type": "EnergyConvergence",
            "tol": self.tol,
            "norm": self.norm,
        }

    @classmethod
    def from_dict(cls, data: dict) -> EnergyConvergence:
        return cls(tol=data["tol"], norm=data.get("norm", "L2"))
