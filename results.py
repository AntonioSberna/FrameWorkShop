"""
results.py
==========
StoreOptions, SolverConfig, SolverCallbacks,
ModelState, StepResult, Results.

These objects carry configuration into the solver and carry results out.

StoreOptions  : controls what gets stored (granularity vs memory tradeoff).
SolverConfig  : bundles all solver configuration in one place.
SolverCallbacks : hooks called by the solver at key events.
ModelState    : snapshot of displacements and element states at a step.
StepResult    : one converged step — displacements, reactions, forces.
Results       : collection of all StepResults with hierarchical access.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from numpy import ndarray


# ---------------------------------------------------------------------------
# StoreOptions
# ---------------------------------------------------------------------------

@dataclass
class StoreOptions:
    """
    Controls what the solver stores in Results.

    Fine-grained control allows trading memory for inspectability:
      - For production runs: store only converged step results.
      - For educational/debug: store everything including iterations
        and individual Gauss point states.

    Attributes
    ----------
    iterations   : store IterationData for each N-R iteration
    gauss_points : store SectionState at each Gauss point per step
    fibers       : store MaterialState for each fiber (FiberSection, Phase 1)
    elements     : list of element IDs to store detail for.
                   None -> store all elements.
                   [] -> store no element detail.
    """

    iterations: bool = False
    gauss_points: bool = False
    fibers: bool = False
    elements: list[str | int] | None = None   # None = all

    def stores_element(self, elem_id: str | int) -> bool:
        """Return True if detail should be stored for this element."""
        if self.elements is None:
            return True
        return elem_id in self.elements


# ---------------------------------------------------------------------------
# SolverCallbacks
# ---------------------------------------------------------------------------

@dataclass
class SolverCallbacks:
    """
    User-defined hooks called by the solver at key events.

    All callbacks receive the relevant data object and return nothing.
    Exceptions raised inside callbacks propagate to the caller.

    on_iteration   : called after each N-R iteration
                     signature: (data: IterationData) -> None
    on_step        : called after each converged step
                     signature: (result: StepResult) -> None
    on_convergence : called when a step converges
                     signature: (data: IterationData) -> None
    on_divergence  : called when a step fails to converge
                     signature: (data: IterationData) -> None

    Example
    -------
    >>> def print_step(result):
    ...     print(f"Step {result.step}: lambda={result.load_factor:.3f}")
    >>> callbacks = SolverCallbacks(on_step=print_step)
    """

    on_iteration: Callable | None = None
    on_step: Callable | None = None
    on_convergence: Callable | None = None
    on_divergence: Callable | None = None


# ---------------------------------------------------------------------------
# SolverConfig
# ---------------------------------------------------------------------------

@dataclass
class SolverConfig:
    """
    Full solver configuration bundled in one object.

    Attributes
    ----------
    store             : what to store in Results
    callbacks         : event hooks
    checkpoint        : path to save Results after each step (None = no save)
    restart_from      : path to load a partial Results to continue from
    check_equilibrium : verify global equilibrium after each converged step
    equilibrium_tol   : tolerance for equilibrium check
    max_iter          : maximum Newton-Raphson iterations per step
    linear_solver     : "spsolve" | "minres" | "lgmres"
    verbose           : print convergence info each iteration

    Example
    -------
    >>> config = SolverConfig(
    ...     store=StoreOptions(iterations=True, gauss_points=True),
    ...     check_equilibrium=True,
    ...     max_iter=50,
    ...     verbose=True,
    ... )
    """

    store: StoreOptions = field(default_factory=StoreOptions)
    callbacks: SolverCallbacks = field(default_factory=SolverCallbacks)
    checkpoint: str | None = None
    restart_from: str | None = None
    check_equilibrium: bool = True
    equilibrium_tol: float = 1e-6
    max_iter: int = 50
    linear_solver: str = "spsolve"
    verbose: bool = False


# ---------------------------------------------------------------------------
# ModelState
# ---------------------------------------------------------------------------

class ModelState:
    """
    Snapshot of the structural state at a converged step.

    Used to carry state between staged analysis stages. Indexed by
    node_id and element_id — never by array position — so it remains
    valid across stages with different DOF numbering.

    Attributes
    ----------
    nodal_displacements : node_id -> [ux, uy, rz]
    element_states      : elem_id -> ElementState
    load_factor         : lambda at this state
    step                : step number
    """

    def __init__(self) -> None:
        self.nodal_displacements: dict[str | int, ndarray] = {}
        self.element_states: dict[str | int, Any] = {}
        self.load_factor: float = 0.0
        self.step: int = 0

    @classmethod
    def from_step_result(cls, result: StepResult, dof_mgr) -> ModelState:
        """Build a ModelState from a StepResult and DOFManager."""
        state = cls()
        state.load_factor = result.load_factor
        state.step = result.step
        state.nodal_displacements = dof_mgr.extract_state_from_U(result.U)
        state.element_states = {
            eid: elem_res.element_state
            for eid, elem_res in result.element_results.items()
        }
        return state

    def to_dict(self) -> dict:
        return {
            "load_factor": self.load_factor,
            "step": self.step,
            "nodal_displacements": {
                k: v.tolist()
                for k, v in self.nodal_displacements.items()
            },
            # element_states serialization deferred — complex nested objects
        }

    def __repr__(self) -> str:
        return (
            f"ModelState(step={self.step}, "
            f"lambda={self.load_factor:.4f}, "
            f"nodes={len(self.nodal_displacements)})"
        )


# ---------------------------------------------------------------------------
# NodeResult — view into a single node at a single step
# ---------------------------------------------------------------------------

class NodeResult:
    """
    Displacement components at a single node for a single step.

    Access pattern:
        results["step_1"]["A"].ux
        results["step_1"]["A"].uy
        results["step_1"]["A"].rz
    """

    def __init__(self, node_id: str | int, u: ndarray) -> None:
        self.node_id = node_id
        self._u = u   # [ux, uy, rz]

    @property
    def ux(self) -> float:
        return float(self._u[0])

    @property
    def uy(self) -> float:
        return float(self._u[1])

    @property
    def rz(self) -> float:
        return float(self._u[2])

    @property
    def u(self) -> ndarray:
        return self._u.copy()

    def __repr__(self) -> str:
        return (
            f"NodeResult(id={self.node_id!r}, "
            f"ux={self.ux:.4e}, uy={self.uy:.4e}, rz={self.rz:.4e})"
        )


# ---------------------------------------------------------------------------
# ElementResult — view into a single element at a single step
# ---------------------------------------------------------------------------

class ElementResult:
    """
    Internal forces and state for a single element at a single step.

    Access pattern:
        results["step_1"]["e1"].forces        # global nodal forces
        results["step_1"]["e1"].element_state # ElementState (if stored)
    """

    def __init__(
        self,
        elem_id: str | int,
        forces: ndarray,
        element_state: Any = None,
    ) -> None:
        self.elem_id = elem_id
        self.forces = forces          # (6,) global internal forces
        self.element_state = element_state   # ElementState | None

    def get_forces(self, ref: str = "global") -> ndarray:
        """
        Return internal nodal forces.

        Parameters
        ----------
        ref : "global" (default) | "local"
              "local" requires the rotation matrix — not yet implemented.
              Will be added when Element stores its last rotation matrix.
        """
        if ref == "global":
            return self.forces.copy()
        raise NotImplementedError(
            "ref='local' for ElementResult.get_forces() is not yet "
            "implemented. Will be added in Phase 1."
        )

    def __repr__(self) -> str:
        return f"ElementResult(id={self.elem_id!r})"


# ---------------------------------------------------------------------------
# StepResult
# ---------------------------------------------------------------------------

class StepResult:
    """
    All results for a single converged load step.

    Attributes
    ----------
    step            : step number (1-based)
    load_factor     : lambda at this step
    U               : global displacement vector (n_dofs,)
    reactions       : node_id -> [Rx, Ry, Mrz]
    element_results : elem_id -> ElementResult
    iterations      : list of IterationData (if StoreOptions.iterations=True)

    Access via Results["step_N"]["node_id"] is the intended pattern.
    Direct access to StepResult attributes is also supported.
    """

    def __init__(
        self,
        step: int,
        load_factor: float,
        U: ndarray,
        reactions: dict[str | int, ndarray],
        element_results: dict[str | int, ElementResult],
        iterations: list | None = None,
    ) -> None:
        self.step = step
        self.load_factor = load_factor
        self.U = U
        self.reactions = reactions
        self.element_results = element_results
        self.iterations = iterations or []

    def node(self, node_id: str | int, dof_mgr) -> NodeResult:
        """Return NodeResult for the given node."""
        dofs = dof_mgr.get_node_dofs(node_id)
        return NodeResult(node_id, self.U[dofs])

    def __repr__(self) -> str:
        return (
            f"StepResult(step={self.step}, "
            f"lambda={self.load_factor:.4f})"
        )


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

class Results:
    """
    Collection of all converged StepResults with hierarchical access.

    Access patterns:
        results["step_1"]                    -> StepResult
        results["step_1"]["A"]               -> NodeResult (requires dof_mgr)
        results["step_1"]["A"].uy            -> float
        results["step_1"].reactions["A"]     -> ndarray [Rx, Ry, Mrz]
        results["step_1"].element_results["e1"].forces -> ndarray

        results.last                         -> last StepResult
        results.load_factors                 -> list of lambda values
        results.node_history("A", "uy")      -> list of uy at each step

    After solver returns, Results is considered immutable.
    Calling add_step() after finalize() raises RuntimeError.
    """

    def __init__(self, dof_mgr=None) -> None:
        self._steps: dict[str, StepResult] = {}   # "step_N" -> StepResult
        self._step_list: list[StepResult] = []    # ordered
        self._finalized: bool = False
        self._dof_mgr = dof_mgr   # stored for node access

    # ------------------------------------------------------------------
    # Building (called by solver)
    # ------------------------------------------------------------------

    def add_step(self, result: StepResult) -> None:
        """Add a converged StepResult. Called by the solver."""
        if self._finalized:
            raise RuntimeError(
                "Results.add_step() called after finalize(). "
                "Results are immutable after the solver returns."
            )
        key = f"step_{result.step}"
        self._steps[key] = result
        self._step_list.append(result)

    def finalize(self) -> None:
        """Mark Results as complete and immutable."""
        self._finalized = True

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    def __getitem__(self, key: str) -> _StepProxy:
        """
        results["step_1"] -> _StepProxy for hierarchical access.
        results["step_1"]["A"] -> NodeResult
        """
        if key not in self._steps:
            available = list(self._steps.keys())
            raise KeyError(
                f"Step '{key}' not found in Results. "
                f"Available: {available}"
            )
        return _StepProxy(self._steps[key], self._dof_mgr)

    @property
    def last(self) -> StepResult:
        """Return the last converged StepResult."""
        if not self._step_list:
            raise RuntimeError("Results is empty — no steps have been added.")
        return self._step_list[-1]

    @property
    def steps(self) -> list[StepResult]:
        """Return all StepResults in order."""
        return list(self._step_list)

    @property
    def n_steps(self) -> int:
        return len(self._step_list)

    @property
    def load_factors(self) -> list[float]:
        """Return lambda for each step."""
        return [s.load_factor for s in self._step_list]

    def node_history(
        self, node_id: str | int, component: str
    ) -> list[float]:
        """
        Return the history of a displacement component across all steps.

        Parameters
        ----------
        node_id   : target node
        component : "ux" | "uy" | "rz"

        Returns
        -------
        list of float, one value per step

        Example
        -------
        >>> uy_history = results.node_history("B", "uy")
        >>> import matplotlib.pyplot as plt
        >>> plt.plot(results.load_factors, uy_history)
        """
        if self._dof_mgr is None:
            raise RuntimeError(
                "node_history() requires a DOFManager. "
                "Pass dof_mgr to Results() constructor."
            )
        component_idx = {"ux": 0, "uy": 1, "rz": 2}
        if component not in component_idx:
            raise ValueError(
                f"Invalid component '{component}'. Use 'ux', 'uy', or 'rz'."
            )
        idx = component_idx[component]
        dofs = self._dof_mgr.get_node_dofs(node_id)
        global_dof = dofs[idx]
        return [float(s.U[global_dof]) for s in self._step_list]

    def reaction_history(
        self, node_id: str | int, component: str
    ) -> list[float]:
        """
        Return the history of a reaction component across all steps.

        Parameters
        ----------
        node_id   : supported node
        component : "Rx" | "Ry" | "Mrz"
        """
        comp_idx = {"Rx": 0, "Ry": 1, "Mrz": 2}
        if component not in comp_idx:
            raise ValueError(
                f"Invalid component '{component}'. Use 'Rx', 'Ry', or 'Mrz'."
            )
        idx = comp_idx[component]
        history = []
        for s in self._step_list:
            if node_id in s.reactions:
                history.append(float(s.reactions[node_id][idx]))
            else:
                history.append(0.0)
        return history

    def summary(self) -> str:
        lines = [
            f"Results: {self.n_steps} steps",
            f"  Load factors: {[f'{lf:.3f}' for lf in self.load_factors]}",
            f"  Finalized: {self._finalized}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"Results(n_steps={self.n_steps}, finalized={self._finalized})"


# ---------------------------------------------------------------------------
# _StepProxy — enables results["step_1"]["A"].uy syntax
# ---------------------------------------------------------------------------

class _StepProxy:
    """
    Thin wrapper around StepResult that enables node/element access by ID.

    results["step_1"]["A"]      -> NodeResult
    results["step_1"]["e1"]     -> ElementResult
    results["step_1"].reactions -> dict node_id -> reaction array
    results["step_1"].load_factor -> float
    """

    def __init__(self, step_result: StepResult, dof_mgr) -> None:
        self._result = step_result
        self._dof_mgr = dof_mgr

    def __getitem__(self, obj_id: str | int):
        """Return NodeResult or ElementResult for the given ID."""
        # try node first
        if self._dof_mgr and obj_id in self._dof_mgr.node_dof_map:
            dofs = self._dof_mgr.get_node_dofs(obj_id)
            return NodeResult(obj_id, self._result.U[dofs])

        # try element
        if obj_id in self._result.element_results:
            return self._result.element_results[obj_id]

        raise KeyError(
            f"ID '{obj_id}' not found as a node or element in this step."
        )

    def __getattr__(self, name: str):
        """Delegate attribute access to the underlying StepResult."""
        return getattr(self._result, name)

    def __repr__(self) -> str:
        return f"_StepProxy({self._result!r})"
