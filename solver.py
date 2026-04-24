"""
solver.py
=========
BaseSolver (ABC), LinearStaticSolver.

The solver orchestrates the full analysis pipeline:
  1. model.finalize()  -> DOFManager
  2. validate LoadCase against model
  3. initialize ConstraintHandler and LoadControl
  4. for each step: assemble K and F, apply BCs, solve, commit, store

LinearStaticSolver:
  - Single Newton-Raphson iteration per step (linear -> always converges).
  - Supports multiple load steps (lambda ramps) for load-displacement curves.
  - Full equilibrium check after each step (SolverConfig.check_equilibrium).
  - Callbacks at each step and iteration.
  - Verbose logging optional.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy import ndarray
from scipy.sparse.linalg import spsolve

from model import Model, DOFManager
from assembler import Assembler
from constraint import ConstraintHandler, PlainConstraintHandler
from load_case import LoadCase
from load_control import LoadControl, PlainLoadControl
from convergence import ConvergenceCriterion, ForceResidual, IterationData
from results import (
    Results, StepResult, ElementResult,
    StoreOptions, SolverConfig, SolverCallbacks, ModelState,
)


# ---------------------------------------------------------------------------
# BaseSolver (ABC)
# ---------------------------------------------------------------------------

class BaseSolver(ABC):
    """
    Abstract base for all solvers.

    Subclasses implement _run_step(), which contains the core iteration
    logic (single step for linear, Newton-Raphson loop for nonlinear).

    The public solve() method handles setup, staging, and teardown.
    """

    def __init__(
        self,
        constraint_handler: ConstraintHandler | None = None,
        convergence: ConvergenceCriterion | None = None,
        config: SolverConfig | None = None,
    ) -> None:
        self.constraint_handler = constraint_handler or PlainConstraintHandler()
        self.convergence = convergence or ForceResidual(tol=1e-8)
        self.config = config or SolverConfig()
        self._assembler = Assembler()

    def solve(
        self,
        model: Model,
        load_case: LoadCase,
        control: LoadControl | None = None,
        initial_state: ModelState | None = None,
    ) -> Results:
        """
        Run the full analysis and return Results.

        Parameters
        ----------
        model         : structural model (finalized here if needed)
        load_case     : spatial load description
        control       : load stepping strategy (default: single full step)
        initial_state : starting state for staged analysis (None = virgin)

        Returns
        -------
        Results with all converged StepResults
        """
        control = control or PlainLoadControl(delta_lambda=1.0, n_steps=1)

        # --- setup ---
        dof_mgr = model.finalize()
        load_case.validate(
            node_ids=set(model.nodes.keys()),
            elem_ids=set(model.elements.keys()),
        )
        self.constraint_handler.initialize(model, dof_mgr)
        control.initialize(dof_mgr)

        # --- initial displacement vector ---
        if initial_state is not None:
            U = dof_mgr.build_U_from_state(initial_state.nodal_displacements)
            # restore element states for elements present in both stages
            for elem_id, elem in model.elements.items():
                if elem_id in initial_state.element_states:
                    elem.state = initial_state.element_states[elem_id]
        else:
            U = np.zeros(dof_mgr.n_dofs)

        results = Results(dof_mgr=dof_mgr)

        # --- step loop ---
        n_steps = control.n_steps
        for step in range(1, n_steps + 1):
            lambda_ = control.get_lambda(step)

            if self.config.verbose:
                print(f"\n--- Step {step}/{n_steps}  lambda={lambda_:.4f} ---")

            step_result, U_new = self._run_step(
                step=step,
                lambda_=lambda_,
                U_prev=U.copy(),
                model=model,
                dof_mgr=dof_mgr,
                load_case=load_case,
                control=control,
            )

            if step_result is None:
                # diverged — call callback and continue or abort
                if self.config.callbacks.on_divergence:
                    self.config.callbacks.on_divergence(None)
                continue

            U = U_new
            results.add_step(step_result)
            control.update(IterationData(
                load_factor=lambda_,
                U_trial=U,
            ))

            if self.config.callbacks.on_step:
                self.config.callbacks.on_step(step_result)

        results.finalize()
        return results

    @abstractmethod
    def _run_step(
        self,
        step: int,
        lambda_: float,
        U_prev: ndarray,
        model: Model,
        dof_mgr: DOFManager,
        load_case: LoadCase,
        control: LoadControl,
    ) -> tuple[StepResult | None, ndarray]:
        """
        Execute one load step. Return (StepResult, U_new) or (None, U_prev).
        """
        ...

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _collect_materials(self, model: Model) -> dict:
        """Return materials from the model registry."""
        return model.materials

    def _collect_element_results(
        self,
        model: Model,
        dof_mgr: DOFManager,
        U: ndarray,
    ) -> dict[str | int, ElementResult]:
        """Build ElementResult for each element from the current U."""
        elem_results = {}
        for elem_id, elem in model.elements.items():
            nodes = [model.get(nid) for nid in elem.node_ids]
            section = model.sections[elem.section_id]
            material = self._assembler._resolve_material(section, model.materials)
            forces = elem.get_internal_forces(nodes, section, material)
            state = elem.state if self.config.store.stores_element(elem_id) else None
            elem_results[elem_id] = ElementResult(elem_id, forces, state)
        return elem_results

    def _check_equilibrium(
        self,
        K: object,
        U: ndarray,
        F_ext: ndarray,
        reactions: dict,
    ) -> None:
        """
        Verify global equilibrium: ||F_ext + R - K*U|| < tol.

        R is the total reaction vector assembled from the reactions dict.
        Prints a warning if equilibrium is violated — never raises.
        """
        # check on free DOFs only — constrained DOFs have non-zero residual
        # by definition (that residual IS the reaction force)
        constrained = set(self.constraint_handler._constrained_dofs)
        free = [i for i in range(len(F_ext)) if i not in constrained]
        if not free:
            return
        F_int = K @ U
        residual_free = F_ext[free] - F_int[free]
        norm = float(np.linalg.norm(residual_free))
        f_norm = float(np.linalg.norm(F_ext[free]))
        tol = self.config.equilibrium_tol
        if norm > tol * (f_norm + 1e-14):
            print(
                f"[Solver] WARNING: global equilibrium check failed on free DOFs. "
                f"||R_free|| = {norm:.3e} (tol = {tol:.3e}). "
                f"Check boundary conditions and loads."
            )


# ---------------------------------------------------------------------------
# LinearStaticSolver
# ---------------------------------------------------------------------------

class LinearStaticSolver(BaseSolver):
    """
    Linear static solver for Phase 0.

    Solves K * U = F for each load step. Converges in exactly one
    Newton-Raphson iteration (the system is linear).

    Supports:
      - Multiple load steps (load-displacement curve for educational use).
      - Global equilibrium check.
      - Element internal force recovery.
      - Support reaction computation.
      - Full callback interface.
      - Verbose logging.

    Does NOT support:
      - Geometric nonlinearity (-> Phase 2).
      - Material nonlinearity (-> Phase 1).
      - Arc-length control (-> Phase 2).

    Example
    -------
    >>> solver = LinearStaticSolver()
    >>> results = solver.solve(model, load_case, control)
    >>> uy_tip = results["step_1"]["B"].uy
    """

    def _run_step(
        self,
        step: int,
        lambda_: float,
        U_prev: ndarray,
        model: Model,
        dof_mgr: DOFManager,
        load_case: LoadCase,
        control: LoadControl,
    ) -> tuple[StepResult | None, ndarray]:
        """
        Execute one linear step:
          1. Assemble K and F (scaled by lambda).
          2. Apply boundary conditions.
          3. Solve the linear system.
          4. Update element states and commit.
          5. Compute reactions and internal forces.
          6. Check equilibrium (optional).
        """
        # 1. assemble
        K = self._assembler.assemble_K(model, dof_mgr)
        F_ref = self._assembler.assemble_F(load_case, model, dof_mgr)
        F = F_ref * lambda_

        # 2. apply boundary conditions
        K_mod, F_mod = self.constraint_handler.apply(K, F)

        # 3. solve
        try:
            U = spsolve(K_mod, F_mod)
        except Exception as e:
            print(f"[LinearStaticSolver] Linear solve failed at step {step}: {e}")
            return None, U_prev

        # 4. update element states and commit
        for elem_id, elem in model.elements.items():
            nodes = [model.get(nid) for nid in elem.node_ids]
            section = model.sections[elem.section_id]
            material = self._assembler._resolve_material(section, model.materials)
            elem.update(nodes, section, material, U)
            elem.commit()

        # 5. compute residual for convergence data
        F_int_vec = K @ U
        residual = F - F_int_vec
        # zero out constrained DOF residuals (not meaningful after elimination)
        for d in self.constraint_handler._constrained_dofs:
            residual[d] = 0.0

        iteration_data = IterationData(
            iteration=0,
            residual=residual,
            delta_U=U - U_prev,
            U_trial=U,
            F_external=F,
            load_factor=lambda_,
        )

        if self.config.verbose:
            print(f"  {self.convergence.description(iteration_data)}")

        if self.config.callbacks.on_iteration:
            self.config.callbacks.on_iteration(iteration_data)

        # 6. reactions and element results
        reactions = self.constraint_handler.get_reactions(U, K, F)
        elem_results = self._collect_element_results(model, dof_mgr, U)

        # 7. optional equilibrium check
        if self.config.check_equilibrium:
            self._check_equilibrium(K, U, F, reactions)

        step_result = StepResult(
            step=step,
            load_factor=lambda_,
            U=U,
            reactions=reactions,
            element_results=elem_results,
            iterations=(
                [iteration_data] if self.config.store.iterations else []
            ),
        )

        if self.config.callbacks.on_convergence:
            self.config.callbacks.on_convergence(iteration_data)

        return step_result, U
