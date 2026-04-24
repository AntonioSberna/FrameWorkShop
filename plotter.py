"""
plotter.py
==========
FEM visualization: undeformed/deformed shape, constraint symbols,
and internal force diagrams (N, V, M).

Convention for moment diagram:
  - Positive moment is drawn on the TENSION side.
  - For horizontal elements, positive M (sagging) is plotted DOWNWARD
    (tension on bottom fiber), following the structural engineering convention.
  - For vertical/inclined elements the diagram is drawn on the side
    with tension fibers (local y > 0 side).

Usage
-----
    from plotter import FEMPlotter
    from solver import LinearStaticSolver

    results = LinearStaticSolver().solve(model, load_case)
    plotter = FEMPlotter(model, results, dof_mgr)

    plotter.plot_structure(show_node_labels=True, show_elem_labels=True)
    plotter.plot_deformed(scale=100)
    plotter.plot_diagrams(n_points=20, scale_N=0.01, scale_V=0.01, scale_M=0.001)
    plt.show()
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.patches import FancyArrowPatch
from numpy import ndarray

from model import Model, DOFManager
from node import Constraint
from element import EulerBernoulliBeam
from results import Results


# ---------------------------------------------------------------------------
# Color / style constants
# ---------------------------------------------------------------------------

COLOR_UNDEFORMED  = "#2c3e50"   # dark blue-grey
COLOR_DEFORMED    = "#e74c3c"   # red
COLOR_N           = "#2980b9"   # blue  — axial
COLOR_V           = "#27ae60"   # green — shear
COLOR_M           = "#8e44ad"   # purple — moment
COLOR_CONSTRAINT  = "#7f8c8d"   # grey
COLOR_LOAD        = "#e67e22"   # orange
COLOR_NODE        = "#2c3e50"
COLOR_LABEL       = "#2c3e50"
ALPHA_FILL        = 0.25
LW_STRUCT         = 2.0
LW_DIAGRAM        = 1.5


# ---------------------------------------------------------------------------
# FEMPlotter
# ---------------------------------------------------------------------------

class FEMPlotter:
    """
    Visualization helper for a solved FEM model.

    Parameters
    ----------
    model   : finalized Model
    results : Results from solver.solve()
    dof_mgr : DOFManager from model.finalize()
    step    : which step to visualize (default: last)
    """

    def __init__(
        self,
        model: Model,
        results: Results,
        dof_mgr: DOFManager,
        step: int | None = None,
    ) -> None:
        self.model   = model
        self.results = results
        self.dof_mgr = dof_mgr

        # pick step
        if step is None:
            self._step_result = results.last
        else:
            self._step_result = results[f"step_{step}"]._result

        self.U = self._step_result.U

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def plot_structure(
        self,
        ax: plt.Axes | None = None,
        show_node_labels: bool = True,
        show_elem_labels: bool = True,
        show_constraints: bool = True,
        show_loads: bool = True,
        load_case=None,
        figsize: tuple = (10, 7),
    ) -> plt.Axes:
        """
        Plot the undeformed structure with optional labels and constraints.

        Parameters
        ----------
        show_node_labels : draw node IDs at node positions
        show_elem_labels : draw element IDs at element midpoints
        show_constraints : draw engineering constraint symbols
        show_loads       : draw nodal load arrows (requires load_case)
        load_case        : LoadCase to visualize loads from
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        self._draw_elements(ax, deformed=False)

        if show_constraints:
            self._draw_constraints(ax)

        if show_loads and load_case is not None:
            self._draw_nodal_loads(ax, load_case)

        if show_node_labels:
            self._draw_node_labels(ax)

        if show_elem_labels:
            self._draw_elem_labels(ax)

        self._draw_nodes(ax)
        self._finalize_axes(ax, title="Undeformed Structure")
        return ax

    def plot_deformed(
        self,
        ax: plt.Axes | None = None,
        scale: float = 1.0,
        show_original: bool = True,
        figsize: tuple = (10, 7),
    ) -> plt.Axes:
        """
        Plot the deformed shape overlaid on the undeformed structure.

        Parameters
        ----------
        scale         : displacement amplification factor
        show_original : draw undeformed structure in grey
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        if show_original:
            self._draw_elements(ax, deformed=False, color="#bdc3c7", lw=1.0, ls="--")

        self._draw_elements(ax, deformed=True, scale=scale)
        self._draw_nodes(ax, deformed=True, scale=scale)
        self._draw_constraints(ax)

        # legend
        orig_line  = mlines.Line2D([], [], color="#bdc3c7", ls="--", lw=1.0, label="Undeformed")
        defor_line = mlines.Line2D([], [], color=COLOR_DEFORMED, lw=LW_STRUCT, label=f"Deformed (×{scale})")
        ax.legend(handles=[orig_line, defor_line], loc="best", fontsize=9)

        self._finalize_axes(ax, title=f"Deformed Shape  (scale ×{scale})")
        return ax

    def plot_diagrams(
        self,
        n_points: int = 20,
        scale_N: float | None = None,
        scale_V: float | None = None,
        scale_M: float | None = None,
        figsize: tuple = (14, 12),
    ) -> tuple[plt.Axes, plt.Axes, plt.Axes]:
        """
        Plot N, V, M diagrams in three subplots.

        Parameters
        ----------
        n_points : number of sample points along each element
        scale_N  : diagram height scale for axial force (auto if None)
        scale_V  : diagram height scale for shear force (auto if None)
        scale_M  : diagram height scale for bending moment (auto if None)

        Returns
        -------
        (ax_N, ax_V, ax_M)
        """
        fig, axes = plt.subplots(3, 1, figsize=figsize)
        ax_N, ax_V, ax_M = axes

        # collect all values to auto-scale
        all_N, all_V, all_M = self._collect_all_forces(n_points)

        def auto_scale(values, ref_length):
            vmax = max(abs(v) for v in values) if values else 1.0
            return (ref_length * 0.25 / vmax) if vmax > 1e-14 else 1.0

        ref = self._reference_length()
        if scale_N is None:
            scale_N = auto_scale(all_N, ref)
        if scale_V is None:
            scale_V = auto_scale(all_V, ref)
        if scale_M is None:
            scale_M = auto_scale(all_M, ref)

        for ax, label, color, scale, values in [
            (ax_N, "Axial Force N  [+tension]",     COLOR_N, scale_N, all_N),
            (ax_V, "Shear Force V",                  COLOR_V, scale_V, all_V),
            (ax_M, "Bending Moment M  [+sagging↓]", COLOR_M, scale_M, all_M),
        ]:
            self._draw_elements(ax, deformed=False, color="#bdc3c7", lw=1.0)
            self._draw_constraints(ax)
            ax.set_title(label, fontsize=11, fontweight="bold")
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")

        self._draw_N_diagram(ax_N, n_points, scale_N)
        self._draw_V_diagram(ax_V, n_points, scale_V)
        self._draw_M_diagram(ax_M, n_points, scale_M)

        fig.tight_layout(pad=2.0)
        return ax_N, ax_V, ax_M

    def plot_all(
        self,
        load_case=None,
        deform_scale: float = 1.0,
        n_points: int = 20,
        figsize: tuple = (16, 14),
    ) -> plt.Figure:
        """
        Convenience: plot structure, deformed, and N/V/M in one figure.
        """
        fig = plt.figure(figsize=figsize)
        gs  = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)

        ax_struct  = fig.add_subplot(gs[0, 0])
        ax_deform  = fig.add_subplot(gs[0, 1])
        ax_N       = fig.add_subplot(gs[1, 0])
        ax_V       = fig.add_subplot(gs[1, 1])
        ax_M       = fig.add_subplot(gs[2, :])

        self.plot_structure(ax=ax_struct, load_case=load_case,
                            show_loads=load_case is not None)
        self.plot_deformed(ax=ax_deform, scale=deform_scale)

        ref = self._reference_length()
        all_N, all_V, all_M = self._collect_all_forces(n_points)

        def auto_scale(values):
            vmax = max(abs(v) for v in values) if values else 1.0
            return (ref * 0.25 / vmax) if vmax > 1e-14 else 1.0

        self._draw_elements(ax_N, deformed=False, color="#bdc3c7", lw=1.0)
        self._draw_constraints(ax_N)
        self._draw_N_diagram(ax_N, n_points, auto_scale(all_N))
        ax_N.set_title("Axial Force N  [+tension]", fontsize=10, fontweight="bold")
        ax_N.set_aspect("equal"); ax_N.grid(True, alpha=0.3)

        self._draw_elements(ax_V, deformed=False, color="#bdc3c7", lw=1.0)
        self._draw_constraints(ax_V)
        self._draw_V_diagram(ax_V, n_points, auto_scale(all_V))
        ax_V.set_title("Shear Force V", fontsize=10, fontweight="bold")
        ax_V.set_aspect("equal"); ax_V.grid(True, alpha=0.3)

        self._draw_elements(ax_M, deformed=False, color="#bdc3c7", lw=1.0)
        self._draw_constraints(ax_M)
        self._draw_M_diagram(ax_M, n_points, auto_scale(all_M))
        ax_M.set_title("Bending Moment M  [+sagging↓]", fontsize=11, fontweight="bold")
        ax_M.set_aspect("equal"); ax_M.grid(True, alpha=0.3)

        fig.suptitle("FEM Results", fontsize=14, fontweight="bold")
        return fig

    # ------------------------------------------------------------------
    # Drawing primitives
    # ------------------------------------------------------------------

    def _draw_elements(
        self,
        ax: plt.Axes,
        deformed: bool = False,
        scale: float = 1.0,
        color: str = COLOR_UNDEFORMED,
        lw: float = LW_STRUCT,
        ls: str = "-",
    ) -> None:
        for elem in self.model.elements.values():
            xi_coord, yi_coord = self._node_coords(elem.node_ids[0], deformed, scale)
            xj_coord, yj_coord = self._node_coords(elem.node_ids[1], deformed, scale)
            ax.plot([xi_coord, xj_coord], [yi_coord, yj_coord],
                    color=color, lw=lw, ls=ls, solid_capstyle="round", zorder=2)

    def _draw_nodes(
        self,
        ax: plt.Axes,
        deformed: bool = False,
        scale: float = 1.0,
    ) -> None:
        for node_id, node in self.model.nodes.items():
            x, y = self._node_coords(node_id, deformed, scale)
            color = COLOR_DEFORMED if deformed else COLOR_NODE
            ax.plot(x, y, "o", color=color, ms=5, zorder=5)

    def _draw_node_labels(self, ax: plt.Axes) -> None:
        offset = self._reference_length() * 0.04
        for node_id, node in self.model.nodes.items():
            ax.annotate(
                str(node_id),
                xy=(node.x, node.y),
                xytext=(node.x + offset, node.y + offset),
                fontsize=9, color=COLOR_LABEL,
                fontweight="bold", zorder=6,
            )

    def _draw_elem_labels(self, ax: plt.Axes) -> None:
        for elem_id, elem in self.model.elements.items():
            ni = self.model.get(elem.node_ids[0])
            nj = self.model.get(elem.node_ids[1])
            mx = (ni.x + nj.x) / 2
            my = (ni.y + nj.y) / 2
            ax.annotate(
                str(elem_id),
                xy=(mx, my),
                fontsize=8, color="#7f8c8d",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
                zorder=6,
            )

    def _draw_constraints(self, ax: plt.Axes) -> None:
        ref = self._reference_length()
        sz  = ref * 0.07   # symbol size

        for constraint in self.model.constraints.values():
            node = self.model.get(constraint.node_id)
            x, y = node.x, node.y

            if constraint.ux and constraint.uy and constraint.rz:
                # fully fixed: filled rectangle + hash lines
                self._draw_fixed_constr(ax, x, y, sz)
            elif constraint.ux and constraint.uy:
                # pin: triangle
                self._draw_pin_constr(ax, x, y, sz)
            elif constraint.uy and not constraint.ux:
                # roller (vertical reaction only): triangle on rollers
                self._draw_roller_constr(ax, x, y, sz, direction="y")
            elif constraint.ux and not constraint.uy:
                # horizontal roller
                self._draw_roller_constr(ax, x, y, sz, direction="x")

    def _draw_fixed_constr(self, ax, x, y, sz):
        """Draw a fixed (clamped) constraint: filled rectangle + hatch."""
        # determine orientation: if the node has elements coming from above
        # we draw the clamp at the bottom; otherwise figure it out from geometry
        orientation = self._constraint_orientation(x, y)

        if orientation == "bottom":   # element above, clamp at bottom
            rect = mpatches.FancyBboxPatch(
                (x - sz, y - sz * 1.2), 2 * sz, sz * 1.2,
                boxstyle="square,pad=0", facecolor=COLOR_CONSTRAINT,
                edgecolor=COLOR_CONSTRAINT, zorder=1, alpha=0.8
            )
            ax.add_patch(rect)
            # hatch lines below
            for i in range(5):
                xi = x - sz + i * sz * 0.5
                ax.plot([xi, xi - sz * 0.3], [y - sz * 1.2, y - sz * 1.8],
                        color=COLOR_CONSTRAINT, lw=1.0, zorder=1)
            ax.plot([x - sz, x + sz], [y - sz * 1.2, y - sz * 1.2],
                    color=COLOR_CONSTRAINT, lw=1.5, zorder=2)

        elif orientation == "left":   # element to the right, clamp at left
            rect = mpatches.FancyBboxPatch(
                (x - sz * 1.2, y - sz), sz * 1.2, 2 * sz,
                boxstyle="square,pad=0", facecolor=COLOR_CONSTRAINT,
                edgecolor=COLOR_CONSTRAINT, zorder=1, alpha=0.8
            )
            ax.add_patch(rect)
            for i in range(5):
                yi = y - sz + i * sz * 0.5
                ax.plot([x - sz * 1.2, x - sz * 1.8], [yi, yi - sz * 0.3],
                        color=COLOR_CONSTRAINT, lw=1.0, zorder=1)
            ax.plot([x - sz * 1.2, x - sz * 1.2], [y - sz, y + sz],
                    color=COLOR_CONSTRAINT, lw=1.5, zorder=2)

        else:   # right side clamp
            rect = mpatches.FancyBboxPatch(
                (x, y - sz), sz * 1.2, 2 * sz,
                boxstyle="square,pad=0", facecolor=COLOR_CONSTRAINT,
                edgecolor=COLOR_CONSTRAINT, zorder=1, alpha=0.8
            )
            ax.add_patch(rect)
            for i in range(5):
                yi = y - sz + i * sz * 0.5
                ax.plot([x + sz * 1.2, x + sz * 1.8], [yi, yi + sz * 0.3],
                        color=COLOR_CONSTRAINT, lw=1.0, zorder=1)
            ax.plot([x + sz * 1.2, x + sz * 1.2], [y - sz, y + sz],
                    color=COLOR_CONSTRAINT, lw=1.5, zorder=2)

    def _draw_pin_constr(self, ax, x, y, sz):
        """Draw a pin constraint: triangle pointing up."""
        tri = plt.Polygon(
            [[x, y], [x - sz, y - sz * 1.4], [x + sz, y - sz * 1.4]],
            closed=True, facecolor="white",
            edgecolor=COLOR_CONSTRAINT, lw=1.5, zorder=3
        )
        ax.add_patch(tri)
        # ground line and hatch
        ax.plot([x - sz * 1.1, x + sz * 1.1], [y - sz * 1.4, y - sz * 1.4],
                color=COLOR_CONSTRAINT, lw=1.5, zorder=3)
        for i in range(5):
            xi = x - sz + i * sz * 0.5
            ax.plot([xi, xi - sz * 0.25], [y - sz * 1.4, y - sz * 1.9],
                    color=COLOR_CONSTRAINT, lw=1.0, zorder=2)

    def _draw_roller_constr(self, ax, x, y, sz, direction="y"):
        """Draw a roller constraint: triangle + two circles."""
        if direction == "y":
            tri = plt.Polygon(
                [[x, y], [x - sz, y - sz * 1.2], [x + sz, y - sz * 1.2]],
                closed=True, facecolor="white",
                edgecolor=COLOR_CONSTRAINT, lw=1.5, zorder=3
            )
            ax.add_patch(tri)
            # rollers
            for dx in [-sz * 0.4, sz * 0.4]:
                circ = plt.Circle((x + dx, y - sz * 1.4), sz * 0.15,
                                   facecolor="white", edgecolor=COLOR_CONSTRAINT,
                                   lw=1.2, zorder=4)
                ax.add_patch(circ)
            ax.plot([x - sz * 1.1, x + sz * 1.1], [y - sz * 1.6, y - sz * 1.6],
                    color=COLOR_CONSTRAINT, lw=1.5, zorder=3)
        else:   # x roller (horizontal reaction only)
            tri = plt.Polygon(
                [[x, y], [x + sz * 1.2, y - sz], [x + sz * 1.2, y + sz]],
                closed=True, facecolor="white",
                edgecolor=COLOR_CONSTRAINT, lw=1.5, zorder=3
            )
            ax.add_patch(tri)
            for dy in [-sz * 0.4, sz * 0.4]:
                circ = plt.Circle((x + sz * 1.4, y + dy), sz * 0.15,
                                   facecolor="white", edgecolor=COLOR_CONSTRAINT,
                                   lw=1.2, zorder=4)
                ax.add_patch(circ)
            ax.plot([x + sz * 1.6, x + sz * 1.6], [y - sz * 1.1, y + sz * 1.1],
                    color=COLOR_CONSTRAINT, lw=1.5, zorder=3)

    def _draw_nodal_loads(self, ax: plt.Axes, load_case) -> None:
        ref = self._reference_length()
        max_load = max(
            (np.linalg.norm(v) for v in load_case.nodal_loads.values()),
            default=1.0
        )
        arrow_len = ref * 0.2

        for node_id, load in load_case.nodal_loads.items():
            node = self.model.get(node_id)
            fx, fy, mz = load
            # force arrow
            if abs(fx) > 1e-14 or abs(fy) > 1e-14:
                f_mag = np.sqrt(fx**2 + fy**2)
                dx = fx / max_load * arrow_len
                dy = fy / max_load * arrow_len
                ax.annotate(
                    "", xy=(node.x, node.y),
                    xytext=(node.x - dx, node.y - dy),
                    arrowprops=dict(
                        arrowstyle="-|>", color=COLOR_LOAD,
                        lw=2.0, mutation_scale=15
                    ), zorder=7
                )
                ax.text(
                    node.x - dx * 1.3, node.y - dy * 1.3,
                    f"{f_mag:.1f}", fontsize=8, color=COLOR_LOAD,
                    ha="center", va="center"
                )

    # ------------------------------------------------------------------
    # Internal force diagrams
    # ------------------------------------------------------------------

    def _draw_N_diagram(self, ax: plt.Axes, n_points: int, scale: float | None = None) -> None:
        """Axial force: positive = tension, drawn perpendicular to element."""
        if scale is None:
            all_N, _, _ = self._collect_all_forces(n_points)
            scale = self._auto_scale(all_N)
        for elem_id, elem in self.model.elements.items():
            if not isinstance(elem, EulerBernoulliBeam):
                continue
            xi_vals, N_vals, _ = self._sample_forces(elem, n_points)
            self._draw_diagram(ax, elem, xi_vals, N_vals, scale,
                               color=COLOR_N, flip_sign=False,
                               label_end=True)

    def _draw_V_diagram(self, ax: plt.Axes, n_points: int, scale: float | None = None) -> None:
        """Shear force diagram perpendicular to element axis."""
        if scale is None:
            _, all_V, _ = self._collect_all_forces(n_points)
            scale = self._auto_scale(all_V)
        for elem_id, elem in self.model.elements.items():
            if not isinstance(elem, EulerBernoulliBeam):
                continue
            xi_vals, _, V_vals, _ = self._sample_forces_full(elem, n_points)
            self._draw_diagram(ax, elem, xi_vals, V_vals, scale,
                               color=COLOR_V, flip_sign=False,
                               label_end=True)

    def _draw_M_diagram(self, ax: plt.Axes, n_points: int, scale: float | None = None) -> None:
        """
        Bending moment diagram.
        scale=None -> auto-scale.

        Convention: drawn on the TENSION side.
        For horizontal elements: positive M (sagging) -> tension on bottom
        -> diagram drawn DOWNWARD (positive side is below the beam).
        For vertical elements: positive M -> tension on right side (local y > 0)
        -> diagram drawn to the right.
        For inclined elements: tension side follows local y direction.
        """
        if scale is None:
            _, _, all_M = self._collect_all_forces(n_points)
            scale = self._auto_scale(all_M)
        for elem_id, elem in self.model.elements.items():
            if not isinstance(elem, EulerBernoulliBeam):
                continue
            xi_vals, _, _, M_vals = self._sample_forces_full(elem, n_points)

            nodes = [self.model.get(nid) for nid in elem.node_ids]
            _, alpha = elem._get_length_and_angle(nodes)

            # flip_sign controls which side of the element the diagram goes
            # For horizontal beam: alpha ~ 0, sin(alpha) ~ 0
            # local y perp direction in global = (-sin(alpha), cos(alpha))
            # For alpha=0: perp = (0, 1) — pointing up
            # Positive M sagging -> tension below -> draw DOWNWARD -> flip
            # For alpha=90 (vertical column): perp = (-1, 0) — pointing left
            # Positive M -> tension on local y side -> no flip needed
            # General rule: if cos(alpha) > 0 (beam leans right/horizontal) -> flip
            flip = np.cos(alpha) > 0.0

            self._draw_diagram(ax, elem, xi_vals, M_vals, scale,
                               color=COLOR_M, flip_sign=flip,
                               label_end=True, filled=True)

    def _auto_scale(self, values: list) -> float:
        """Auto-scale: diagram height = 25% of reference length."""
        ref = self._reference_length()
        vmax = max(abs(v) for v in values) if values else 1.0
        return (ref * 0.25 / vmax) if vmax > 1e-14 else 1.0

    def _draw_diagram(
        self,
        ax: plt.Axes,
        elem,
        xi_vals: list,
        force_vals: list,
        scale: float,
        color: str,
        flip_sign: bool = False,
        label_end: bool = False,
        filled: bool = True,
    ) -> None:
        """
        Draw a force diagram along an element.

        The diagram is plotted perpendicular to the element axis.
        flip_sign=True draws positive values on the opposite side.
        """
        nodes = [self.model.get(nid) for nid in elem.node_ids]
        ni, nj = nodes
        L, alpha = elem._get_length_and_angle(nodes)

        # unit vectors
        ex = np.array([np.cos(alpha), np.sin(alpha)])   # along element
        # perpendicular (local y in global) — pointing to tension side
        ey = np.array([-np.sin(alpha), np.cos(alpha)])

        sign = -1.0 if flip_sign else 1.0

        # build diagram polygon
        xs_base, ys_base = [], []
        xs_tip,  ys_tip  = [], []

        for xi, f in zip(xi_vals, force_vals):
            base_pt = np.array([ni.x, ni.y]) + xi * L * ex
            tip_pt  = base_pt + sign * f * scale * ey

            xs_base.append(base_pt[0])
            ys_base.append(base_pt[1])
            xs_tip.append(tip_pt[0])
            ys_tip.append(tip_pt[1])

        # filled polygon: base + reversed tips
        poly_x = xs_base + xs_tip[::-1]
        poly_y = ys_base + ys_tip[::-1]

        if filled:
            ax.fill(poly_x, poly_y, color=color, alpha=ALPHA_FILL, zorder=3)

        # outline
        ax.plot(xs_tip, ys_tip, color=color, lw=LW_DIAGRAM, zorder=4)

        # zero line (element axis)
        ax.plot(xs_base, ys_base, color=COLOR_UNDEFORMED, lw=LW_STRUCT, zorder=2)

        # end value labels
        if label_end:
            for xi, f, xb, yb, xt, yt in [
                (xi_vals[0],  force_vals[0],  xs_base[0],  ys_base[0],  xs_tip[0],  ys_tip[0]),
                (xi_vals[-1], force_vals[-1], xs_base[-1], ys_base[-1], xs_tip[-1], ys_tip[-1]),
            ]:
                if abs(f) > 1e-10 * (max(abs(v) for v in force_vals) + 1e-14):
                    ax.text(
                        xt + (xt - xb) * 0.15,
                        yt + (yt - yb) * 0.15,
                        f"{f:.3g}",
                        fontsize=7, color=color,
                        ha="center", va="center", zorder=6,
                    )

    # ------------------------------------------------------------------
    # Force sampling
    # ------------------------------------------------------------------

    def _sample_forces(self, elem, n_points: int):
        """Sample N at n_points along element. Returns (xi_vals, N_vals, L)."""
        nodes   = [self.model.get(nid) for nid in elem.node_ids]
        section = self.model.sections[elem.section_id]
        mat_id  = section.material_id
        mat     = self.model.materials[mat_id]

        xi_vals = np.linspace(0.0, 1.0, n_points)
        N_vals  = []
        for xi in xi_vals:
            sf = elem.get_section_forces(xi, nodes, section, mat)
            N_vals.append(sf.N)
        return xi_vals.tolist(), N_vals, None

    def _sample_forces_full(self, elem, n_points: int):
        """Sample N, V, M at n_points. Returns (xi_vals, N_vals, V_vals, M_vals)."""
        nodes   = [self.model.get(nid) for nid in elem.node_ids]
        section = self.model.sections[elem.section_id]
        mat     = self.model.materials[section.material_id]

        xi_vals = np.linspace(0.0, 1.0, n_points)
        N_vals, V_vals, M_vals = [], [], []
        for xi in xi_vals:
            sf = elem.get_section_forces(xi, nodes, section, mat)
            N_vals.append(sf.N)
            V_vals.append(sf.V)
            M_vals.append(sf.M)
        return xi_vals.tolist(), N_vals, V_vals, M_vals

    def _collect_all_forces(self, n_points: int):
        """Collect all N, V, M values for auto-scaling."""
        all_N, all_V, all_M = [], [], []
        for elem in self.model.elements.values():
            if not isinstance(elem, EulerBernoulliBeam):
                continue
            xi_vals, N, V, M = self._sample_forces_full(elem, n_points)
            all_N.extend(N); all_V.extend(V); all_M.extend(M)
        return all_N, all_V, all_M

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _node_coords(
        self,
        node_id: str | int,
        deformed: bool = False,
        scale: float = 1.0,
    ) -> tuple[float, float]:
        node = self.model.get(node_id)
        if not deformed:
            return node.x, node.y
        dofs = self.dof_mgr.get_node_dofs(node_id)
        return node.x + scale * self.U[dofs[0]], node.y + scale * self.U[dofs[1]]

    def _reference_length(self) -> float:
        """Characteristic length of the structure (max element length)."""
        lengths = []
        for elem in self.model.elements.values():
            ni = self.model.get(elem.node_ids[0])
            nj = self.model.get(elem.node_ids[1])
            lengths.append(np.sqrt((nj.x - ni.x)**2 + (nj.y - ni.y)**2))
        return max(lengths) if lengths else 1.0

    def _constraint_orientation(self, x: float, y: float) -> str:
        """
        Guess constraint orientation from which side elements connect.
        Returns 'bottom', 'left', or 'right'.
        """
        for elem in self.model.elements.values():
            ni = self.model.get(elem.node_ids[0])
            nj = self.model.get(elem.node_ids[1])
            for node in [ni, nj]:
                if abs(node.x - x) < 1e-10 and abs(node.y - y) < 1e-10:
                    # element connects here — figure out direction
                    other = nj if node is ni else ni
                    dx = other.x - x
                    dy = other.y - y
                    if abs(dy) > abs(dx):
                        return "bottom" if dy > 0 else "top"
                    else:
                        return "left" if dx > 0 else "right"
        return "bottom"

    def _finalize_axes(self, ax: plt.Axes, title: str = "") -> None:
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(title, fontsize=12, fontweight="bold")

        # add padding around structure
        xs = [n.x for n in self.model.nodes.values()]
        ys = [n.y for n in self.model.nodes.values()]
        pad = self._reference_length() * 0.35
        ax.set_xlim(min(xs) - pad, max(xs) + pad)
        ax.set_ylim(min(ys) - pad, max(ys) + pad)
