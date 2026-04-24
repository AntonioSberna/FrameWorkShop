"""
load_case.py
============
LoadCase.

A LoadCase describes the SPATIAL distribution of loads — it is completely
independent of how the load grows over time or over load steps (that is
the responsibility of LoadControl and, in Phase 3, TimeSeries).

Key design decisions:
  - LoadCase does NOT enter the Model. It lives outside and is passed
    directly to the solver.
  - ABSOLUTE approach: each LoadCase describes the TOTAL load at that
    stage, not an increment. This makes it reusable across stages.
  - Nodal loads   : node_id  -> [Fx, Fy, Mz]  (global coordinates)
  - Distributed loads : elem_id -> {q, direction, reference}
    reference = "local" | "global"
  - Validation happens in solver.solve(), not here. The LoadCase can be
    built before the model exists.

Update interface (predisposed, partial implementation):
  - update_nodal_load()      : modify or add a nodal load
  - update_distributed_load(): modify or add a distributed load
  - remove_nodal_load()      : remove a nodal load entry
  - remove_distributed_load(): remove a distributed load entry
  - scale()                  : return a scaled copy (useful for load stepping)
  - combine()                : return the sum of two LoadCases
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy import ndarray


# ---------------------------------------------------------------------------
# DistributedLoad — descriptor for a single distributed load entry
# ---------------------------------------------------------------------------

@dataclass
class DistributedLoad:
    """
    Uniformly distributed load on a beam element.

    Attributes
    ----------
    q         : load intensity (force per unit length)
    direction : "y" | "x"  in the chosen reference frame
    reference : "local" | "global"
                "local"  -> q acts along the local element axes
                "global" -> q acts along the global axes (e.g. gravity)

    Notes
    -----
    In Phase 0 only uniform loads are supported.
    Linearly varying loads (trapezoidal) are deferred to Phase 4.
    The equivalent nodal forces are computed by the Element, not here.
    """

    q: float
    direction: str = "y"       # "x" | "y"
    reference: str = "global"  # "local" | "global"

    def __post_init__(self) -> None:
        if self.direction not in ("x", "y"):
            raise ValueError(
                f"DistributedLoad: invalid direction '{self.direction}'. "
                f"Must be 'x' or 'y'."
            )
        if self.reference not in ("local", "global"):
            raise ValueError(
                f"DistributedLoad: invalid reference '{self.reference}'. "
                f"Must be 'local' or 'global'."
            )

    def scaled(self, factor: float) -> DistributedLoad:
        """Return a new DistributedLoad with q multiplied by factor."""
        return DistributedLoad(
            q=self.q * factor,
            direction=self.direction,
            reference=self.reference,
        )

    def to_dict(self) -> dict:
        return {
            "q": self.q,
            "direction": self.direction,
            "reference": self.reference,
        }

    @classmethod
    def from_dict(cls, data: dict) -> DistributedLoad:
        return cls(
            q=data["q"],
            direction=data.get("direction", "y"),
            reference=data.get("reference", "global"),
        )

    def __repr__(self) -> str:
        return (
            f"DistributedLoad(q={self.q}, direction={self.direction!r}, "
            f"reference={self.reference!r})"
        )


# ---------------------------------------------------------------------------
# LoadCase
# ---------------------------------------------------------------------------

class LoadCase:
    """
    Spatial description of a load configuration.

    Contains:
      nodal_loads       : node_id  -> ndarray([Fx, Fy, Mz])
      distributed_loads : elem_id  -> DistributedLoad

    The LoadCase is immutable by default after creation — use the update
    methods to modify it. update_*() methods return self to allow chaining.

    Example
    -------
    >>> lc = LoadCase(id="wind")
    >>> lc.update_nodal_load("B", [0.0, -10.0, 0.0])
    >>> lc.update_distributed_load("beam_1", q=-5.0, reference="global")
    """

    def __init__(self, id: str | int) -> None:
        self.id = id
        self.nodal_loads: dict[str | int, ndarray] = {}
        self.distributed_loads: dict[str | int, DistributedLoad] = {}

    # ------------------------------------------------------------------
    # Update interface — predisposed for incremental modifications
    # ------------------------------------------------------------------

    def update_nodal_load(
        self,
        node_id: str | int,
        load: list[float] | ndarray,
    ) -> LoadCase:
        """
        Set or replace the nodal load at node_id.

        Parameters
        ----------
        node_id : target node (validated by solver against the model)
        load    : [Fx, Fy, Mz] in global coordinates

        Returns self for chaining.

        TODO (Phase 4): add increment=True mode to ADD to existing load
        instead of replacing it, useful for building load histories.
        """
        load_arr = np.asarray(load, dtype=float)
        if load_arr.shape != (3,):
            raise ValueError(
                f"LoadCase '{self.id}': nodal load for node '{node_id}' "
                f"must be a 3-component vector [Fx, Fy, Mz] "
                f"(got shape {load_arr.shape})."
            )
        self.nodal_loads[node_id] = load_arr
        return self

    def update_distributed_load(
        self,
        elem_id: str | int,
        q: float,
        direction: str = "y",
        reference: str = "global",
    ) -> LoadCase:
        """
        Set or replace the distributed load on elem_id.

        Parameters
        ----------
        elem_id   : target element (validated by solver against the model)
        q         : load intensity (force per unit length)
        direction : "x" | "y" in the chosen reference frame
        reference : "local" | "global"

        Returns self for chaining.

        TODO (Phase 4): support trapezoidal loads (q_i, q_j at the two ends).
        TODO (Phase 4): support self-weight as a special flag on the element.
        """
        self.distributed_loads[elem_id] = DistributedLoad(
            q=q, direction=direction, reference=reference
        )
        return self

    def remove_nodal_load(self, node_id: str | int) -> LoadCase:
        """
        Remove the nodal load at node_id. No-op if not present.

        Returns self for chaining.
        """
        self.nodal_loads.pop(node_id, None)
        return self

    def remove_distributed_load(self, elem_id: str | int) -> LoadCase:
        """
        Remove the distributed load on elem_id. No-op if not present.

        Returns self for chaining.
        """
        self.distributed_loads.pop(elem_id, None)
        return self

    def clear(self) -> LoadCase:
        """Remove all loads. Returns self for chaining."""
        self.nodal_loads.clear()
        self.distributed_loads.clear()
        return self

    # ------------------------------------------------------------------
    # Derived LoadCases
    # ------------------------------------------------------------------

    def scale(self, factor: float) -> LoadCase:
        """
        Return a NEW LoadCase with all loads multiplied by factor.

        The original LoadCase is not modified.
        Useful for manual load stepping or load combinations.

        TODO (Phase 4): combine() will add two LoadCases together.
        """
        lc = LoadCase(id=f"{self.id}_x{factor}")
        for node_id, load in self.nodal_loads.items():
            lc.nodal_loads[node_id] = load * factor
        for elem_id, dl in self.distributed_loads.items():
            lc.distributed_loads[elem_id] = dl.scaled(factor)
        return lc

    def combine(self, other: LoadCase, factor_self: float = 1.0,
                factor_other: float = 1.0) -> LoadCase:
        """
        Return a NEW LoadCase = factor_self * self + factor_other * other.

        Loads on the same node/element are summed.
        The original LoadCases are not modified.

        TODO (Phase 4): full implementation with load combination rules
        (e.g. ASCE 7, Eurocode). For now this is a simple linear combination.
        """
        lc = LoadCase(id=f"comb_{self.id}_{other.id}")

        # nodal loads
        all_node_ids = set(self.nodal_loads) | set(other.nodal_loads)
        for node_id in all_node_ids:
            load = np.zeros(3)
            if node_id in self.nodal_loads:
                load += factor_self * self.nodal_loads[node_id]
            if node_id in other.nodal_loads:
                load += factor_other * other.nodal_loads[node_id]
            lc.nodal_loads[node_id] = load

        # distributed loads — same element: sum q values
        all_elem_ids = set(self.distributed_loads) | set(other.distributed_loads)
        for elem_id in all_elem_ids:
            if elem_id in self.distributed_loads and elem_id in other.distributed_loads:
                dl_a = self.distributed_loads[elem_id]
                dl_b = other.distributed_loads[elem_id]
                if dl_a.direction != dl_b.direction or dl_a.reference != dl_b.reference:
                    raise ValueError(
                        f"LoadCase.combine(): distributed loads on element "
                        f"'{elem_id}' have incompatible direction or reference "
                        f"frames and cannot be summed automatically."
                    )
                lc.distributed_loads[elem_id] = DistributedLoad(
                    q=factor_self * dl_a.q + factor_other * dl_b.q,
                    direction=dl_a.direction,
                    reference=dl_a.reference,
                )
            elif elem_id in self.distributed_loads:
                dl = self.distributed_loads[elem_id]
                lc.distributed_loads[elem_id] = dl.scaled(factor_self)
            else:
                dl = other.distributed_loads[elem_id]
                lc.distributed_loads[elem_id] = dl.scaled(factor_other)

        return lc

    # ------------------------------------------------------------------
    # Validation (called by solver, not here)
    # ------------------------------------------------------------------

    def validate(self, node_ids: set, elem_ids: set) -> None:
        """
        Check that all referenced IDs exist in the model.

        Called by the solver before analysis. Not called at construction
        time so that the LoadCase can be built before the model.

        Parameters
        ----------
        node_ids : set of valid node IDs from the model
        elem_ids : set of valid element IDs from the model
        """
        for node_id in self.nodal_loads:
            if node_id not in node_ids:
                raise ValueError(
                    f"LoadCase '{self.id}': node '{node_id}' not found "
                    f"in the model."
                )
        for elem_id in self.distributed_loads:
            if elem_id not in elem_ids:
                raise ValueError(
                    f"LoadCase '{self.id}': element '{elem_id}' not found "
                    f"in the model."
                )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "type": "LoadCase",
            "id": self.id,
            "nodal_loads": {
                k: v.tolist() for k, v in self.nodal_loads.items()
            },
            "distributed_loads": {
                k: v.to_dict() for k, v in self.distributed_loads.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> LoadCase:
        lc = cls(id=data["id"])
        for node_id, load in data.get("nodal_loads", {}).items():
            lc.nodal_loads[node_id] = np.array(load, dtype=float)
        for elem_id, dl in data.get("distributed_loads", {}).items():
            lc.distributed_loads[elem_id] = DistributedLoad.from_dict(dl)
        return lc

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def summary(self) -> str:
        lines = [
            f"LoadCase '{self.id}'",
            f"  Nodal loads        : {len(self.nodal_loads)}",
            f"  Distributed loads  : {len(self.distributed_loads)}",
        ]
        for node_id, load in self.nodal_loads.items():
            lines.append(f"    Node {node_id!r}: Fx={load[0]}, Fy={load[1]}, Mz={load[2]}")
        for elem_id, dl in self.distributed_loads.items():
            lines.append(f"    Elem {elem_id!r}: {dl}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"LoadCase(id={self.id!r}, "
            f"nodal={len(self.nodal_loads)}, "
            f"distributed={len(self.distributed_loads)})"
        )
