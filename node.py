"""
node.py
=======
Node, Constraint, EndRelease.

These are pure data containers — no computation logic.

Node    : geometric point with DOF indices assigned by DOFManager.
Constraint : boundary condition attached to a node (by ID).
EndRelease : internal hinge at an element end (by DOF name).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

@dataclass
class Node:
    """
    Geometric node in the 2D plane.

    Attributes
    ----------
    id   : user-assigned identifier
    x, y : coordinates (any consistent unit system)
    dofs : global DOF indices [d_ux, d_uy, d_rz], assigned by DOFManager
           after model.finalize(). Empty list before finalize().

    Notes
    -----
    - No mass stored here — see NodalMass.
    - No material, section, or element references — pure geometry.
    - dofs is written by DOFManager; the user should never set it directly.
    """

    id: str | int
    x: float
    y: float
    dofs: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "x": self.x,
            "y": self.y,
            "dofs": self.dofs,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Node:
        return cls(
            id=data["id"],
            x=data["x"],
            y=data["y"],
            dofs=data.get("dofs", []),
        )

    def __repr__(self) -> str:
        return f"Node(id={self.id!r}, x={self.x}, y={self.y})"


# ---------------------------------------------------------------------------
# Constraint
# ---------------------------------------------------------------------------

@dataclass
class Constraint:
    """
    Boundary condition applied to a node.

    Each flag (ux, uy, rz) marks the corresponding DOF as constrained.
    The ConstraintHandler translates these flags into modifications of
    the global stiffness matrix.

    Attributes
    ----------
    id      : user-assigned identifier
    node_id : target node (reference by ID, never direct object)
    ux      : constrain horizontal displacement
    uy      : constrain vertical displacement
    rz      : constrain rotation

    Examples
    --------
    Fully fixed (all DOFs constrained):
        Constraint(id="fix_A", node_id="A", ux=True, uy=True, rz=True)

    Pin (translations constrained, rotation free):
        Constraint(id="pin_B", node_id="B", ux=True, uy=True, rz=False)

    Roller (vertical reaction only):
        Constraint(id="roll_C", node_id="C", ux=False, uy=True, rz=False)
    """

    id: str | int
    node_id: str | int
    ux: bool = False
    uy: bool = False
    rz: bool = False

    def constrained_dofs(self) -> list[str]:
        """Return list of constrained DOF names, e.g. ['ux', 'uy']."""
        result = []
        if self.ux:
            result.append("ux")
        if self.uy:
            result.append("uy")
        if self.rz:
            result.append("rz")
        return result

    def is_fully_fixed(self) -> bool:
        """Return True if all three DOFs are constrained."""
        return self.ux and self.uy and self.rz

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "node_id": self.node_id,
            "ux": self.ux,
            "uy": self.uy,
            "rz": self.rz,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Constraint:
        return cls(
            id=data["id"],
            node_id=data["node_id"],
            ux=data.get("ux", False),
            uy=data.get("uy", False),
            rz=data.get("rz", False),
        )

    def __repr__(self) -> str:
        flags = "".join([
            "ux" if self.ux else "",
            "uy" if self.uy else "",
            "rz" if self.rz else "",
        ])
        return f"Constraint(id={self.id!r}, node_id={self.node_id!r}, [{flags}])"


# ---------------------------------------------------------------------------
# EndRelease
# ---------------------------------------------------------------------------

@dataclass
class EndRelease:
    """
    Internal hinge at one end of a beam element.

    Implemented as a very soft spring (finite stiffness) to avoid
    singularity in the global stiffness matrix. The element incorporates
    the release directly in its local stiffness formulation — no extra
    DOFs are needed in the DOFManager.

    Attributes
    ----------
    dof       : released degree of freedom — "ux", "uy", or "rz"
    stiffness : residual spring stiffness (default 1e-6).
                Must be > 0 to avoid a singular K.
                Choose relative to EI/L of the connected element.
    end       : which end of the element — "i" (start node) or "j" (end node)
    law       : constitutive law for nonlinear hinge (Phase 1+).
                None -> linear spring with `stiffness`.

    Notes
    -----
    The most common case is a moment release (rz) at one or both ends,
    which turns a fixed-end beam into a pin-ended or simply supported one.

    Example
    -------
    Moment release at the j-end of element "beam_1":
        EndRelease(dof="rz", end="j")
    """

    dof: str              # "ux" | "uy" | "rz"
    end: str              # "i"  | "j"
    stiffness: float = 1e-6
    law: object = None    # ConstitutiveLaw | None — placeholder for Phase 1

    def __post_init__(self) -> None:
        valid_dofs = {"ux", "uy", "rz"}
        if self.dof not in valid_dofs:
            raise ValueError(
                f"EndRelease: invalid dof '{self.dof}'. "
                f"Must be one of {sorted(valid_dofs)}."
            )
        valid_ends = {"i", "j"}
        if self.end not in valid_ends:
            raise ValueError(
                f"EndRelease: invalid end '{self.end}'. Must be 'i' or 'j'."
            )
        if self.stiffness <= 0:
            raise ValueError(
                f"EndRelease: stiffness must be positive to avoid a singular "
                f"stiffness matrix (got {self.stiffness})."
            )

    def to_dict(self) -> dict:
        return {
            "dof": self.dof,
            "end": self.end,
            "stiffness": self.stiffness,
            # law is not serialized yet — placeholder for Phase 1
        }

    @classmethod
    def from_dict(cls, data: dict) -> EndRelease:
        return cls(
            dof=data["dof"],
            end=data["end"],
            stiffness=data.get("stiffness", 1e-6),
        )

    def __repr__(self) -> str:
        return (
            f"EndRelease(dof={self.dof!r}, end={self.end!r}, "
            f"stiffness={self.stiffness})"
        )


# ---------------------------------------------------------------------------
# NodalMass
# ---------------------------------------------------------------------------

@dataclass
class NodalMass:
    """
    Concentrated mass applied at a node.

    Used for tanks, machinery, or additional seismic masses.
    Added directly to the diagonal of the global mass matrix by the Assembler.

    Attributes
    ----------
    id      : user-assigned identifier
    node_id : target node (reference by ID)
    mx      : mass in x direction
    my      : mass in y direction
    I_theta : rotational inertia about z axis
    """

    id: str | int
    node_id: str | int
    mx: float
    my: float
    I_theta: float

    def __post_init__(self) -> None:
        if self.mx < 0 or self.my < 0 or self.I_theta < 0:
            raise ValueError(
                f"NodalMass '{self.id}': mass components must be non-negative."
            )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "node_id": self.node_id,
            "mx": self.mx,
            "my": self.my,
            "I_theta": self.I_theta,
        }

    @classmethod
    def from_dict(cls, data: dict) -> NodalMass:
        return cls(
            id=data["id"],
            node_id=data["node_id"],
            mx=data["mx"],
            my=data["my"],
            I_theta=data["I_theta"],
        )

    def __repr__(self) -> str:
        return (
            f"NodalMass(id={self.id!r}, node_id={self.node_id!r}, "
            f"mx={self.mx}, my={self.my}, I_theta={self.I_theta})"
        )
