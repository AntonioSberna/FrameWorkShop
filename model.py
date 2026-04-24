"""
model.py
========
DOFManager, Model.

Model  : registry of all structural objects. Validates, stores, and
         provides access by user ID. Does NOT assemble, solve, or
         post-process.

DOFManager : assigns global DOF indices to all nodes after finalize().
             Numbering is sequential — RCM optimization deferred to Phase 4.

Canonical add() order:
    materials -> sections -> nodes -> constraints -> elements -> nodal masses

LoadCase is NOT added to the Model (it lives outside, passed to the solver).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from numpy import ndarray

from node import Node, Constraint, NodalMass
from element import Element
from section import Section
from material import Material


# ---------------------------------------------------------------------------
# DOFManager
# ---------------------------------------------------------------------------

class DOFManager:
    """
    Assigns and tracks global DOF indices for all nodes.

    Each node gets exactly 3 DOFs: [ux, uy, rz].
    Numbering is sequential in node insertion order.
    RCM bandwidth optimization is deferred to Phase 4.

    After finalize(), node.dofs is populated with the global indices.
    The DOFManager is the single source of truth for DOF numbering.
    """

    def __init__(self) -> None:
        self.node_dof_map: dict[str | int, list[int]] = {}
        self.elem_dof_map: dict[str | int, list[int]] = {}
        self.n_dofs: int = 0

    def build(self, model: "Model") -> None:
        """
        Assign DOF indices to all nodes and build element DOF maps.

        Called by Model.finalize(). Idempotent if called again.
        """
        self.node_dof_map = {}
        self.elem_dof_map = {}
        dof_counter = 0

        for node_id, node in model.nodes.items():
            dofs = [dof_counter, dof_counter + 1, dof_counter + 2]
            self.node_dof_map[node_id] = dofs
            node.dofs = dofs          # write back to node (the only place)
            dof_counter += 3

        self.n_dofs = dof_counter

        for elem_id, elem in model.elements.items():
            id_i, id_j = elem.node_ids
            dofs = self.node_dof_map[id_i] + self.node_dof_map[id_j]
            self.elem_dof_map[elem_id] = dofs

    def get_node_dofs(self, node_id: str | int) -> list[int]:
        """Return [d_ux, d_uy, d_rz] for the given node."""
        if node_id not in self.node_dof_map:
            raise KeyError(
                f"Node '{node_id}' not found in DOFManager. "
                f"Has model.finalize() been called?"
            )
        return self.node_dof_map[node_id]

    def get_elem_dofs(self, elem_id: str | int) -> list[int]:
        """Return the 6 global DOF indices for the given element."""
        if elem_id not in self.elem_dof_map:
            raise KeyError(
                f"Element '{elem_id}' not found in DOFManager. "
                f"Has model.finalize() been called?"
            )
        return self.elem_dof_map[elem_id]

    def local_to_global(self, elem_id: str | int) -> list[int]:
        """Alias for get_elem_dofs — matches CLAUDE.md naming."""
        return self.get_elem_dofs(elem_id)

    def build_U_from_state(
        self, nodal_displacements: dict[str | int, ndarray]
    ) -> ndarray:
        """
        Build a global displacement vector from a node_id -> [ux,uy,rz] map.

        Used in staged analysis to carry displacements between stages
        with potentially different DOF numbering.
        """
        U = np.zeros(self.n_dofs)
        for node_id, u_node in nodal_displacements.items():
            if node_id in self.node_dof_map:
                dofs = self.node_dof_map[node_id]
                U[dofs] = u_node
        return U

    def extract_state_from_U(
        self, U: ndarray
    ) -> dict[str | int, ndarray]:
        """
        Extract a node_id -> [ux, uy, rz] map from a global displacement vector.

        Used after solving to store results in a DOF-numbering-independent format.
        """
        state = {}
        for node_id, dofs in self.node_dof_map.items():
            state[node_id] = U[dofs].copy()
        return state

    def __repr__(self) -> str:
        return (
            f"DOFManager(n_dofs={self.n_dofs}, "
            f"nodes={len(self.node_dof_map)}, "
            f"elements={len(self.elem_dof_map)})"
        )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Model:
    """
    Registry and validator for all structural model objects.

    Responsibilities:
      - Store nodes, elements, sections, constraints, nodal masses by user ID.
      - Validate references (e.g. element node_ids must exist).
      - Auto-assign integer IDs when the user does not provide one.
      - Expose finalize() which triggers DOF numbering.
      - Provide read access via get(), get_all(), find().

    NOT responsible for:
      - Assembly (-> Assembler)
      - Solving  (-> Solver)
      - Post-processing (-> Results)
      - LoadCase storage (LoadCase lives outside the Model)

    Sections and Materials are NOT registered here — they are collected
    on demand by iterating elements (collect-on-demand pattern).
    """

    def __init__(self) -> None:
        self.materials: dict[str | int, Material] = {}
        self.nodes: dict[str | int, Node] = {}
        self.elements: dict[str | int, Element] = {}
        self.sections: dict[str | int, Section] = {}
        self.constraints: dict[str | int, Constraint] = {}
        self.nodal_masses: dict[str | int, NodalMass] = {}

        self._next_id: dict[type, int] = {}
        self._dirty: bool = True
        self._dof_mgr: DOFManager | None = None

    # ------------------------------------------------------------------
    # add()
    # ------------------------------------------------------------------

    def add(self, obj: Any) -> Any:
        """
        Register an object in the model.

        Accepts: Node, Element, Section, Constraint, NodalMass.
        Auto-assigns an integer ID if obj.id is None.
        Validates ID uniqueness and cross-references.

        Returns the object (with ID assigned if auto-generated),
        so the caller can chain: node = model.add(Node(id=None, ...))
        """
        if isinstance(obj, Material):
            return self._add_material(obj)
        elif isinstance(obj, Node):
            return self._add_node(obj)
        elif isinstance(obj, Element):
            return self._add_element(obj)
        elif isinstance(obj, Section):
            return self._add_section(obj)
        elif isinstance(obj, Constraint):
            return self._add_constraint(obj)
        elif isinstance(obj, NodalMass):
            return self._add_nodal_mass(obj)
        else:
            raise TypeError(
                f"Model.add() does not accept objects of type "
                f"'{type(obj).__name__}'. "
                f"Accepted types: Material, Node, Element, Section, Constraint, NodalMass."
            )

    def _auto_id(self, type_: type) -> int:
        """Return next auto-incremented integer ID for the given type."""
        current = self._next_id.get(type_, 1)
        self._next_id[type_] = current + 1
        return current

    def _add_material(self, material: Material) -> Material:
        if material.id is None:
            material.id = self._auto_id(Material)
        if material.id in self.materials:
            raise ValueError(
                f"A material with id '{material.id}' already exists in the model."
            )
        self.materials[material.id] = material
        self._dirty = True
        return material

    def _add_node(self, node: Node) -> Node:
        if node.id is None:
            node.id = self._auto_id(Node)
        if node.id in self.nodes:
            raise ValueError(
                f"A node with id '{node.id}' already exists in the model."
            )
        self.nodes[node.id] = node
        self._dirty = True
        return node

    def _add_element(self, elem: Element) -> Element:
        if elem.id is None:
            elem.id = self._auto_id(Element)
        if elem.id in self.elements:
            raise ValueError(
                f"An element with id '{elem.id}' already exists in the model."
            )
        # validate node references
        for nid in elem.node_ids:
            if nid not in self.nodes:
                raise ValueError(
                    f"Element '{elem.id}': node '{nid}' not found. "
                    f"Add nodes before elements."
                )
        # validate section reference
        if elem.section_id not in self.sections:
            raise ValueError(
                f"Element '{elem.id}': section '{elem.section_id}' not found. "
                f"Add sections before elements."
            )
        self.elements[elem.id] = elem
        self._dirty = True
        return elem

    def _add_section(self, section: Section) -> Section:
        if section.id is None:
            section.id = self._auto_id(Section)
        if section.id in self.sections:
            raise ValueError(
                f"A section with id '{section.id}' already exists in the model."
            )
        if section.material_id not in self.materials:
            raise ValueError(
                f"Section '{section.id}': material '{section.material_id}' not found. "
                f"Add materials before sections."
            )
        self.sections[section.id] = section
        self._dirty = True
        return section

    def _add_constraint(self, constraint: Constraint) -> Constraint:
        if constraint.id is None:
            constraint.id = self._auto_id(Constraint)
        if constraint.id in self.constraints:
            raise ValueError(
                f"A constraint with id '{constraint.id}' already exists in the model."
            )
        if constraint.node_id not in self.nodes:
            raise ValueError(
                f"Constraint '{constraint.id}': node '{constraint.node_id}' not found."
            )
        self.constraints[constraint.id] = constraint
        self._dirty = True
        return constraint

    def _add_nodal_mass(self, mass: NodalMass) -> NodalMass:
        if mass.id is None:
            mass.id = self._auto_id(NodalMass)
        if mass.id in self.nodal_masses:
            raise ValueError(
                f"A nodal mass with id '{mass.id}' already exists in the model."
            )
        if mass.node_id not in self.nodes:
            raise ValueError(
                f"NodalMass '{mass.id}': node '{mass.node_id}' not found."
            )
        self.nodal_masses[mass.id] = mass
        self._dirty = True
        return mass

    # ------------------------------------------------------------------
    # remove()
    # ------------------------------------------------------------------

    def remove(self, obj_id: str | int) -> None:
        """
        Remove an object from the model by ID.

        Raises if other objects reference it (e.g. cannot remove a node
        that is used by an element).
        """
        # check if it is a node referenced by elements
        if obj_id in self.materials:
            for sid, sec in self.sections.items():
                if sec.material_id == obj_id:
                    raise ValueError(
                        f"Cannot remove material '{obj_id}': "
                        f"it is referenced by section '{sid}'."
                    )
            del self.materials[obj_id]
            self._dirty = True
            return

        if obj_id in self.nodes:
            for eid, elem in self.elements.items():
                if obj_id in elem.node_ids:
                    raise ValueError(
                        f"Cannot remove node '{obj_id}': "
                        f"it is referenced by element '{eid}'."
                    )
            for sid, con in self.constraints.items():
                if con.node_id == obj_id:
                    raise ValueError(
                        f"Cannot remove node '{obj_id}': "
                        f"it is referenced by constraint '{sid}'."
                    )
            for mid, mass in self.nodal_masses.items():
                if mass.node_id == obj_id:
                    raise ValueError(
                        f"Cannot remove node '{obj_id}': "
                        f"it is referenced by nodal mass '{mid}'."
                    )
            del self.nodes[obj_id]
            self._dirty = True
            return

        if obj_id in self.elements:
            del self.elements[obj_id]
            self._dirty = True
            return

        if obj_id in self.sections:
            for eid, elem in self.elements.items():
                if elem.section_id == obj_id:
                    raise ValueError(
                        f"Cannot remove section '{obj_id}': "
                        f"it is referenced by element '{eid}'."
                    )
            del self.sections[obj_id]
            self._dirty = True
            return

        if obj_id in self.constraints:
            del self.constraints[obj_id]
            self._dirty = True
            return

        if obj_id in self.nodal_masses:
            del self.nodal_masses[obj_id]
            self._dirty = True
            return

        raise KeyError(f"No object with id '{obj_id}' found in the model.")

    # ------------------------------------------------------------------
    # without()
    # ------------------------------------------------------------------

    def without(self, element_id: str | int) -> "Model":
        """
        Return a new Model without the specified element.

        The original model is not modified. The new model shares the same
        node, section, constraint, and nodal mass objects (shallow copy).
        Used for staged analysis.
        """
        if element_id not in self.elements:
            raise KeyError(
                f"Element '{element_id}' not found in the model."
            )
        new_model = Model()
        new_model.materials = dict(self.materials)
        new_model.nodes = dict(self.nodes)
        new_model.sections = dict(self.sections)
        new_model.constraints = dict(self.constraints)
        new_model.nodal_masses = dict(self.nodal_masses)
        new_model.elements = {
            eid: elem
            for eid, elem in self.elements.items()
            if eid != element_id
        }
        new_model._next_id = dict(self._next_id)
        new_model._dirty = True
        return new_model

    # ------------------------------------------------------------------
    # finalize()
    # ------------------------------------------------------------------

    def finalize(self) -> DOFManager:
        """
        Assign DOF indices to all nodes and build element DOF maps.

        Idempotent: if the model has not changed since the last call
        (_dirty=False), returns the cached DOFManager.

        After finalize(), node.dofs is populated and the model is
        considered read-only until the next add() or remove().
        """
        if not self._dirty and self._dof_mgr is not None:
            return self._dof_mgr

        self._validate_all()

        dof_mgr = DOFManager()
        dof_mgr.build(self)

        self._dof_mgr = dof_mgr
        self._dirty = False
        return dof_mgr

    def _validate_all(self) -> None:
        """Full cross-reference validation before DOF assignment."""
        for sec_id, sec in self.sections.items():
            if sec.material_id not in self.materials:
                raise ValueError(
                    f"Section '{sec_id}': material '{sec.material_id}' not found."
                )
        for elem_id, elem in self.elements.items():
            for nid in elem.node_ids:
                if nid not in self.nodes:
                    raise ValueError(
                        f"Element '{elem_id}': node '{nid}' not found."
                    )
            if elem.section_id not in self.sections:
                raise ValueError(
                    f"Element '{elem_id}': section '{elem.section_id}' "
                    f"not found."
                )
        for con_id, con in self.constraints.items():
            if con.node_id not in self.nodes:
                raise ValueError(
                    f"Constraint '{con_id}': node '{con.node_id}' not found."
                )
        for mid, mass in self.nodal_masses.items():
            if mass.node_id not in self.nodes:
                raise ValueError(
                    f"NodalMass '{mid}': node '{mass.node_id}' not found."
                )

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    def get(self, obj_id: str | int) -> Any:
        """Return any object by ID, searching all registries."""
        for registry in (
            self.nodes, self.elements, self.sections,
            self.constraints, self.nodal_masses
        ):
            if obj_id in registry:
                return registry[obj_id]
        raise KeyError(f"No object with id '{obj_id}' found in the model.")

    def get_all(self, type_: type) -> dict:
        """Return the full registry for a given type."""
        mapping = {
            Material: self.materials,
            Node: self.nodes,
            Element: self.elements,
            Section: self.sections,
            Constraint: self.constraints,
            NodalMass: self.nodal_masses,
        }
        if type_ not in mapping:
            raise TypeError(
                f"Unknown type '{type_.__name__}'. "
                f"Use Material, Node, Element, Section, Constraint, or NodalMass."
            )
        return mapping[type_]

    def find(self, predicate: Callable[[Any], bool]) -> dict:
        """
        Return all objects (across all registries) matching predicate.

        Example: find all nodes with x > 5.0
            model.find(lambda obj: isinstance(obj, Node) and obj.x > 5.0)
        """
        result = {}
        for registry in (
            self.materials, self.nodes, self.elements,
            self.sections, self.constraints, self.nodal_masses
        ):
            for obj_id, obj in registry.items():
                if predicate(obj):
                    result[obj_id] = obj
        return result

    def get_materials(self) -> dict:
        """Return the registered materials dict (material_id -> Material)."""
        return self.materials

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "materials": {k: v.to_dict() for k, v in self.materials.items()},
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
            "elements": {k: v.to_dict() for k, v in self.elements.items()},
            "sections": {k: v.to_dict() for k, v in self.sections.items()},
            "constraints": {k: v.to_dict() for k, v in self.constraints.items()},
            "nodal_masses": {
                k: v.to_dict() for k, v in self.nodal_masses.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Model":
        from element import EulerBernoulliBeam
        from section import ElasticSection
        from material import ElasticMaterial

        model = cls()

        # materials first
        for d in data.get("materials", {}).values():
            type_ = d.get("type")
            if type_ == "ElasticMaterial":
                model._add_material(ElasticMaterial.from_dict(d))
            else:
                raise ValueError(f"Unknown material type: '{type_}'")

        # then nodes
        for d in data.get("nodes", {}).values():
            model._add_node(Node.from_dict(d))

        # sections before elements
        for d in data.get("sections", {}).values():
            type_ = d.get("type")
            if type_ == "ElasticSection":
                model._add_section(ElasticSection.from_dict(d))
            else:
                raise ValueError(f"Unknown section type: '{type_}'")

        # elements
        for d in data.get("elements", {}).values():
            type_ = d.get("type")
            if type_ == "EulerBernoulliBeam":
                model._add_element(EulerBernoulliBeam.from_dict(d))
            else:
                raise ValueError(f"Unknown element type: '{type_}'")

        # constraints
        for d in data.get("constraints", {}).values():
            model._add_constraint(Constraint.from_dict(d))

        # nodal masses
        for d in data.get("nodal_masses", {}).values():
            model._add_nodal_mass(NodalMass.from_dict(d))

        return model

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a short human-readable summary of the model contents."""
        lines = [
            "Model summary",
            f"  Materials    : {len(self.materials)}",
            f"  Nodes        : {len(self.nodes)}",
            f"  Elements     : {len(self.elements)}",
            f"  Sections     : {len(self.sections)}",
            f"  Constraints  : {len(self.constraints)}",
            f"  Nodal masses : {len(self.nodal_masses)}",
            f"  Finalized    : {not self._dirty}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"Model(materials={len(self.materials)}, nodes={len(self.nodes)}, "
            f"elements={len(self.elements)}, finalized={not self._dirty})"
        )
