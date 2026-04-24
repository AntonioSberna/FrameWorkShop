"""
fem/__init__.py
===============
Public API of the FEM framework.

Import everything from here:
    from fem import Model, Node, LinearStaticSolver, LoadCase, ...

Or import the whole namespace:
    import fem
    model = fem.Model()
"""

from material import MaterialState, Material, ElasticMaterial
from section import SectionState, Section, ElasticSection
from element import ElementState, Element, EulerBernoulliBeam, SectionForces, DeformedGeometry
from node import Node, Constraint, EndRelease, NodalMass
from model import Model, DOFManager
from load_case import LoadCase, DistributedLoad
from constraint import ConstraintHandler, PlainConstraintHandler
from assembler import Assembler
from convergence import IterationData, ConvergenceCriterion, ForceResidual, DisplacementIncrement, EnergyConvergence
from load_control import LoadControl, PlainLoadControl, DisplacementControl
from results import StoreOptions, SolverConfig, SolverCallbacks, ModelState, NodeResult, ElementResult, StepResult, Results
from solver import BaseSolver, LinearStaticSolver

__all__ = [
    "MaterialState", "Material", "ElasticMaterial",
    "SectionState", "Section", "ElasticSection",
    "ElementState", "Element", "EulerBernoulliBeam", "SectionForces", "DeformedGeometry",
    "Node", "Constraint", "EndRelease", "NodalMass",
    "Model", "DOFManager",
    "LoadCase", "DistributedLoad",
    "ConstraintHandler", "PlainConstraintHandler",
    "Assembler",
    "IterationData", "ConvergenceCriterion", "ForceResidual", "DisplacementIncrement", "EnergyConvergence",
    "LoadControl", "PlainLoadControl", "DisplacementControl",
    "StoreOptions", "SolverConfig", "SolverCallbacks", "ModelState",
    "NodeResult", "ElementResult", "StepResult", "Results",
    "BaseSolver", "LinearStaticSolver",
]
