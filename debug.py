
import sys
import numpy as np

from node import Node, Constraint
from section import ElasticSection
from element import EulerBernoulliBeam
from material import ElasticMaterial
from model import Model
from load_case import LoadCase
from load_control import PlainLoadControl
from results import SolverConfig, StoreOptions, SolverCallbacks
from solver import LinearStaticSolver


L = 5           # m
E = 210_000     # kN/m^2

b = 0.1          # m
h = 0.5          # m

A = b * h        # m^2
I = (b*h**3)/12  # m^4
F = 10          # kN




mat = ElasticMaterial(id="steel", E=E, rho=7.85)
sec = ElasticSection(id="sec", material_id="steel", A=A, I=I)



m = Model()
m.add(Node(id=1, x=0, y=0))
m.add(Node(id=2, x=L, y=0))
m.add(sec)
m.add(EulerBernoulliBeam(id=1, node_ids=(1, 2), section_id="sec"))
m.add(Constraint(id=1, node_id=1, ux=True, uy=True, rz=True))

lc = LoadCase(id="tip_load").update_nodal_load(2, [0, F, 0])

solv =  LinearStaticSolver(config=SolverConfig(
        check_equilibrium=True,
        verbose=True,
        store=StoreOptions(iterations=False)))

results = solv.solve(m, lc, materials={"steel": mat})

