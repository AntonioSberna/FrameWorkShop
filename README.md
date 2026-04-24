# FrameWork·Shop

A didactic Python framework for linear static analysis of 2D plane frames, built incrementally as a learning tool. Every component is inspectable, readable, and testable in isolation.

**Current phase:** Phase 0 — linear elastic static analysis (complete, 347 tests passing)
**Roadmap:** material nonlinearity → geometric nonlinearity → dynamics

---

## What it does

Given a plane frame made of Euler-Bernoulli beam elements, the framework:

- assembles the global stiffness matrix **K** from element contributions
- applies boundary conditions via direct elimination
- solves **K U = F** with `scipy.sparse.linalg.spsolve`
- returns displacements, rotations, and support reactions accessible by user-assigned ID

```python
from __init__ import *

m = Model()
m.add(ElasticMaterial(id="steel", E=210_000, rho=7.85e-3))
m.add(ElasticSection(id="IPE300", material_id="steel", A=0.053, I=8.36e-3))
m.add(Node(id="A", x=0.0, y=0.0))
m.add(Node(id="B", x=5.0, y=0.0))
m.add(EulerBernoulliBeam(id="beam", node_ids=("A", "B"), section_id="IPE300"))
m.add(Support(id="fix", node_id="A", ux=True, uy=True, rz=True))

lc = LoadCase(id="tip").update_nodal_load("B", [0.0, -10.0, 0.0])

results = LinearStaticSolver().solve(m, lc)

print(results["step_1"]["B"].uy)           # vertical displacement at B
print(results["step_1"].reactions["A"])    # [Rx, Ry, Mrz] at A
```

---

## Design principles

**Objects reference each other by ID only — never by pointer.** A node does not hold a reference to its elements; an element does not hold a reference to its section object. The solver resolves IDs into objects through `Model` at runtime. This makes the system serializable, parallelizable, and easy to reason about.

**No pickle, ever.** Every class implements `to_dict()` / `from_dict()`. Large numpy arrays use `.npy` or HDF5; metadata goes in JSON.

**Read freely, write through the official interface.** All internal state is readable (so you can inspect the strain in a specific fiber at a specific iteration), but writing happens only through `update()`, `commit()`, and `revert()`.

**Unit-agnostic.** The framework does not know whether you work in kN/m or N/mm. Consistency is the user's responsibility.

**Explicit user IDs.** You assign meaningful IDs (`"col_left"`, `"A1"`). The framework never hides them behind numeric indices.

---

## Architecture

```
Node  Material  Section  Element  Support        ← model objects (live in Model)
         ↓
      Model  +  DOFManager                       ← registry + DOF numbering
         ↓
  ConstraintHandler    Assembler                 ← K, F assembly + BC enforcement
         ↓
    LoadCase  LoadControl  ConvergenceCriterion  ← analysis objects (live outside Model)
         ↓
       BaseSolver  →  LinearStaticSolver         ← solves K U = F
         ↓
         Results                                 ← hierarchical access by ID
```

Model objects enter via `model.add()` and are serializable. Analysis objects are constructed by the user and passed to the solver — `LoadCase` deliberately does not live in `Model`.

---

## Project structure

```
fem_telaio/
├── material.py          # MaterialState, Material (ABC), ElasticMaterial
├── section.py           # SectionState, Section (ABC), ElasticSection
├── element.py           # ElementState, Element (ABC), EulerBernoulliBeam
├── node.py              # Node, Support, EndRelease, NodalMass
├── model.py             # Model, DOFManager
├── constraint.py        # ConstraintHandler (ABC), PlainConstraintHandler
├── assembler.py         # Assembler — assemble_K, assemble_M, assemble_F
├── load_case.py         # LoadCase, DistributedLoad
├── convergence.py       # ConvergenceCriterion (ABC), ForceResidual, ...
├── load_control.py      # LoadControl (ABC), PlainLoadControl, DisplacementControl
├── results.py           # StoreOptions, SolverConfig, ModelState, Results, ...
├── solver.py            # BaseSolver (ABC), LinearStaticSolver
├── plotter.py           # FEMPlotter — structure, deformed shape, N/V/M diagrams
├── examples.py          # six benchmark cases
├── matrices_inspection.py  # minimal solve + print example
├── plot_examples.py     # cantilever + portal frame figures
└── tests/
    ├── test_material.py
    ├── test_section.py
    ├── test_element.py
    ├── test_node.py
    ├── test_model.py
    ├── test_constraint.py
    ├── test_assembler.py
    ├── test_load_case.py
    ├── test_convergence.py
    ├── test_load_control.py
    ├── test_results.py
    └── test_solver.py
```

---

## Installation

```bash
git clone https://github.com/<you>/planar-frame-fem
cd planar-frame-fem/fem_telaio
pip install numpy scipy matplotlib pytest
```

No package installation required — run directly from the source directory.

---

## Running the tests

```bash
cd fem_telaio
pytest tests/ -v
```

347 tests, all passing. Each module is tested in isolation; the solver tests compare against closed-form analytical solutions.

---

## Examples

**Cantilever beam — tip load:**

```python
results = LinearStaticSolver().solve(m, lc)

# FEM vs analytic: error ~1e-14 %
v_B  = results["step_1"]["B"].uy   # -PL³/(3EI)
th_B = results["step_1"]["B"].rz   # -PL²/(2EI)
```

**Portal frame — wind + uniform distributed load:**

```python
lc = (
    LoadCase(id="combo")
    .update_nodal_load("B", [H, 0.0, 0.0])
    .update_distributed_load("beam", q=-15.0, direction="y", reference="global")
)
results = LinearStaticSolver().solve(m, lc)
```

The UDL is automatically converted to equivalent nodal forces (fixed-end force formulas) inside `Assembler.assemble_F`.

**Visualisation:**

```python
dof_mgr = m.finalize()
plotter  = FEMPlotter(m, results, dof_mgr)

plotter.plot_structure(show_node_labels=True, load_case=lc)
plotter.plot_deformed(scale=50)
plotter.plot_diagrams()   # N, V, M — auto-scaled, M positive downward for horizontal beams
```

**Low-level matrix inspection:**

```python
from assembler import Assembler
from constraint import PlainConstraintHandler
from scipy.sparse.linalg import spsolve

dof_mgr          = m.finalize()
K                = Assembler().assemble_K(m, dof_mgr)
F                = Assembler().assemble_F(lc, m, dof_mgr)
K_hat, F_hat     = PlainConstraintHandler().apply(K, F)
U                = spsolve(K_hat, F_hat)
```

---

## Roadmap

| Phase | Status | Content |
|-------|--------|---------|
| 0 — Linear elastic | ✅ complete | `ElasticMaterial`, `ElasticSection`, `EulerBernoulliBeam`, `LinearStaticSolver` |
| 1 — Material NL | ⬜ next | `EPMaterial`, `FiberSection`, `NonlinearStaticSolver`, Newton-Raphson |
| 2 — Geometric NL | ⬜ | `DeformedGeometry`, geometric stiffness, `ArcLengthControl` |
| 3 — Dynamics | ⬜ | `TimeSeries`, `EigenSolver`, Newmark / HHT integrators |

---

## Dependencies

| Package | Use |
|---------|-----|
| `numpy` | arrays, linear algebra |
| `scipy` | sparse matrices, `spsolve`, quadrature |
| `matplotlib` | post-processing plots |
| `pytest` | testing |

Python 3.11+. No external FEM library dependencies.

---

## License

MIT


## Disclaimer
Readme.md file created with Claude, it can contain errors