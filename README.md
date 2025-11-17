# Acoustic Optimization on Fractal Boundaries

## 1. Project Description

This project implements a computational framework to optimize the distribution of absorbing materials on wall boundaries to minimize acoustic energy reflection. It specifically investigates the acoustic performance of **Fractal (Koch) geometries** compared to standard flat surfaces.

The simulation solves the **Helmholtz equation** using the Finite Difference Method (FDM) and employs the **Adjoint State Method** for gradient-based optimization of the material density ($\chi$).

### Core Objectives
* **Minimize Acoustic Energy:** Find the optimal layout of absorbing material to reduce noise reflection.
* **Compare Geometries:** Analyze if fractal surfaces offer better absorption properties than flat surfaces.
* **Ensure Manufacturability:** Project continuous optimization results into binary designs (material is either present or absent).

### Key Methodologies
* **Physics Modeling:** Uses JCAL/Biot models for porous materials (e.g., Melamine, Concrete).
* **Numerical Solver:** Solves the linear system $Ax=b$ for the acoustic pressure field $u$.
* **Optimization:** Uses Gradient Descent with an adaptive step size, enforcing volume constraints ($V_{obj}$) via projection.

---

## 2. Installation & Dependencies

The code relies on the standard Python scientific stack. You can install the required dependencies via pip:

```bash
pip install numpy scipy matplotlib
```
## 3. Repository Structure


README.md                   # Project documentation


 _env.py                     # [Core] Environment constants
 
 preprocessing.py            # [Core] Geometry and mesh generation
 
 processing.py               # [Core] Helmholtz & Adjoint solvers
 
 postprocessing.py           # [Core] Visualization tools
 
 multilevel.py               # [Core] Optimization logic & library
 

compute_alpha.py            # [Step 1] Material physics generator

demo_control_polycopie.py   # [Step 2] Single-run demonstration


energy_analysis.py          # [Analysis] Baseline (Fully absorbent)

energy_analysis_optimised.py# [Analysis] Initial vs Relaxed vs Binary

energymultilevel.py         # [Analysis] Theoretical limits (Sweep)

energymultilevelcomp.py     # [Analysis] Robustness (Optimize once, sweep)

mltifreq_multi_lev.py       # [Advanced] Multi-frequency optimization




