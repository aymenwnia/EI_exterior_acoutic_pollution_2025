# -*- coding: utf-8 -*-
"""
Demonstration of Acoustic Control Optimization.
Single-frequency optimization example for educational purposes.
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Local packages
import _env
import preprocessing
import processing
import postprocessing
from multilevel import project_Uad_star, project_to_binary, compute_parametric_gradient, compute_objective_function

def run_demo():
    # Geometry Setup
    N = 50
    M = 2 * N
    spacestep = 1.0 / N
    level = 0 # Flat wall

    # Physics
    c = 343.0
    f_Hz = 295.0
    k = 2 * np.pi * f_Hz / c
    omega = k

    print(f"Running Demo. Frequency: {f_Hz} Hz")

    # PDE System
    beta_pde, alpha_pde, alpha_dir, beta_neu, alpha_rob, beta_rob = preprocessing._set_coefficients_of_pde(M, N)
    f, f_dir, f_neu, f_rob = preprocessing._set_rhs_of_pde(M, N)
    domain_omega, x, y, _, _ = preprocessing._set_geometry_of_domain(M, N, level)

    # Traffic Source
    from multilevel import build_road_source
    f, f_dir, f_neu, f_rob = build_road_source(f, f_dir, f_neu, f_rob)

    # Optimization Constraints
    alpha_rob[:, :] = -omega * 1j
    mask_R = (domain_omega == _env.NODE_ROBIN)
    V_obj = 0.4
    V_0 = V_obj * np.sum(mask_R)
    
    # Optimizer Settings
    zeta0 = 0.7
    mu1 = 1e-10
    max_iter = 100
    delta = 5e-5
    
    # Initial Material
    chi = preprocessing._set_chi(M, N, x, y,V_obj)
    chi = preprocessing.set2zero(chi, domain_omega)
    
    # Material Property (Example complex value for demonstration)
    Alpha = 11.43 - 14.30j 
    alpha_rob = Alpha * chi

    # 1. Solve Uncontrolled
    u0 = processing.solve_helmholtz(domain_omega, spacestep, omega, f, f_dir, f_neu, f_rob,
                                    beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
    J0 = compute_objective_function(domain_omega, u0, spacestep, mu1, V_0, chi)
    print(f"Uncontrolled Energy J0: {J0:.6e}")

    # 2. Optimization Loop
    energy_history = []
    chi_curr = chi.copy()
    
    for n in range(max_iter):
        # Adjoint
        p = processing.solve_adjoint(domain_omega, spacestep, omega, u0, beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
        # Gradient
        grad = compute_parametric_gradient(domain_omega, Alpha, u0, p)
        # Descent
        chi_curr = project_Uad_star(chi_curr + zeta0 * grad, mask_R, V_obj)
        alpha_rob = Alpha * chi_curr
        # Direct
        u0 = processing.solve_helmholtz(domain_omega, spacestep, omega, f, f_dir, f_neu, f_rob,
                                        beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
        
        J = compute_objective_function(domain_omega, u0, spacestep, mu1, V_0, chi_curr)
        energy_history.append(J)
        
        if n % 10 == 0: print(f"Iter {n}: J = {J:.6e}")

    # 3. Binary Projection
    chi_bin = project_to_binary(chi_curr, mask_R, V_obj)
    u_bin = processing.solve_helmholtz(domain_omega, spacestep, omega, f, f_dir, f_neu, f_rob,
                                       beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, Alpha * chi_bin)
    J_bin = compute_objective_function(domain_omega, u_bin, spacestep, mu1, V_0, chi_bin)
    
    print(f"Final Binary Energy: {J_bin:.6e}")
    
    # Plots
    postprocessing.myimshow(np.real(u_bin), title='Controlled Solution Re(u)', filename='demo_u_controlled.jpg')
    postprocessing.myimshow(chi_bin, title='Optimal Material Distribution', filename='demo_chi_optimal.jpg')
    plt.figure()
    plt.semilogy(energy_history)
    plt.title('Energy Convergence')
    plt.savefig('demo_convergence.jpg')

if __name__ == '__main__':
    run_demo()