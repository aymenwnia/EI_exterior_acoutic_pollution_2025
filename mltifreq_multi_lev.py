# -*- coding: utf-8 -*-
"""
Multi-Frequency Acoustic Optimization (Advanced).
Optimizes Chi to minimize the SUM of energies across multiple frequencies simultaneously.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import time
from scipy.io import mmread

# Local imports
import _env
import preprocessing
import processing
import postprocessing
from multilevel import load_alpha_table, alpha_of_freq, project_Uad_star, project_to_binary, compute_parametric_gradient, build_road_source

def compute_energy(domain_omega, u, spacestep):
    """Computes Scalar Energy for a specific wave field u."""
    mask_dom = (domain_omega == _env.NODE_INTERIOR)
    # Normalized energy (factor 1/6 specific to FE/FD stencil weighting if applicable, kept consistent with original)
    return float(np.real(np.sum(np.abs(u[mask_dom])**2) * (spacestep**2))) / 6.0

def optimization_multifrequency(domain_omega, spacestep, omegas, Alphas,
                               f, f_dir, f_neu, f_rob,
                               beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob,
                               zeta0, chi_init, V_obj, mu1, V_0,
                               max_iter=50, delta=1e-3, verbose=True):
    """
    Optimizes Chi for a weighted sum of frequencies.
    Cost J = Sum(Energy(wi)) + Penalty.
    """
    M, N = np.shape(domain_omega)
    mask_R = (domain_omega == _env.NODE_ROBIN)
    n_freq = len(omegas)
    
    cost_history = np.zeros((max_iter + 1, 1))
    chi = chi_init.copy()

    # Initial Cost Calculation
    J_total = 0.0
    for k in range(n_freq):
        u_k = processing.solve_helmholtz(domain_omega, spacestep, omegas[k], f, f_dir, f_neu, f_rob,
                                         beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, Alphas[k] * chi)
        J_total += compute_energy(domain_omega, u_k, spacestep)
    
    J_penalty = mu1 * (np.sum(chi[mask_R]) - V_0)**2
    J = J_total + J_penalty
    cost_history[0] = J
    
    if verbose: print(f"Initial J = {J:.6e} (Energy={J_total:.6e}, Pen={J_penalty:.6e})")

    zeta = zeta0

    for n in range(max_iter):
        grad_total = np.zeros((M, N))
        u_list = []

        # 1. Direct States & 2. Adjoint States per Frequency
        for k in range(n_freq):
            # Direct
            u_k = processing.solve_helmholtz(domain_omega, spacestep, omegas[k], f, f_dir, f_neu, f_rob,
                                             beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, Alphas[k] * chi)
            u_list.append(u_k)
            
            # Adjoint
            p_k = processing.solve_adjoint(domain_omega, spacestep, omegas[k], u_k,
                                           beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, Alphas[k] * chi)
            
            # Accumulate Gradient
            grad_total += compute_parametric_gradient(domain_omega, Alphas[k], u_k, p_k)

        # Add Penalty Gradient: d/dChi [mu*(Vol - V0)^2] = 2*mu*(Vol - V0)
        vol_current = np.sum(chi[mask_R])
        grad_penalty = np.zeros((M, N))
        grad_penalty[mask_R] = 2.0 * mu1 * (vol_current - V_0)
        grad_total += grad_penalty

        # 3. Line Search
        best_J, best_chi = J, chi
        found_better = False
        
        for j in range(10):
            zeta_trial = zeta / (2.0**j)
            
            # Descent direction: chi_new = chi + zeta * grad 
            # (Note: grad calculation usually includes the negative sign in Adjoint derivation)
            chi_tent = project_Uad_star(chi + zeta_trial * grad_total, mask_R, V_obj)
            
            # Evaluate J_tent
            J_E_tent = 0.0
            for k in range(n_freq):
                u_tent = processing.solve_helmholtz(domain_omega, spacestep, omegas[k], f, f_dir, f_neu, f_rob,
                                                    beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, Alphas[k] * chi_tent)
                J_E_tent += compute_energy(domain_omega, u_tent, spacestep)
            
            J_pen_tent = mu1 * (np.sum(chi_tent[mask_R]) - V_0)**2
            J_tent = J_E_tent + J_pen_tent

            if J_tent < best_J:
                best_J, best_chi = J_tent, chi_tent
                zeta = zeta_trial # Update step size baseline
                found_better = True
                break
        
        if not found_better:
            if verbose: print("Stalled.")
            break

        chi = best_chi
        J = best_J
        cost_history[n+1] = J
        
        # Convergence check
        if np.max(np.abs(chi[mask_R] - chi_init[mask_R])) < delta:
            if verbose: print("Converged.")
            break
        chi_init = chi.copy()

        if verbose and n % 10 == 0:
            print(f"Iter {n}: J = {J:.6e}")

    return chi, cost_history

def run_multifreq_optimization():
    # Configuration
    N = 50
    level = 2
    frequencies_hz = [215.0, 260.0, 390.0, 460.0, 540.0, 580.0]
    V_obj = 0.61
    
    # Physics
    M = 2 * N
    spacestep = 1.0 / N
    c = 343.0
    omegas, Alphas = [], []
    
    freq_tab, alpha_tab = load_alpha_table('dta_freq_MELAMINE.mtx', 'dta_alpha_MELAMINE.mtx')
    
    print("Target Frequencies:")
    for f in frequencies_hz:
        k = 2 * np.pi * f / c
        a = alpha_of_freq(f, freq_tab, alpha_tab)
        omegas.append(k)
        Alphas.append(a)
        print(f"  {f} Hz : Alpha={a:.2f}")

    # Setup Domain
    beta_pde, alpha_pde, alpha_dir, beta_neu, _, beta_rob = preprocessing._set_coefficients_of_pde(M, N)
    f_rhs, f_dir, f_neu, f_rob = preprocessing._set_rhs_of_pde(M, N)
    domain_omega, x, y, _, _ = preprocessing._set_geometry_of_domain(M, N, level)
    f_rhs, f_dir, f_neu, f_rob = build_road_source(f_rhs, f_dir, f_neu, f_rob)

    # Initial Chi
    chi_init = preprocessing._set_chi(M, N, x, y,V_obj)
    chi_init = preprocessing.set2zero(chi_init, domain_omega)
    mask_R = (domain_omega == _env.NODE_ROBIN)
    V_0 = V_obj * np.sum(mask_R)

    # Run
    print(f"\nStarting Multi-Freq Optimization (Target Beta={V_obj})...")
    chi_opt, cost_hist = optimization_multifrequency(
        domain_omega, spacestep, omegas, Alphas,
        f_rhs, f_dir, f_neu, f_rob,
        beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob,
        zeta0=2.0, chi_init=chi_init, V_obj=V_obj, mu1=1e-10, V_0=V_0,
        max_iter=100, delta=1e-6
    )

    # Binary Projection
    chi_bin = project_to_binary(chi_opt, mask_R, V_obj)

    # Visualize
    output_dir = f'results_multifreq_lev{level}'
    os.makedirs(output_dir, exist_ok=True)
    postprocessing.myimshow(chi_bin, title='Optimized Chi (Binary)', filename=f'{output_dir}/chi_final.jpg')
    
    plt.figure()
    plt.semilogy(cost_hist)
    plt.title("Convergence")
    plt.savefig(f'{output_dir}/convergence.jpg')
    plt.close()
    print(f"Done. Results in {output_dir}")

if __name__ == '__main__':
    run_multifreq_optimization()