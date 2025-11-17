# -*- coding: utf-8 -*-
"""
Acoustic Absorption Optimization for Multiple Fractal Levels.
Optimizes the distribution of absorbing material (Chi) on boundaries
to minimize acoustic energy using the Adjoint Method.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import time
from scipy.io import mmread

# Local packages
import _env
import preprocessing
import processing
import postprocessing

def load_alpha_table(freq_file, alpha_file):
    """Loads frequency-dependent alpha values from .mtx files."""
    freq_tab = np.array(mmread(freq_file)).flatten()
    alpha_tab = np.array(mmread(alpha_file)).flatten()
    # Sort by frequency
    idx = np.argsort(freq_tab)
    return freq_tab[idx], alpha_tab[idx]

def alpha_of_freq(freq, freq_tab, alpha_tab):
    """Interpolates complex alpha for a specific frequency."""
    freq_clamped = np.clip(freq, freq_tab[0], freq_tab[-1])
    re = np.interp(freq_clamped, freq_tab, np.real(alpha_tab))
    im = np.interp(freq_clamped, freq_tab, np.imag(alpha_tab))
    return re + 1j * im

def project_Uad_star(chi_tentative, mask_R, beta_target, tol=1e-6, max_iter=50):
    """
    Projects Chi onto the admissible set U*_ad(β):
    1. 0 <= Chi <= 1
    2. Mean(Chi) = beta_target
    Uses a binary search for the Lagrange multiplier.
    """
    vals = chi_tentative[mask_R]
    if vals.size == 0:
        return chi_tentative

    ell_min, ell_max = -2.0, 2.0
    proj_final = vals

    for _ in range(max_iter):
        ell_mid = 0.5 * (ell_min + ell_max)
        proj = np.clip(vals + ell_mid, 0.0, 1.0)
        m = proj.mean()
        
        if abs(m - beta_target) < tol:
            break
        if m > beta_target:
            ell_max = ell_mid
        else:
            ell_min = ell_mid
            
    proj_final = np.clip(vals + ell_mid, 0.0, 1.0)
    chi_new = chi_tentative.copy()
    chi_new[mask_R] = proj_final
    return chi_new

def project_to_binary(chi_relaxed, mask_R, beta_target):
    """
    Projects the relaxed Chi (continuous 0-1) to Binary Chi {0,1}.
    Sets the highest values to 1 until the target volume fraction is reached.
    Includes visualization of the projection.
    """
    chi_bin = np.zeros_like(chi_relaxed)
    idx_i, idx_j = np.where(mask_R)
    vals_relaxed = chi_relaxed[idx_i, idx_j]
    S = vals_relaxed.size

    if S == 0:
        return chi_bin

    # Sort values descending and select top percentage
    nb_ones = int(round(beta_target * S))
    nb_ones = max(0, min(nb_ones, S))
    order = np.argsort(vals_relaxed)[::-1]
    sel = order[:nb_ones]
    chi_bin[idx_i[sel], idx_j[sel]] = 1.0
    
    # Visualization
    vals_bin = chi_bin[idx_i, idx_j]
    x_position_unrolled = np.arange(S)

    plt.figure(figsize=(15, 7))
    plt.title(f"Binary Projection (Target beta = {beta_target:.2%})")
    plt.plot(x_position_unrolled, vals_relaxed, 'b-', label='Relaxed Chi (Input)', alpha=0.7)
    plt.step(x_position_unrolled, vals_bin, 'r-', where='mid', label='Binary Chi (Output)')
    plt.xlabel("Boundary Point Index")
    plt.ylabel("Chi Value")
    plt.ylim(-0.1, 1.1)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show() # Keeping plots as requested

    return chi_bin

def compute_parametric_gradient(domain_omega, Alpha, u, p):
    """
    Computes the gradient of the objective function with respect to Chi.
    g(x) = -Re(Alpha * p * conj(u)) on the Robin boundary.
    """
    M, N = np.shape(domain_omega)
    grad = np.zeros((M, N), dtype=np.float64)

    for i in range(M):
        for j in range(N):
            if domain_omega[i, j] == _env.NODE_ROBIN:
                vals_u, vals_p = [], []
                # Gather neighbors
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ii, jj = i + di, j + dj
                    if 0 <= ii < M and 0 <= jj < N:
                        if domain_omega[ii, jj] == _env.NODE_INTERIOR:
                            vals_u.append(u[ii, jj])
                            vals_p.append(p[ii, jj])
                
                if vals_u:
                    u_avg = sum(vals_u) / len(vals_u)
                    p_avg = sum(vals_p) / len(vals_p)
                    grad[i, j] = -np.real(Alpha * p_avg * np.conj(u_avg))

    return grad

def compute_objective_function(domain_omega, u, spacestep, mu1, V_0, chi=None):
    """
    Calculates the objective function J = Energy + Penalty(Volume).
    J(u,chi) = integral(|u|^2) + mu1 * (Vol(chi) - V0)^2
    """
    mask_dom = (domain_omega == _env.NODE_INTERIOR)
    E = np.sum(np.abs(u[mask_dom])**2) * (spacestep**2)

    if chi is not None:
        mask_R = (domain_omega == _env.NODE_ROBIN)
        Vol = np.sum(chi[mask_R])
        J = E + mu1 * (Vol - V_0)**2
    else:
        J = E
    return float(np.real(J))

def optimization_procedure(domain_omega, spacestep, omega, f, f_dir, f_neu, f_rob,
                          beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
                          Alpha, zeta0, chi_init, V_obj, mu1, V_0,
                          max_iter=50, delta=1e-3, verbose=True):
    """
    Main Gradient Descent Loop with Adaptive Step Size.
    1. Direct Solve -> 2. Adjoint Solve -> 3. Gradient -> 4. Update & Project
    """
    M, N = np.shape(domain_omega)
    mask_R = (domain_omega == _env.NODE_ROBIN)
    energy = np.zeros((max_iter+1, 1), dtype=np.float64)

    chi = chi_init.copy()
    alpha_rob_curr = alpha_rob.copy()

    # Initial State
    u = processing.solve_helmholtz(domain_omega, spacestep, omega, f, f_dir, f_neu, f_rob,
                                   beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob_curr)
    J = compute_objective_function(domain_omega, u, spacestep, mu1, V_0, chi=chi)
    energy[0] = J
    
    if verbose:
        print(f"Initial J = {J:.6e}, Beta = {np.mean(chi[mask_R]):.4f}")

    zeta = zeta0
    rel_tol = 1e-6

    for n in range(max_iter):
        # 1. Adjoint Problem
        p = processing.solve_adjoint(domain_omega, spacestep, omega, u,
                                     beta_pde, alpha_pde, alpha_dir,
                                     beta_neu, beta_rob, alpha_rob_curr)

        # 2. Compute Gradient
        grad = compute_parametric_gradient(domain_omega, Alpha, u, p)

        # 3. Gradient Descent with Backtracking Line Search
        best_J, best_chi = J, chi
        best_u, best_alpha_rob = u, alpha_rob_curr
        best_zeta = zeta
        found_better = False

        for j in range(10): # Max 10 halvings
            zeta_trial = zeta / (2.0**j)
            
            # Update and Project
            chi_tent = project_Uad_star(chi + zeta_trial * grad, mask_R, V_obj)
            alpha_rob_tent = Alpha * chi_tent

            # Evaluate New State
            u_tent = processing.solve_helmholtz(domain_omega, spacestep, omega, f, f_dir, f_neu, f_rob,
                                                beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob_tent)
            J_tent = compute_objective_function(domain_omega, u_tent, spacestep, mu1, V_0, chi=chi_tent)

            if J_tent < best_J * (1.0 - rel_tol):
                best_J, best_chi = J_tent, chi_tent
                best_u, best_alpha_rob = u_tent, alpha_rob_tent
                best_zeta = zeta_trial
                found_better = True
                break

        if not found_better:
            if verbose: print("Convergence: No improvement found.")
            energy[n+1:] = J
            break

        # Update current state
        chi, u, alpha_rob_curr = best_chi, best_u, best_alpha_rob
        J = best_J
        zeta = min(best_zeta * 1.2, 2.0) # Adapt step size
        energy[n+1] = J

        # Check Convergence
        diff = np.max(np.abs(chi[mask_R] - chi_init[mask_R])) # Compare with previous step
        chi_init = chi.copy()
        
        if verbose and n % 5 == 0:
             print(f"Iter {n}: J = {J:.6e}, Zeta = {zeta:.4e}, Diff = {diff:.6e}")

        if diff < delta:
            if verbose: print("Convergence: Tolerance reached.")
            energy[n+2:] = J
            break

    return chi, energy, u, grad

def build_road_source(f, f_dir, f_neu, f_rob, amplitude=2.0, n_sources=6):
    """Models traffic noise as Gaussian sources on the top boundary."""
    M, N = f_dir.shape
    i_source = 0 
    
    f_dir[:, :] = 0.0
    spacing = N // (n_sources + 1)

    # Point sources
    for n in range(n_sources):
        j_src = (n + 1) * spacing
        if j_src < N:
            f_dir[i_source, j_src] = amplitude * (1.0 + 0.0j)

    # Gaussian spread
    for j in range(N):
        dist_center = abs(j - N // 2)
        f_dir[i_source, j] *= np.exp(-(dist_center / (N / 3)) ** 2)

    return f, f_dir, f_neu, f_rob

def run_optimization_for_level(level, N, f_Hz, V_obj, zeta0, mu1, max_iter, delta):
    """Orchestrates the optimization process for a specific fractal level."""
    print(f"\n{'#'*60}\n# OPTIMIZING FRACTAL LEVEL {level}\n{'#'*60}")
    
    start_time = time.time()
    
    # Geometry & Physics
    M = 2 * N
    spacestep = 1.0 / N
    c = 343.0
    k = 2 * np.pi * f_Hz / c * 1.0
    
    # Load Material Properties
    freq_tab, alpha_tab = load_alpha_table('dta_freq_MELAMINE.mtx', 'dta_alpha_MELAMINE.mtx')
    Alpha = alpha_of_freq(f_Hz, freq_tab, alpha_tab)
    print(f"Frequency: {f_Hz} Hz, Alpha: {Alpha:.4f}")

    # Setup PDE
    beta_pde, alpha_pde, alpha_dir, beta_neu, alpha_rob, beta_rob = preprocessing._set_coefficients_of_pde(M, N)
    f, f_dir, f_neu, f_rob = preprocessing._set_rhs_of_pde(M, N)
    domain_omega, x, y, _, _ = preprocessing._set_geometry_of_domain(M, N, level)
    f, f_dir, f_neu, f_rob = build_road_source(f, f_dir, f_neu, f_rob, amplitude=2.0, n_sources=6)

    alpha_rob[:, :] = -k * 1j
    mask_R = (domain_omega == _env.NODE_ROBIN)
    V_0 = V_obj * np.sum(mask_R)

    # Initialization
    chi = preprocessing._set_chi(M, N, x, y,V_obj)
    chi = preprocessing.set2zero(chi, domain_omega)
    alpha_rob = Alpha * chi

    # Uncontrolled Reference
    u0 = processing.solve_helmholtz(domain_omega, spacestep, k, f, f_dir, f_neu, f_rob,
                                    beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
    J0 = compute_objective_function(domain_omega, u0, spacestep, mu1, V_0, chi=chi)
    print(f"Reference Energy J0: {J0:.6e}")

    # Optimization
    chi_opt, energy, u_opt, _ = optimization_procedure(
        domain_omega, spacestep, k, f, f_dir, f_neu, f_rob,
        beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
        Alpha, zeta0, chi, V_obj, mu1, V_0, max_iter, delta
    )

    # Binary Projection
    chi_bin = project_to_binary(chi_opt, mask_R, V_obj)
    alpha_rob_bin = Alpha * chi_bin
    u_bin = processing.solve_helmholtz(domain_omega, spacestep, k, f, f_dir, f_neu, f_rob,
                                       beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob_bin)
    J_bin = compute_objective_function(domain_omega, u_bin, spacestep, mu1, V_0, chi=chi_bin)

    # Save Results
    output_dir = f'results_level_{level}'
    os.makedirs(output_dir, exist_ok=True)
    postprocessing.myimshow(np.real(u0), title=f'Level {level}: Re(u0)', filename=f'{output_dir}/fig_u0_re.jpg')
    postprocessing.myimshow(chi_bin, title=f'Level {level}: Chi Binary', filename=f'{output_dir}/fig_chin_binary.jpg')
    
    elapsed = time.time() - start_time
    print(f"Done. Improvement: {(J0 - J_bin)/J0:.2%}. Time: {elapsed:.1f}s")

    return {
        'level': level, 'J0': J0, 'J_bin': J_bin, 'improvement': (J0 - J_bin) / J0,
        'chi_bin': chi_bin, 'domain': domain_omega, 'computation_time': elapsed
    }

if __name__ == '__main__':
    run_optimization_for_level(level=1, N=50, f_Hz=180.0, V_obj=0.4, zeta0=0.7, mu1=1e-10, max_iter=300, delta=5e-5)