# -*- coding: utf-8 -*-
"""
Energy Analysis with Optimization Comparison.
Compares:
1. Initial Uniform Distribution (Chi = Beta)
2. Optimized Relaxed Solution (0 <= Chi <= 1)
3. Optimized Binary Solution (Chi in {0, 1})
Results are cached using pickle.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import time
import pickle
from scipy.io import mmread

# Local imports
import _env
import preprocessing
import processing
from multilevel import (
    load_alpha_table, alpha_of_freq, project_Uad_star, 
    project_to_binary, compute_parametric_gradient, 
    compute_objective_function, optimization_procedure, 
    build_road_source
)

def create_fractal_boundary(M, N, level, spacestep):
    """Generates domain geometry."""
    domain_omega = np.zeros((M, N), dtype=np.int64)
    domain_omega[0:M, 0:N] = _env.NODE_INTERIOR
    domain_omega[0, 0:N] = _env.NODE_DIRICHLET
    domain_omega[M - 1, 0:N] = _env.NODE_NEUMANN
    domain_omega[0:M, 0] = _env.NODE_NEUMANN
    domain_omega[0:M, N - 1] = _env.NODE_NEUMANN

    if level == 0:
        domain_omega[M - 1, 0:N] = _env.NODE_ROBIN
        shape_name = "Flat"
        x, y = np.arange(N), np.full(N, M - 1)
    else:
        nodes = preprocessing.create_fractal_nodes(
            [np.array([[0], [N]]), np.array([[N], [N]])], level
        )
        x, y = preprocessing.create_fractal_coordinates(nodes, domain_omega)
        for k in range(0, len(x) - 1):
            domain_omega[int(y[k]), int(x[k])] = _env.NODE_ROBIN
        domain_omega = preprocessing.partition_domain(domain_omega, [M - 2, N - 2])
        shape_name = f"Fractal_Level_{level}"
    
    return domain_omega, x, y, shape_name

def compute_energy_all_cases(domain_omega, spacestep, frequencies,
                             f, f_dir, f_neu, f_rob,
                             beta_pde, alpha_pde, alpha_dir,
                             beta_neu, beta_rob,
                             freq_tab, alpha_tab,
                             V_obj, zeta0, mu1, max_iter, delta,
                             cache_file=None, wall_name="wall"):
    """
    Computes energy vs frequency for Initial, Relaxed, and Binary optimized states.
    Uses caching to avoid recomputing optimization if parameters match.
    """
    if cache_file and os.path.exists(cache_file):
        with open(cache_file, 'rb') as cf:
            cached_data = pickle.load(cf)
        if np.allclose(cached_data['frequencies'], frequencies) and cached_data['V_obj'] == V_obj:
            print("Loaded results from cache.")
            return (cached_data['energies_initial'], 
                    cached_data['energies_relaxed'], 
                    cached_data['energies_binary'])
    
    c = 343.0
    L_ref = 1.0
    (M, N) = np.shape(domain_omega)
    
    energies_initial = np.zeros(len(frequencies))
    energies_relaxed = np.zeros(len(frequencies))
    energies_binary = np.zeros(len(frequencies))
    
    mask_R = (domain_omega == _env.NODE_ROBIN)
    mask_dom = (domain_omega == _env.NODE_INTERIOR)
    S = np.sum(mask_R)
    V_0 = V_obj * S

    for idx, freq in enumerate(frequencies):
        if idx % 10 == 0: print(f"Processing {freq:.1f} Hz...")

        Alpha_f = alpha_of_freq(freq, freq_tab, alpha_tab)
        k_phys = 2.0 * np.pi * freq / c
        omega = k_phys * L_ref
        
        # 1. Initial Uniform State
        chi_init = np.ones((M, N)) * V_obj
        chi_init = preprocessing.set2zero(chi_init, domain_omega)
        alpha_init = Alpha_f * chi_init
        
        u_init = processing.solve_helmholtz(
            domain_omega, spacestep, omega, f, f_dir, f_neu, f_rob,
            beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_init
        )
        energies_initial[idx] = float(np.real(np.sum(np.abs(u_init[mask_dom])**2) * spacestep**2))
        
        # 2. Optimization
        alpha_rob = Alpha_f * chi_init
        chi_relaxed, _, u_relaxed, _ = optimization_procedure(
            domain_omega, spacestep, omega, f, f_dir, f_neu, f_rob,
            beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
            Alpha_f, zeta0, chi_init, V_obj, mu1, V_0, max_iter, delta, verbose=False
        )
        energies_relaxed[idx] = float(np.real(np.sum(np.abs(u_relaxed[mask_dom])**2) * spacestep**2))
        
        # 3. Binary Projection
        chi_binary = project_to_binary(chi_relaxed, mask_R, V_obj)
        u_binary = processing.solve_helmholtz(
            domain_omega, spacestep, omega, f, f_dir, f_neu, f_rob,
            beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, Alpha_f * chi_binary
        )
        energies_binary[idx] = float(np.real(np.sum(np.abs(u_binary[mask_dom])**2) * spacestep**2))

    # Cache results
    if cache_file:
        with open(cache_file, 'wb') as cf:
            pickle.dump({
                'frequencies': frequencies,
                'V_obj': V_obj,
                'energies_initial': energies_initial,
                'energies_relaxed': energies_relaxed,
                'energies_binary': energies_binary
            }, cf)

    return energies_initial, energies_relaxed, energies_binary

if __name__ == '__main__':
    output_dir = 'energy_optimization_results'
    os.makedirs(output_dir, exist_ok=True)
    
    N = 50
    M = 2 * N
    spacestep = 1.0 / N
    
    freq_tab, alpha_tab = load_alpha_table('dta_freq_MELAMINE.mtx', 'dta_alpha_MELAMINE.mtx')
    frequencies = np.linspace(100.0, 700.0, 200)
    
    V_obj = 0.4
    zeta0 = 0.7
    mu1 = 1e-10
    max_iter = 100
    delta = 5e-5
    
    beta_pde, alpha_pde, alpha_dir, beta_neu, _, beta_rob = preprocessing._set_coefficients_of_pde(M, N)
    f, f_dir, f_neu, f_rob = preprocessing._set_rhs_of_pde(M, N)
    f, f_dir, f_neu, f_rob = build_road_source(f, f_dir, f_neu, f_rob, amplitude=2.0)
    
    wall_shapes = [
        {'level': 0, 'name': 'Flat Wall'},
        {'level': 1, 'name': 'Fractal Level 1'},
        {'level': 2, 'name': 'Fractal Level 2'},
    ]
    
    for wall in wall_shapes:
        level = wall['level']
        print(f"Processing {wall['name']}...")
        
        domain_omega, x, y, shape_name = create_fractal_boundary(M, N, level, spacestep)
        cache_file = os.path.join(output_dir, f'cache_lev{level}_beta{V_obj}.pkl')
        
        e_init, e_rel, e_bin = compute_energy_all_cases(
            domain_omega, spacestep, frequencies, f, f_dir, f_neu, f_rob,
            beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob,
            freq_tab, alpha_tab, V_obj, zeta0, mu1, max_iter, delta,
            cache_file=cache_file, wall_name=wall['name']
        )
        
        plt.figure(figsize=(14, 8))
        plt.plot(frequencies, e_init, 'g', label=f'Initial (Beta={V_obj})', alpha=0.8)
        plt.plot(frequencies, e_rel, 'b', label='Optimized Relaxed', alpha=0.8)
        plt.plot(frequencies, e_bin, 'r--', label='Optimized Binary', alpha=0.8)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Energy')
        plt.title(f'Comparison: {wall["name"]}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, f"comparison_{shape_name}.png"))
        plt.close()

    print("Optimization analysis complete.")