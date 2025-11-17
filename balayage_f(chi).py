# -*- coding: utf-8 -*-
"""
Post-Processing: Frequency Sweep for Loaded Material Distributions.
Loads Chi from file (Binary/Relaxed) and computes energy spectrum.
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
from multilevel import (
    load_alpha_table, alpha_of_freq, build_road_source, 
    compute_objective_function
)

def load_chi_from_file(filepath):
    """Loads Chi distribution from .txt or .npy file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.txt':
        # Expecting format: "Position, Value"
        data = np.loadtxt(filepath, delimiter=',', comments='#')
        return data[:, 1]
    elif ext == '.npy':
        return np.load(filepath)
    else:
        raise ValueError("Unsupported file format.")

def reconstruct_chi_2D(chi_1D, domain_omega):
    """Maps 1D boundary values back to 2D grid based on Robin nodes."""
    chi_2D = np.zeros_like(domain_omega, dtype=np.float64)
    mask_R = (domain_omega == _env.NODE_ROBIN)
    idx_i, idx_j = np.where(mask_R)
    
    # Ensure geometric sorting matches 1D array order
    if np.std(idx_j) > np.std(idx_i):
        sort_indices = np.lexsort((idx_i, idx_j))
    else:
        sort_indices = np.lexsort((idx_j, idx_i))
        
    inverse_sort = np.argsort(sort_indices)
    
    if len(chi_1D) != len(idx_i):
         # Fallback for simple array assignment if lengths match exactly
         if len(chi_1D) == np.sum(mask_R):
             chi_2D[idx_i, idx_j] = chi_1D
             return chi_2D
         raise ValueError(f"Dimension mismatch: Chi 1D has {len(chi_1D)}, Boundary has {len(idx_i)}")

    chi_2D[idx_i, idx_j] = chi_1D[inverse_sort]
    return chi_2D

def frequency_sweep_analysis(level, N, freq_min, freq_max, n_freq,
                             chi_binary_file, chi_relaxed_file, output_dir):
    """Performs frequency sweep on loaded Chi distributions."""
    print(f"Analyzing Level {level}...")
    
    M = 2 * N
    spacestep = 1.0 / N
    c = 343.0
    L_ref = 1.0

    # Load Physics
    freq_tab, alpha_tab = load_alpha_table('dta_freq_MELAMINE.mtx', 'dta_alpha_MELAMINE.mtx')

    # Setup Domain
    domain_omega, _, _, _, _ = preprocessing._set_geometry_of_domain(M, N, level)
    beta_pde, alpha_pde, alpha_dir, beta_neu, _, beta_rob = preprocessing._set_coefficients_of_pde(M, N)
    f, f_dir, f_neu, f_rob = preprocessing._set_rhs_of_pde(M, N)
    f, f_dir, f_neu, f_rob = build_road_source(f, f_dir, f_neu, f_rob, amplitude=2.0)

    # Load Chi
    chi_1D_bin = load_chi_from_file(chi_binary_file)
    chi_bin = reconstruct_chi_2D(chi_1D_bin, domain_omega)
    
    chi_1D_rel = load_chi_from_file(chi_relaxed_file)
    chi_rel = reconstruct_chi_2D(chi_1D_rel, domain_omega)

    # Sweep
    frequencies = np.linspace(freq_min, freq_max, n_freq)
    E_0 = np.zeros(n_freq)
    E_bin = np.zeros(n_freq)
    E_rel = np.zeros(n_freq)

    for i, freq in enumerate(frequencies):
        k = 2 * np.pi * freq / c * L_ref
        Alpha = alpha_of_freq(freq, freq_tab, alpha_tab)

        # Uncontrolled (Chi=0)
        u0 = processing.solve_helmholtz(domain_omega, spacestep, k, f, f_dir, f_neu, f_rob,
                                        beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, Alpha * np.zeros_like(chi_bin))
        E_0[i] = compute_objective_function(domain_omega, u0, spacestep, 0, 0) # No penalty for energy check

        # Binary
        ubin = processing.solve_helmholtz(domain_omega, spacestep, k, f, f_dir, f_neu, f_rob,
                                          beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, Alpha * chi_bin)
        E_bin[i] = compute_objective_function(domain_omega, ubin, spacestep, 0, 0)

        # Relaxed
        urel = processing.solve_helmholtz(domain_omega, spacestep, k, f, f_dir, f_neu, f_rob,
                                          beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, Alpha * chi_rel)
        E_rel[i] = compute_objective_function(domain_omega, urel, spacestep, 0, 0)

    # Plotting
    plt.figure(figsize=(12, 7))
    plt.semilogy(frequencies, E_0, 'r', label='Uncontrolled')
    plt.semilogy(frequencies, E_bin, 'b', label='Binary Optimized')
    plt.semilogy(frequencies, E_rel, 'g--', label='Relaxed Optimized')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Energy (Log Scale)')
    plt.title(f'Level {level} Frequency Response')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.savefig(os.path.join(output_dir, f'sweep_level_{level}.png'))
    plt.close()

if __name__ == '__main__':
    # Example usage parameters
    LEVEL = 2
    results_folder = f'results_level_{LEVEL}'
    
    # Check if files exist
    bin_file = f'{results_folder}/chi_binary_1D.txt'
    rel_file = f'{results_folder}/chi_relaxed_1D.txt'
    
    if os.path.exists(bin_file) and os.path.exists(rel_file):
        frequency_sweep_analysis(
            level=LEVEL, N=50, freq_min=100.0, freq_max=600.0, n_freq=100,
            chi_binary_file=bin_file,
            chi_relaxed_file=rel_file,
            output_dir=f'frequency_analysis_level_{LEVEL}'
        )
    else:
        print(f"Input files not found in {results_folder}. Run optimization first.")