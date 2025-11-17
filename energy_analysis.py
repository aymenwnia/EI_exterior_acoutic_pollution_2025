# -*- coding: utf-8 -*-
"""
Energy Analysis for Fully Absorbent Walls (Reference Case).
Computes acoustic energy vs frequency for walls with uniform absorption.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.io import mmread

# Local packages
import _env
import preprocessing
import processing

def load_alpha_table(freq_file, alpha_file):
    """Loads frequency-dependent alpha values."""
    freq_tab = np.array(mmread(freq_file)).flatten()
    alpha_tab = np.array(mmread(alpha_file)).flatten()
    idx = np.argsort(freq_tab)
    return freq_tab[idx], alpha_tab[idx]

def alpha_of_freq(freq, freq_tab, alpha_tab):
    """Interpolates complex alpha for a given frequency."""
    freq_clamped = np.clip(freq, freq_tab[0], freq_tab[-1])
    re = np.interp(freq_clamped, freq_tab, np.real(alpha_tab))
    im = np.interp(freq_clamped, freq_tab, np.imag(alpha_tab))
    return re + 1j * im

def build_road_source(f, f_dir, f_neu, f_rob, amplitude=2.0, n_sources=6):
    """Models highway noise as a Gaussian distribution on the top boundary."""
    f[:, :] = 0.0
    f_dir[:, :] = 0.0
    M, N = f_dir.shape
    i_source = 0 
    spacing = N // (n_sources + 1)

    for n in range(n_sources):
        j_src = (n + 1) * spacing
        if j_src < N:
            f_dir[i_source, j_src] = amplitude * (1.0 + 0.0j)

    for j in range(N):
        dist_center = abs(j - N // 2)
        f_dir[i_source, j] *= np.exp(-(dist_center / (N / 3)) ** 2)

    return f, f_dir, f_neu, f_rob

def compute_energy_vs_frequency(domain_omega, spacestep, frequencies,
                                f, f_dir, f_neu, f_rob,
                                beta_pde, alpha_pde, alpha_dir,
                                beta_neu, beta_rob,
                                freq_tab_alpha, alpha_tab_alpha,
                                chi, wall_name="wall"):
    """
    Sweeps through frequencies and calculates acoustic energy for a fixed wall configuration.
    """
    c = 343.0
    L_ref = 1.0
    energies = np.zeros(len(frequencies), dtype=np.float64)

    print(f"Computing energy for {wall_name}...")

    for idx, freq in enumerate(frequencies):
        k_phys = 2.0 * np.pi * freq / c
        omega = k_phys * L_ref

        Alpha_f = alpha_of_freq(freq, freq_tab_alpha, alpha_tab_alpha)
        alpha_rob = Alpha_f * chi

        u = processing.solve_helmholtz(domain_omega, spacestep, omega,
                                       f, f_dir, f_neu, f_rob,
                                       beta_pde, alpha_pde, alpha_dir,
                                       beta_neu, beta_rob, alpha_rob)

        mask_dom = (domain_omega == _env.NODE_INTERIOR)
        # Factor 2 used for energy normalization in this context
        energy = np.sum(np.abs(u[mask_dom])*2) * (spacestep*2)
        energies[idx] = float(np.real(energy))

    return energies

def create_fractal_boundary(M, N, level, spacestep):
    """Generates domain geometry (Flat or Fractal)."""
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

if __name__ == '__main__':
    output_dir = 'energy_analysis_results'
    os.makedirs(output_dir, exist_ok=True)

    N = 50
    M = 2 * N
    spacestep = 1.0 / N

    # Load Material
    freq_tab, alpha_tab = load_alpha_table('dta_freq_MELAMINE.mtx', 'dta_alpha_MELAMINE.mtx')
    
    # Frequency Sweep Settings
    freq_min = max(100.0, freq_tab[0])
    freq_max = min(1000.0, freq_tab[-1])
    frequencies = np.linspace(freq_min, freq_max, 700)

    # PDE Setup
    beta_pde, alpha_pde, alpha_dir, beta_neu, _, beta_rob = preprocessing._set_coefficients_of_pde(M, N)
    f, f_dir, f_neu, f_rob = preprocessing._set_rhs_of_pde(M, N)
    f, f_dir, f_neu, f_rob = build_road_source(f, f_dir, f_neu, f_rob, amplitude=2.0)

    wall_shapes = [
        {'level': 0, 'name': 'Flat Wall'},
        {'level': 1, 'name': 'Fractal Level 1'},
        {'level': 2, 'name': 'Fractal Level 2'},
    ]

    all_results = {}

    for wall_config in wall_shapes:
        level = wall_config['level']
        wall_name = wall_config['name']
        
        domain_omega, x, y, shape_name = create_fractal_boundary(M, N, level, spacestep)
        
        # Full absorption everywhere on the boundary
        chi = np.ones((M, N), dtype=np.float64)
        chi = preprocessing.set2zero(chi, domain_omega)

        energies = compute_energy_vs_frequency(
            domain_omega, spacestep, frequencies,
            f, f_dir, f_neu, f_rob,
            beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob,
            freq_tab, alpha_tab, chi, wall_name=wall_name
        )

        all_results[shape_name] = {'frequencies': frequencies, 'energies': energies, 'wall_name': wall_name}

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(frequencies, energies, linewidth=2.5, color='darkblue', label='Energy')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Acoustic Energy')
        plt.title(f'Energy vs Frequency - {wall_name} (Fully Absorbent)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"energy_{shape_name.lower().replace(' ', '_')}_absorbent.png"))
        plt.close()

    # Combined Plot
    plt.figure(figsize=(12, 8))
    for idx, (shape_name, res) in enumerate(all_results.items()):
        plt.plot(res['frequencies'], res['energies'], label=res['wall_name'], linewidth=2)

    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Acoustic Energy')
    plt.title('Comparison: Fully Absorbent Fractal Levels')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'energy_all_shapes_absorbent.png'))
    plt.close()
    
    print(f"Analysis complete. Results saved in '{output_dir}'.")