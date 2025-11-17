# -*- coding: utf-8 -*-
"""
Frequency Sweep Analysis with Optimized Material Distribution.
1. Optimize Chi at a reference frequency (f_opt).
2. Freeze Chi (binary).
3. Sweep frequencies [f_min, f_max] to evaluate broadband performance.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Local imports
import _env
import preprocessing
import processing
import multilevel 

def compute_energy_for_chi(domain_omega, chi, frequencies,
                           freq_tab_alpha, alpha_tab_alpha,
                           c=343.0, L_ref=1.0,
                           amplitude=2.0, n_sources=6):
    """
    Calculates Acoustic Energy J(f) over a frequency range for a fixed Chi distribution.
    Returns: array of energy values.
    """
    M, N = domain_omega.shape
    spacestep = 1.0 / N
    mask_dom = (domain_omega == _env.NODE_INTERIOR)
    energies = np.zeros_like(frequencies, dtype=float)

    for kf, f_Hz in enumerate(frequencies):
        # Setup PDE params for this frequency
        beta_pde, alpha_pde, alpha_dir, beta_neu, _, beta_rob = \
            preprocessing._set_coefficients_of_pde(M, N)

        f, f_dir, f_neu, f_rob = preprocessing._set_rhs_of_pde(M, N)
        f, f_dir, f_neu, f_rob = multilevel.build_road_source(
            f, f_dir, f_neu, f_rob, amplitude=amplitude, n_sources=n_sources
        )

        # Physics
        k_phys = 2.0 * np.pi * f_Hz / c
        omega = k_phys * L_ref
        Alpha_f = multilevel.alpha_of_freq(f_Hz, freq_tab_alpha, alpha_tab_alpha)
        alpha_rob = Alpha_f * chi

        # Solve
        u = processing.solve_helmholtz(
            domain_omega, spacestep, omega,
            f, f_dir, f_neu, f_rob,
            beta_pde, alpha_pde, alpha_dir,
            beta_neu, beta_rob, alpha_rob
        )

        # Compute Energy
        J = np.sum(np.abs(u[mask_dom]) ** 2) * (spacestep ** 2)
        energies[kf] = float(np.real(J))

        if kf % 20 == 0:
            print(f"    Sweep f = {f_Hz:.1f} Hz -> J = {energies[kf]:.4e}")

    return energies

if __name__ == '__main__':
    # --- Configuration ---
    N = 50
    levels_to_test = [0, 1, 2, 3]
    beta_list = [0.4, 0.5, 0.6, 0.8]
    
    # Frequency setup
    f_opt = 180.0  # Optimization target
    frequencies = np.linspace(100.0, 400.0, 200) # Sweep range

    # Optimizer params
    zeta0 = 0.15
    mu1 = 1e-9
    max_iter = 200
    delta = 1e-4
    
    out_dir = 'freq_scan_results'
    os.makedirs(out_dir, exist_ok=True)

    print(f"Starting Analysis. Levels: {levels_to_test}, Betas: {beta_list}")
    print(f"Opt Freq: {f_opt} Hz. Sweep: {frequencies[0]}-{frequencies[-1]} Hz")

    # Load Materials
    freq_tab, alpha_tab = multilevel.load_alpha_table('dta_freq_MELAMINE.mtx', 'dta_alpha_MELAMINE.mtx')
    all_results = {}

    for level in levels_to_test:
        print(f"\n--- Processing Level {level} ---")
        level_results = {}

        # 1. Optimize for each Beta
        for beta in beta_list:
            print(f"  Optimizing for Beta={beta}...")
            res_opt = multilevel.run_optimization_for_level(
                level=level, N=N, f_Hz=f_opt, V_obj=beta,
                zeta0=zeta0, mu1=mu1, max_iter=max_iter, delta=delta
            )
            
            print("  Sweeping frequencies for optimized Chi...")
            energies = compute_energy_for_chi(
                res_opt['domain'], res_opt['chi_bin'], frequencies, freq_tab, alpha_tab
            )
            
            level_results[f'beta_{beta:.2f}'] = {
                'chi': res_opt['chi_bin'], 'energies': energies
            }

        # 2. Compute Full Absorbent Reference (Beta=1.0 everywhere on Robin)
        print("  Computing Fully Absorbent Reference...")
        domain = res_opt['domain'] # Reuse last domain
        chi_full = preprocessing.set2zero(np.ones_like(domain, dtype=float), domain)
        
        energies_full = compute_energy_for_chi(
            domain, chi_full, frequencies, freq_tab, alpha_tab
        )
        level_results['full_absorbent'] = {'energies': energies_full}

        # 3. Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(frequencies, energies_full, 'k--', linewidth=2, label='Fully Absorbent')
        
        for beta in beta_list:
            plt.plot(frequencies, level_results[f'beta_{beta:.2f}']['energies'], 
                     linewidth=2, label=f'Optimized Beta={beta}')

        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Energy')
        plt.title(f'Level {level}: Broadband Performance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(out_dir, f'level_{level}_sweep.png'), dpi=200)
        plt.close()

    print(f"Analysis Complete. Results in '{out_dir}'.")