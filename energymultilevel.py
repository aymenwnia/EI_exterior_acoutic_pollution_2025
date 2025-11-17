# -*- coding: utf-8 -*-
"""
Theoretical Limit Analysis: Multi-Level Frequency Sweep.
Optimizes the wall configuration independently at every frequency point 
to find the absolute minimum energy envelope.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Import optimization library
import multilevel 

def run_theoretical_sweep(levels_to_test, N, V_obj, f_min, f_max, n_freq):
    """
    Runs optimization at each frequency step for multiple fractal levels.
    Returns dictionary of results containing Uncontrolled (J0) and Optimized (J_bin) energies.
    """
    output_dir = "freq_sweep_results"
    os.makedirs(output_dir, exist_ok=True)
    frequencies = np.linspace(f_min, f_max, n_freq)
    results = {}

    print(f"Starting Theoretical Sweep. Range: {f_min}-{f_max} Hz")

    for level in levels_to_test:
        print(f"\n--- Optimizing Level {level} ---")
        J0_list, Jbin_list = [], []

        for i, f_Hz in enumerate(frequencies):
            print(f"  Freq {i+1}/{n_freq}: {f_Hz:.1f} Hz")
            try:
                # Optimize for this specific frequency
                res = multilevel.run_optimization_for_level(
                    level=level, N=N, f_Hz=f_Hz, V_obj=V_obj,
                    zeta0=0.15, mu1=1e-9, max_iter=150, delta=1e-4
                )
                J0_list.append(res["J0"])
                Jbin_list.append(res["J_bin"])
            except Exception as e:
                print(f"  Error at {f_Hz} Hz: {e}")
                J0_list.append(np.nan)
                Jbin_list.append(np.nan)

        results[level] = {
            "frequencies": frequencies,
            "J0": np.array(J0_list),
            "J_bin": np.array(Jbin_list),
        }

        # Plot per level
        plt.figure(figsize=(10, 6))
        plt.plot(frequencies, results[level]["J0"], "k--", label="Uncontrolled")
        plt.plot(frequencies, results[level]["J_bin"], "b-", label="Optimized Limit")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Energy")
        plt.title(f"Level {level} Theoretical Limit (Beta={V_obj})")
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"limit_level_{level}.png"))
        plt.close()

    return results

if __name__ == "__main__":
    # Settings
    N = 50
    V_obj = 0.4
    f_min, f_max = 100.0, 1000.0
    n_freq = 25 # Low number for demonstration speed
    levels = [0, 1, 2, 3]

    run_theoretical_sweep(levels, N, V_obj, f_min, f_max, n_freq)
    print("Theoretical sweep completed.")