# -*- coding: utf-8 -*-
"""
Physics module for computing frequency-dependent absorption coefficients (alpha).
Based on JCAL/Biot models for porous materials.
"""

import numpy
import scipy.io
from scipy.optimize import minimize

# Material properties dictionary
MATERIALS = {
    "MELAMINE": {"phi": 0.99, "sigma": 14000.0, "alpha_h": 1.02},
    "BIRCH": {"phi": 0.529, "sigma": 151429.0, "alpha_h": 1.37},
    "LAINE_ROCHE": {"phi": 0.98, "sigma": 30000.0, "alpha_h": 1.2},
    "LAINE_VERRE": {"phi": 0.95, "sigma": 40000.0, "alpha_h": 1.3},
    "BETON": {"phi": 0.45, "sigma": 5000.0, "alpha_h": 2.2},
}

def real_to_complex(z):
    """Converts a 2-element real array [Re, Im] back to a complex number."""
    return z[0] + 1j * z[1]

def complex_to_real(z):
    """Converts a complex number into a 2-element real array [Re, Im]."""
    return numpy.array([numpy.real(z), numpy.imag(z)])

class Memoize:
    """Decorator to cache function results to improve performance."""
    def __init__(self, f):
        self.f = f
        self.memo = {}

    def __call__(self, *args):
        if args not in self.memo:
            self.memo[args] = self.f(*args)
        return self.memo[args]

def compute_alpha(omega, material):
    """
    Computes the optimal absorption coefficient alpha for a given circular frequency.
    Uses physical parameters (porosity, resistivity, tortuosity) and minimizes error.
    """
    # Extract material properties
    phi = MATERIALS[material]["phi"]
    sigma = MATERIALS[material]["sigma"]
    alpha_h = MATERIALS[material]["alpha_h"]
    gamma_p = 7.0 / 5.0
    rho_0 = 1.2
    c_0 = 340.0

    # Geometry and Mesh parameters
    L = 0.01
    resolution = 12 

    # Derived physical parameters
    mu_0 = 1.0
    ksi_0 = 1.0 / (c_0 ** 2)
    mu_1 = phi / alpha_h
    ksi_1 = phi * gamma_p / (c_0 ** 2)
    a = sigma * (phi ** 2) * gamma_p / ((c_0 ** 2) * rho_0 * alpha_h)

    # Optimization weights
    A = 1.0
    B = 1.0

    # --- Internal Memoized Physics Functions ---
    @Memoize
    def lambda_0(k, omega):
        if k ** 2 >= (omega ** 2) * ksi_0 / mu_0:
            return numpy.sqrt(k ** 2 - (omega ** 2) * ksi_0 / mu_0)
        else:
            return numpy.sqrt((omega ** 2) * ksi_0 / mu_0 - k ** 2) * 1j

    @Memoize
    def lambda_1(k, omega):
        temp1 = (omega ** 2) * ksi_1 / mu_1
        temp2 = numpy.sqrt((k ** 2 - temp1) ** 2 + (a * omega / mu_1) ** 2)
        real = (1.0 / numpy.sqrt(2.0)) * numpy.sqrt(k ** 2 - temp1 + temp2)
        im = (-1.0 / numpy.sqrt(2.0)) * numpy.sqrt(temp1 - k ** 2 + temp2)
        return complex(real, im)

    @Memoize
    def g_k(k):
        return 1.0 if k == 0 else 0.0

    @Memoize
    def f(x, k):
        return ((lambda_0(k, omega) * mu_0 - x) * numpy.exp(-lambda_0(k, omega) * L) \
                + (lambda_0(k, omega) * mu_0 + x) * numpy.exp(lambda_0(k, omega) * L))

    @Memoize
    def chi(k, alpha, omega):
        return (g_k(k) * ((lambda_0(k, omega) * mu_0 - lambda_1(k, omega) * mu_1) \
                          / f(lambda_1(k, omega) * mu_1, k) - (lambda_0(k, omega) * mu_0 - alpha) / f(alpha, k)))

    @Memoize
    def eta(k, alpha, omega):
        return (g_k(k) * ((lambda_0(k, omega) * mu_0 + lambda_1(k, omega) * mu_1) \
                          / f(lambda_1(k, omega) * mu_1, k) - (lambda_0(k, omega) * mu_0 + alpha) / f(alpha, k)))

    @Memoize
    def e_k(k, alpha, omega):
        expm = numpy.exp(-2.0 * lambda_0(k, omega) * L)
        expp = numpy.exp(+2.0 * lambda_0(k, omega) * L)

        if k ** 2 >= (omega ** 2) * ksi_0 / mu_0:
            return ((A + B * (numpy.abs(k) ** 2)) \
                    * ((1.0 / (2.0 * lambda_0(k, omega))) \
                    * ((numpy.abs(chi(k, alpha, omega)) ** 2) * (1.0 - expm) \
                    + (numpy.abs(eta(k, alpha, omega)) ** 2) * (expp - 1.0)) \
                    + 2 * L * numpy.real(chi(k, alpha, omega) * numpy.conj(eta(k, alpha, omega)))) \
                    + B * numpy.abs(lambda_0(k, omega)) / 2.0 * ((numpy.abs(chi(k, alpha, omega)) ** 2) * (1.0 - expm) \
                    + (numpy.abs(eta(k, alpha, omega)) ** 2) * (expp - 1.0)) \
                    - 2 * B * (lambda_0(k, omega) ** 2) * L * numpy.real(
                        chi(k, alpha, omega) * numpy.conj(eta(k, alpha, omega))))
        else:
            return ((A + B * (numpy.abs(k) ** 2)) * (L \
                    * ((numpy.abs(chi(k, alpha, omega)) ** 2) + (numpy.abs(eta(k, alpha, omega)) ** 2)) \
                    + complex(0.0, 1.0) * (1.0 / lambda_0(k, omega)) * numpy.imag(
                        chi(k, alpha, omega) * numpy.conj(eta(k, alpha, omega) * (1.0 - expm))))) \
                    + B * L * (numpy.abs(lambda_0(k, omega)) ** 2) \
                    * ((numpy.abs(chi(k, alpha, omega)) ** 2) + (numpy.abs(eta(k, alpha, omega)) ** 2)) \
                    + complex(0.0, 1.0) * B * lambda_0(k, omega) * numpy.imag(
                        chi(k, alpha, omega) * numpy.conj(eta(k, alpha, omega) * (1.0 - expm)))

    @Memoize
    def sum_e_k(omega):
        def sum_func(alpha):
            s = 0.0
            for n in range(-resolution, resolution + 1):
                k = n * numpy.pi / L
                s += e_k(k, alpha, omega)
            return s
        return sum_func

    # Solve minimization problem to find alpha
    alpha_0 = numpy.array(complex(40.0, -40.0))
    res = minimize(lambda z: numpy.real(sum_e_k(omega)(real_to_complex(z))), 
                   complex_to_real(alpha_0), 
                   tol=1e-4)
    
    temp_alpha = real_to_complex(res.x)
    temp_error = numpy.real(sum_e_k(omega)(temp_alpha))

    return temp_alpha, temp_error

def run_compute_alpha(material):
    """Generates alpha values for a frequency range and saves to .mtx files."""
    print(f'Computing alpha for {material}...')
    numb_omega = 1000
    omegas = numpy.linspace(2.0 * numpy.pi, 2.0 * numpy.pi * 1000, num=numb_omega)
    
    temp = [compute_alpha(omega, material=material) for omega in omegas]
    alphas, errors = map(list, zip(*temp))
    
    alphas = numpy.array(alphas)
    errors = numpy.array(errors)
    frequencies = omegas / (2.0 * numpy.pi)

    print(f'Writing data for {material}...')
    scipy.io.mmwrite(f'dta_omega_{material}.mtx', omegas.reshape(len(omegas), 1))
    scipy.io.mmwrite(f'dta_freq_{material}.mtx', frequencies.reshape(len(frequencies), 1))
    scipy.io.mmwrite(f'dta_alpha_{material}.mtx', alphas.reshape(len(alphas), 1), field='complex')
    scipy.io.mmwrite(f'dta_error_{material}.mtx', errors.reshape(len(errors), 1))

def run():
    materials = ["MELAMINE", "BIRCH", "LAINE_ROCHE", "LAINE_VERRE", "BETON"]
    for mat in materials:
        run_compute_alpha(mat)

if __name__ == '__main__':
    run()
    print('End.')