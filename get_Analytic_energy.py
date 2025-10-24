#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  5 14:16:54 2025

@author: gabriel
"""

import numpy as np

def get_cuartic_solution(c0, c1, c2, c3):
    # Componentes de la expresión
    term1 = -((2 * c2) / 3)
    term2 = c3**2 / 4
    term3_numerator = 12 * c0 + c2**2 - 3 * c1 * c3
    term3_denominator = (27 * c1**2 - 72 * c0 * c2 + 2 * c2**3 - 9 * c1 * c2 * c3 + 27 * c0 * c3**2) + np.sqrt(
        -4 * (12 * c0 + c2**2 - 3 * c1 * c3)**3 + (27 * c1**2 - 72 * c0 * c2 + 2 * c2**3 - 9 * c1 * c2 * c3 + 27 * c0 * c3**2)**2
        )
    term3 = (2**(1/3) * term3_numerator) / (3 * term3_denominator**(1/3))
    term4 = 1 / (3 * 2**(1/3)) * term3_denominator**(1/3)
    
    sqrt_term1 = np.sqrt(term1 + term2 + term3 + term4)
    
    term5 = -((4 * c2) / 3)
    term6 = c3**2 / 2
    term7 = -(2**(1/3) * term3_numerator) / (3 * term3_denominator**(1/3))
    term8 = -1 / (3 * 2**(1/3)) * term3_denominator**(1/3)
    
    additional_term = (-8 * c1 + 4 * c2 * c3 - c3**3) / (4 * np.sqrt(term1 + term2 + term3 + term4))

    sqrt_term2 = np.sqrt(term5 + term6 + term7 + term8 - additional_term)
    sqrt_term3 = np.sqrt(term5 + term6 + term7 + term8 + additional_term)
    
    result = np.array([-c3 / 4 - 1/2 * sqrt_term1 - 1/2 * sqrt_term2,
              -c3 / 4 - 1/2 * sqrt_term1 + 1/2 * sqrt_term2,
              -c3 / 4 + 1/2 * sqrt_term1 - 1/2 * sqrt_term3,
              -c3 / 4 + 1/2 * sqrt_term1 + 1/2 * sqrt_term3])
    
    return result

def solve_quartic_numpy(e, d, c, b, a):
    """
    Solves a quartic equation ax^4 + bx^3 + cx^2 + dx + e = 0 using numpy.roots.

    Args:
        a, b, c, d, e: Coefficients of the quartic equation.

    Returns:
        An array containing the four roots of the quartic equation.
    """
    coefficients = [a, b, c, d, e]
    roots = np.roots(coefficients)
    return roots

def compute_array(kx, ky, B, gamma, lambda_, phi_x, Delta, mu):
    """
    Compute the array from the given mathematical expression
    
    Parameters:
    B, ky, kx, gamma, lambda_, phi_x, Delta, mu: scalar parameters
    
    Returns:
    numpy array of length 5
    """
    
    # First element (the long complex one)
    elem1 = (1/16) * (
        B**4 + ky**8 - 
        4 * B**3 * lambda_ * phi_x + 
        (Delta**2 + (kx * (-kx * gamma + lambda_) + mu)**2 - 
         (2 * kx**2 * gamma**2 - 2 * kx * gamma * lambda_ + lambda_**2 + 2 * gamma * mu) * phi_x**2 + 
         gamma**2 * phi_x**4) * 
        (Delta**2 + (-kx * (kx * gamma + lambda_) + mu)**2 - 
         (2 * kx**2 * gamma**2 + 2 * kx * gamma * lambda_ + lambda_**2 + 2 * gamma * mu) * phi_x**2 + 
         gamma**2 * phi_x**4) + 
        ky**6 * (-2 * (lambda_**2 + 2 * mu) + 4 * gamma * (kx**2 + phi_x**2)) + 
        ky**4 * (6 * kx**4 * gamma**2 + 2 * Delta**2 + lambda_**4 + 
                 6 * (mu - gamma * phi_x**2)**2 - 
                 2 * kx**2 * (lambda_**2 + 2 * gamma * lambda_**2 + 6 * gamma * mu - 2 * gamma**2 * phi_x**2) + 
                 lambda_**2 * (4 * mu - 2 * (1 + 2 * gamma) * phi_x**2)) - 
        2 * B**2 * (ky**4 + kx**4 * gamma**2 + Delta**2 - 
                    3 * lambda_**2 * phi_x**2 + (mu - gamma * phi_x**2)**2 + 
                    kx**2 * (lambda_**2 - 2 * gamma * mu + 6 * gamma**2 * phi_x**2) + 
                    ky**2 * (-lambda_**2 - 2 * mu + 2 * gamma * (kx**2 + phi_x**2))) + 
        4 * B * lambda_ * phi_x * (ky**4 - 
                                   3 * kx**4 * gamma**2 + Delta**2 - lambda_**2 * phi_x**2 + 
                                   (mu - gamma * phi_x**2)**2 - 
                                   ky**2 * (lambda_**2 + 2 * mu + 2 * gamma * (kx - phi_x) * (kx + phi_x)) + 
                                   kx**2 * (lambda_**2 + 2 * gamma * (mu + gamma * phi_x**2))) + 
        2 * ky**2 * (2 * kx**6 * gamma**3 + 
                     Delta**2 * (lambda_**2 - 2 * mu + 2 * gamma * phi_x**2) + 
                     (lambda_**2 + 2 * mu - 2 * gamma * phi_x**2) * 
                     (mu + phi_x * (lambda_ - gamma * phi_x)) * 
                     (-mu + phi_x * (lambda_ + gamma * phi_x)) - 
                     kx**4 * gamma * ((2 + gamma) * lambda_**2 + 2 * gamma * (3 * mu + gamma * phi_x**2)) + 
                     kx**2 * (lambda_**4 + 2 * lambda_**2 * mu + 
                              2 * gamma * (Delta**2 + lambda_**2 * (mu + (2 - 3 * gamma) * phi_x**2) + 
                                          (mu - gamma * phi_x**2) * (3 * mu + gamma * phi_x**2))))
    )
    
    # Second element
    elem2 = kx * (
        B**2 * gamma * phi_x + 
        B * lambda_ * (ky**2 - mu + gamma * (kx - phi_x) * (kx + phi_x)) + 
        phi_x * (ky**4 * gamma + 
                 kx**4 * gamma**3 + lambda_**2 * mu - 
                 2 * kx**2 * gamma**2 * (mu + gamma * phi_x**2) + 
                 gamma * (Delta**2 + (mu - gamma * phi_x**2)**2) + 
                 ky**2 * (-lambda_**2 + gamma * (lambda_**2 - 2 * mu + 2 * gamma * (kx**2 + phi_x**2))))
    )
    
    # Third element
    elem3 = 0.5 * (
        -B**2 - ky**4 - kx**4 * gamma**2 - Delta**2 - 
        ky**2 * (2 * kx**2 * gamma + lambda_**2 - 2 * mu) - mu**2 - 
        kx**2 * (lambda_**2 - 2 * gamma * mu) + 
        2 * B * lambda_ * phi_x - 
        phi_x**2 * (lambda_**2 + gamma * (2 * ky**2 - 2 * (5 * kx**2 * gamma + mu) + gamma * phi_x**2))
    )
    
    # Fourth element
    elem4 = -4 * kx * gamma * phi_x
    
    elem5 = 1
    
    return np.array([elem1, elem2, elem3, elem4, elem5])

#@jit
def GetAnalyticEnergies(k_x, k_y, B, gamma, Lambda, phi_x, Delta, mu):
    energies = np.zeros(4)
    coefficient_array = compute_array(k_x, k_y, B, gamma, Lambda, phi_x, Delta, mu)    
    c_0, c_1, c_2, c_3, c_4 = coefficient_array
    for m in range(4):
        energies[m] = np.real(solve_quartic_numpy(c_0, c_1, c_2, c_3, c_4)[m])  
    return energies

def GetSumOfPositiveAnalyticEnergy(k_x, k_y, B, gamma, Lambda, phi_x, Delta, mu):
    positive_energy = []
    coefficient_array = compute_array(k_x, k_y, B, gamma, Lambda, phi_x, Delta, mu)      
    c_0, c_1, c_2, c_3, c_4 = coefficient_array
    for m in range(4):
        energy = np.real(solve_quartic_numpy(c_0, c_1, c_2, c_3, c_4)[m])  
        if energy>0:
            positive_energy.append(energy)
    return np.sum(np.array(positive_energy))

if __name__ == "__main__":
    
    B = 2
    gamma = 1
    Lambda = 1
    phi_x = 1
    Delta = 1
    mu = 3
    
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    k_x_values = np.linspace(-np.pi, np.pi, 300)
    k_y_values = np.linspace(-np.pi, np.pi, 300)
    X, Y = np.meshgrid(k_x_values, k_y_values)
    Energies = np.zeros((len(k_x_values), len(k_y_values), 4))
    for i, k_x in enumerate(k_x_values):
        for j, k_y in enumerate(k_y_values):
            Energies[i, j, :] = GetAnalyticEnergies(k_x, k_y, B, gamma, Lambda, phi_x, Delta, mu)
    
    for i in range(4):
        #ax.contour(X, Y, get_Effective_Analytic_Energy(X, Y, mu, B, Delta, phi_x, gamma, Lambda)[i],
         #          levels=[0.0],
          #         linestyles="dashed")
        ax.contour(Y, X, Energies[:,:, i], levels=[0.0], colors=["blue"])
    
    #ax.set_xlim(-2, -1)
    ax.set_xlabel("$k_x$")
    ax.set_ylabel("$k_y$");
    plt.show()