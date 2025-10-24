#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 11 13:26:53 2025

@author: gabriel
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.linalg import eigh
from scipy.interpolate import CubicSpline
from pauli_matrices import tau_0, tau_z, sigma_0, tau_x, sigma_z, sigma_x, sigma_y

def continuous_eigenvalues(H_func, param_range, n_points=100, 
                          continuity_weight=0.5, lookback=3, 
                          smoothness_constraint=True):
    """
    Compute eigenvalues with enforced continuity using multiple constraints.
    
    Parameters:
    -----------
    H_func : function
        Function that takes a parameter and returns a Hermitian matrix
    param_range : tuple
        (start, end) range of the parameter
    n_points : int
        Number of points to sample
    continuity_weight : float
        Weight for eigenvalue continuity vs eigenvector similarity (0-1)
    lookback : int
        Number of previous points to consider for continuity
    smoothness_constraint : bool
        Whether to apply smoothness constraints through interpolation
    
    Returns:
    --------
    params : array
        Parameter values
    eigenvalues : array (n_points x n)
        Continuous eigenvalues
    eigenvectors : array (n_points x n x n)
        Corresponding eigenvectors
    """
    
    params = np.linspace(param_range[0], param_range[1], n_points)
    
    # Initial computation
    H0 = H_func(params[0])
    n = H0.shape[0]
    
    eigenvalues = np.zeros((n_points, n))
    eigenvectors = np.zeros((n_points, n, n), dtype=complex)
    
    # Compute first point
    eigvals, eigvecs = eigh(H_func(params[0]))
    eigenvalues[0] = eigvals
    eigenvectors[0] = eigvecs
    
    # Compute second point with simple assignment
    H1 = H_func(params[1])
    eigvals1, eigvecs1 = eigh(H1)
    overlap = np.abs(eigvecs1.conj().T @ eigenvectors[0])
    cost_matrix = 1 - overlap
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    eigenvalues[1, col_ind] = eigvals1[row_ind]
    eigenvectors[1, :, col_ind] = eigvecs1[:, row_ind]
    
    # Process remaining points with continuity constraints
    for i in range(2, n_points):
        H = H_func(params[i])
        eigvals, eigvecs = eigh(H)
        
        # Enhanced cost matrix with multiple continuity constraints
        cost_matrix = build_continuity_cost_matrix(
            eigvals, eigvecs, eigenvalues, eigenvectors, i, 
            lookback, continuity_weight
        )
        
        # Solve assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Apply smoothness constraint if requested
        if smoothness_constraint and i > 2:
            col_ind = apply_smoothness_constraint(
                eigvals, eigenvalues, i, row_ind, col_ind
            )
        
        eigenvalues[i, col_ind] = eigvals[row_ind]
        eigenvectors[i, :, col_ind] = eigvecs[:, row_ind]
    
    return params, eigenvalues, eigenvectors

def build_continuity_cost_matrix(current_eigvals, current_eigvecs, 
                               all_eigenvalues, all_eigenvectors, 
                               current_idx, lookback, continuity_weight):
    """
    Build cost matrix considering both eigenvector similarity and eigenvalue continuity.
    """
    n = current_eigvals.shape[0]
    cost_matrix = np.zeros((n, n))
    
    # Consider multiple previous points for better continuity
    start_idx = max(0, current_idx - lookback)
    
    for j in range(start_idx, current_idx):
        weight = 1.0 / (current_idx - j)  # Recent points have higher weight
        
        # Eigenvector overlap cost
        overlap = np.abs(current_eigvecs.conj().T @ all_eigenvectors[j])
        eigenvector_cost = 1 - overlap
        
        # Eigenvalue continuity cost (normalized)
        prev_eigvals = all_eigenvalues[j]
        eigenvalue_diff = np.abs(current_eigvals[:, None] - prev_eigvals)
        max_diff = np.max(eigenvalue_diff) if np.max(eigenvalue_diff) > 0 else 1.0
        eigenvalue_cost = eigenvalue_diff / max_diff
        
        # Combined cost
        combined_cost = (1 - continuity_weight) * eigenvector_cost + \
                       continuity_weight * eigenvalue_cost
        
        cost_matrix += weight * combined_cost
    
    return cost_matrix

def apply_smoothness_constraint(current_eigvals, all_eigenvalues, 
                              current_idx, row_ind, col_ind):
    """
    Apply additional smoothness constraint using eigenvalue trajectory.
    """
    n = current_eigvals.shape[0]
    
    # Predict eigenvalues using quadratic extrapolation
    predicted_eigvals = predict_eigenvalues(all_eigenvalues, current_idx, n)
    
    # Compute deviation from predicted values
    deviations = []
    for i in range(n):
        assigned_eigval = current_eigvals[row_ind[i]]
        predicted_val = predicted_eigvals[col_ind[i]]
        deviation = np.abs(assigned_eigval - predicted_val)
        deviations.append(deviation)
    
    deviations = np.array(deviations)
    
    # If any deviation is too large, consider reordering
    max_deviation = np.max(deviations)
    if max_deviation > 2.0 * np.mean(deviations):
        # Use Hungarian algorithm with smoothness constraint
        smoothness_cost = np.abs(current_eigvals[:, None] - predicted_eigvals)
        max_smooth_cost = np.max(smoothness_cost) if np.max(smoothness_cost) > 0 else 1.0
        smoothness_cost = smoothness_cost / max_smooth_cost
        
        # Combine with original assignment cost
        final_cost = 0.7 * smoothness_cost + 0.3 * (deviations[:, None] / max_deviation)
        
        row_ind, col_ind = linear_sum_assignment(final_cost)
    
    return col_ind

def predict_eigenvalues(all_eigenvalues, current_idx, n):
    """
    Predict eigenvalues using polynomial extrapolation.
    """
    predicted = np.zeros(n)
    
    for j in range(n):
        # Use last 3 points for quadratic prediction
        start_idx = max(0, current_idx - 3)
        x = np.arange(start_idx, current_idx)
        y = all_eigenvalues[start_idx:current_idx, j]
        
        if len(x) >= 2:
            # Linear or quadratic extrapolation
            if len(x) == 2:
                # Linear
                coeffs = np.polyfit(x, y, 1)
            else:
                # Quadratic
                coeffs = np.polyfit(x, y, 2)
            predicted[j] = np.polyval(coeffs, current_idx)
        else:
            predicted[j] = all_eigenvalues[current_idx-1, j]
    
    return predicted

def continuous_eigenvalues_with_restarts(H_func, param_range, n_points=100, 
                                        max_restarts=5, continuity_threshold=0.1):
    """
    Version with restart capability to handle difficult crossings.
    """
    params = np.linspace(param_range[0], param_range[1], n_points)
    
    # Initial computation
    H0 = H_func(params[0])
    n = H0.shape[0]
    
    eigenvalues = np.zeros((n_points, n))
    eigenvectors = np.zeros((n_points, n, n), dtype=complex)
    
    # Compute first point
    eigvals, eigvecs = eigh(H_func(params[0]))
    eigenvalues[0] = eigvals
    eigenvectors[0] = eigvecs
    
    restarts = 0
    i = 1
    
    while i < n_points and restarts < max_restarts:
        try:
            H = H_func(params[i])
            eigvals, eigvecs = eigh(H)
            
            # Enhanced continuity check
            if i > 0:
                continuity_ok = check_continuity(
                    eigvals, eigvecs, eigenvalues[i-1], eigenvectors[i-1], 
                    continuity_threshold
                )
                
                if not continuity_ok and i > 1:
                    # Potential crossing detected, use more sophisticated method
                    col_ind = handle_crossing(
                        eigvals, eigvecs, eigenvalues, eigenvectors, i
                    )
                else:
                    # Normal assignment
                    overlap = np.abs(eigvecs.conj().T @ eigenvectors[i-1])
                    cost_matrix = 1 - overlap
                    row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            eigenvalues[i, col_ind] = eigvals[row_ind]
            eigenvectors[i, :, col_ind] = eigvecs[:, row_ind]
            i += 1
            
        except Exception as e:
            print(f"Restarting at point {i} due to: {e}")
            restarts += 1
            # Backtrack and try with different parameters
            i = max(1, i - 2)
    
    return params, eigenvalues, eigenvectors

def check_continuity(current_eigvals, current_eigvecs, prev_eigvals, prev_eigvecs, threshold):
    """
    Check if the eigenvalue-eigenvector pair maintains continuity.
    """
    overlap = np.abs(current_eigvecs.conj().T @ prev_eigvecs)
    max_overlap = np.max(overlap, axis=1)
    
    eigenvalue_diff = np.abs(current_eigvals - prev_eigvals)
    max_eig_diff = np.max(eigenvalue_diff)
    
    # Continuity is broken if overlaps are low AND eigenvalues change abruptly
    return np.min(max_overlap) > threshold or max_eig_diff < 2.0 * np.mean(eigenvalue_diff)

def handle_crossing(current_eigvals, current_eigvecs, all_eigenvalues, all_eigenvectors, idx):
    """
    Special handling for eigenvalue crossings.
    """
    n = current_eigvals.shape[0]
    
    # Use multiple previous points to determine the correct branch
    lookback = min(3, idx)
    cost_matrices = []
    
    for j in range(idx - lookback, idx):
        overlap = np.abs(current_eigvecs.conj().T @ all_eigenvectors[j])
        eigenvector_cost = 1 - overlap
        
        eigenvalue_diff = np.abs(current_eigvals[:, None] - all_eigenvalues[j])
        max_diff = np.max(eigenvalue_diff) if np.max(eigenvalue_diff) > 0 else 1.0
        eigenvalue_cost = eigenvalue_diff / max_diff
        
        # Weight recent points more heavily
        weight = 1.0 / (idx - j)
        combined_cost = 0.3 * eigenvector_cost + 0.7 * eigenvalue_cost
        cost_matrices.append(weight * combined_cost)
    
    # Average cost matrix
    avg_cost = np.mean(cost_matrices, axis=0)
    row_ind, col_ind = linear_sum_assignment(avg_cost)
    
    return col_ind

# Demonstration with various test cases
def test_continuous_eigenvalues():
    """Test the continuous eigenvalue computation with various systems."""
    
    # Test case 1: Simple crossing
    def crossing_hamiltonian(t):
        return np.array([[t, 0.5], [0.5, -t]], dtype=complex)
    
    # Test case 2: Avoided crossing
    def avoided_crossing(t):
        gap = 0.2
        return np.array([[t, gap], [gap, -t]], dtype=complex)
    
    # Test case 3: Three-level system with multiple crossings
    def three_level_crossing(t):
        return np.array([
            [t, 0.3, 0.1],
            [0.3, t-1, 0.2],
            [0.1, 0.2, -2*t]
        ], dtype=complex)
    
    test_cases = [
        ("Simple Crossing", crossing_hamiltonian, (-2, 2)),
        ("Avoided Crossing", avoided_crossing, (-2, 2)),
        ("Three-Level System", three_level_crossing, (-2, 2))
    ]
    
    for name, H_func, param_range in test_cases:
        print(f"\nTesting {name}:")
        
        # Compute with continuity
        params, eigenvalues, eigenvectors = continuous_eigenvalues(
            H_func, param_range, n_points=100, 
            continuity_weight=0.6, lookback=3, smoothness_constraint=True
        )
        
        # Plot results
        plt.figure(figsize=(10, 6))
        for i in range(eigenvalues.shape[1]):
            plt.plot(params, eigenvalues[:, i], 'o-', markersize=2, linewidth=1.5, 
                    label=f'Level {i+1}')
        
        plt.xlabel('Parameter')
        plt.ylabel('Eigenvalue')
        plt.title(f'{name} - Continuous Eigenvalues')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # Check continuity by computing derivatives
        derivatives = np.gradient(eigenvalues, params, axis=0)
        max_derivative_change = np.max(np.abs(np.gradient(derivatives, params, axis=0)))
        print(f"Maximum derivative change: {max_derivative_change:.6f}")

def analyze_continuity(eigenvalues, params):
    """
    Analyze the continuity of eigenvalue trajectories.
    """
    n_levels = eigenvalues.shape[1]
    
    plt.figure(figsize=(12, 4))
    
    # Plot eigenvalues
    plt.subplot(1, 2, 1)
    for i in range(n_levels):
        plt.plot(params, eigenvalues[:, i], 'o-', markersize=2, label=f'Level {i+1}')
    plt.xlabel('Parameter')
    plt.ylabel('Eigenvalue')
    plt.title('Eigenvalue Trajectories')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot derivatives to check smoothness
    plt.subplot(1, 2, 2)
    derivatives = np.gradient(eigenvalues, params, axis=0)
    for i in range(n_levels):
        plt.plot(params, derivatives[:, i], '-', alpha=0.7, label=f'dE_{i+1}/dt')
    plt.xlabel('Parameter')
    plt.ylabel('Derivative')
    plt.title('Eigenvalue Derivatives (Smoothness Check)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Quantitative continuity measures
    second_derivatives = np.gradient(derivatives, params, axis=0)
    continuity_measures = {
        'max_derivative': np.max(np.abs(derivatives)),
        'mean_derivative': np.mean(np.abs(derivatives)),
        'max_second_derivative': np.max(np.abs(second_derivatives)),
        'mean_second_derivative': np.mean(np.abs(second_derivatives))
    }
    
    print("Continuity Analysis:")
    for key, value in continuity_measures.items():
        print(f"{key}: {value:.6f}")

if __name__ == "__main__":
    print("Testing continuous eigenvalue computation...")
    
    # Run test cases
    test_continuous_eigenvalues()
    c = 3e18 # nm/s  #3e9 # m/s
    m_e =  5.1e8 / c**2 # meV s²/m²
    m = 0.403 * m_e # meV s²/m²
    hbar = 6.58e-13 # meV s
    gamma = hbar**2 / (2*m) # meV (nm)²
    E_F = 50.6 # meV
    k_F = np.sqrt(E_F / gamma ) # 1/nm
    
    def get_Hamiltonian(k_x, k_y, mu, B, Delta, phi_x, gamma, Lambda):
        """Return the Hamiltonian for a given k."""
        chi_k_plus = gamma * ( (k_x + phi_x)**2 + k_y**2) - mu
        chi_k_minus = gamma * ( (-k_x + phi_x)**2 + k_y**2 ) - mu
        return 1/2 * ( chi_k_plus * np.kron( ( tau_0 + tau_z )/2, sigma_0)
                       - chi_k_minus * np.kron( ( tau_0 - tau_z )/2, sigma_0)
                       - B * np.kron(tau_0, sigma_y)
                       - Delta * np.kron(tau_x, sigma_0)
                       + Lambda * (k_x + phi_x) * np.kron( ( tau_0 + tau_z )/2, sigma_y )
                       + Lambda * (-k_x + phi_x) * np.kron( ( tau_0 - tau_z )/2, sigma_y )
                       - Lambda * k_y * np.kron( tau_z, sigma_x )
                     )
    # Detailed analysis of one case
    def demo_hamiltonian(t):
        Delta = 0.08 # meV
        Delta = 0.08   #  meV
        mu = 632 * Delta   #50.6  #  meV
        B = 2*Delta  # 2 * Delta
        phi_x = 0  # 0.002 * k_F
        gamma = 9479 # meV (nm)²
        Lambda = 8 * Delta # 8 * Delta  #0.644 meV 
        return get_Hamiltonian(t, 0, mu, B, Delta, phi_x, gamma, Lambda)
    
    params, eigenvalues, eigenvectors = continuous_eigenvalues(
        demo_hamiltonian, (-1.01*k_F, -0.99*k_F), continuity_weight=0.9 ,
        lookback=4, smoothness_constraint=True, n_points=1000
    )
    
    analyze_continuity(eigenvalues, params)