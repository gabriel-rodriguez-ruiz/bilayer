#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  5 10:24:33 2025

@author: gabriel
"""

import numpy as np
from scipy.integrate import dblquad
import numba
import matplotlib.pyplot as plt

class NegativeZDomainIntegrator:
    def __init__(self, coeff_funcs, use_jit=True):
        """
        coeff_funcs: list of 5 functions [a4(x,y), a3(x,y), a2(x,y), a1(x,y), a0(x,y)]
        """
        self.coeff_funcs = coeff_funcs
        
        if use_jit:
            # Compile the domain check for speed
            self._in_domain = numba.jit(nopython=True)(self._in_domain_numba)
            self._has_negative_real_roots = numba.jit(nopython=True)(self._has_negative_real_roots_numba)
        else:
            self._in_domain = self._in_domain_standard
    
    def _has_negative_real_roots_numba(self, a, b, c, d, e):
        """
        Check if quartic has real roots with z <= 0
        Uses a combination of discriminant analysis and interval testing
        """
        tol = 1e-12
        
        # Handle degenerate cases
        if abs(a) < tol:
            return self._degenerate_negative_roots_numba(a, b, c, d, e)
        
        # Method 1: Check polynomial at z=0 and negative z
        val_at_zero = e  # g(0) = e
        
        # If g(0) <= 0, there might be a root at z=0 or negative root
        if val_at_zero <= tol:
            return True
        
        # Method 2: Check for sign changes in negative region
        z_test_negative = np.linspace(-10, 0, 50)  # Focus on negative z
        
        prev_val = a*z_test_negative[0]**4 + b*z_test_negative[0]**3 + c*z_test_negative[0]**2 + d*z_test_negative[0] + e
        
        for z in z_test_negative[1:]:
            curr_val = a*z**4 + b*z**3 + c*z**2 + d*z + e
            
            # Sign change indicates a real root
            if prev_val * curr_val <= tol:
                return True
            
            # If we cross zero (within tolerance)
            if abs(curr_val) < 0.1:
                return True
                
            prev_val = curr_val
        
        # Method 3: Check derivative for local minima/maxima in negative region
        # that might indicate roots we missed
        return self._check_negative_extrema_numba(a, b, c, d, e)
    
    @staticmethod
    @numba.jit(nopython=True)
    def _degenerate_negative_roots_numba(a, b, c, d, e):
        """Handle cases where quartic degenerates to lower degree"""
        tol = 1e-12
        
        if abs(b) < tol:
            if abs(c) < tol:
                # Linear case: d*z + e = 0
                if abs(d) > tol:
                    root = -e/d
                    return root <= tol
                return False
            else:
                # Quadratic: c*z^2 + d*z + e = 0
                discriminant = d**2 - 4*c*e
                if discriminant < -tol:
                    return False
                elif abs(discriminant) < tol:
                    root = -d/(2*c)
                    return root <= tol
                else:
                    root1 = (-d - np.sqrt(discriminant)) / (2*c)
                    root2 = (-d + np.sqrt(discriminant)) / (2*c)
                    return root1 <= tol or root2 <= tol
        else:
            # Cubic: b*z^3 + c*z^2 + d*z + e = 0
            # Use sampling for cubic roots in negative region
            z_test = np.linspace(-10, 0, 100)
            prev_val = b*z_test[0]**3 + c*z_test[0]**2 + d*z_test[0] + e
            
            for z in z_test[1:]:
                curr_val = b*z**3 + c*z**2 + d*z + e
                if prev_val * curr_val <= tol:
                    return True
                prev_val = curr_val
            return False
    
    @staticmethod
    @numba.jit(nopython=True)
    def _check_negative_extrema_numba(a, b, c, d, e):
        """Check for local extrema in negative region that might indicate roots"""
        # Cubic derivative: 4a*z^3 + 3b*z^2 + 2c*z + d
        # Find critical points in negative region
        z_test = np.linspace(-10, 0, 100)
        
        for z in z_test:
            # Evaluate polynomial
            poly_val = a*z**4 + b*z**3 + c*z**2 + d*z + e
            
            # If value is small, we're near a root
            if abs(poly_val) < 0.5:
                return True
        
        return False
    
    def _in_domain_numba(self, x, y):
        """Check if (x,y) is in domain (has real roots with z <= 0)"""
        a = self.coeff_funcs[0](x, y)
        b = self.coeff_funcs[1](x, y)
        c = self.coeff_funcs[2](x, y)
        d = self.coeff_funcs[3](x, y)
        e = self.coeff_funcs[4](x, y)
        
        return self._has_negative_real_roots_numba(a, b, c, d, e)
    
    def _in_domain_standard(self, x, y):
        """Standard Python implementation"""
        a, b, c, d, e = [func(x, y) for func in self.coeff_funcs]
        return self._has_negative_real_roots_standard(a, b, c, d, e)
    
    def _has_negative_real_roots_standard(self, a, b, c, d, e):
        """
        More sophisticated analysis for negative real roots
        """
        tol = 1e-12
        
        # Check if there's a root exactly at z=0
        if abs(e) < tol:
            return True
        
        # For negative roots, we can use Budan-Fourier theorem or Sturm sequences
        # Here's a practical approach:
        
        # 1. Check polynomial behavior at z=0 and z→-∞
        val_at_zero = e
        # As z→-∞, sign is sign(a) if degree 4, etc.
        
        # 2. Use Descartes' rule of signs for negative roots
        # For negative roots, substitute z = -t and check sign changes
        coeffs_neg = [a, -b, c, -d, e]  # coefficients for g(-t)
        sign_changes = self._count_sign_changes(coeffs_neg)
        
        # Number of negative real roots is sign_changes minus an even number
        if sign_changes > 0:
            return True
        
        # 3. Fallback to sampling
        z_test = np.linspace(-100, 0, 1000)
        poly_vals = a*z_test**4 + b*z_test**3 + c*z_test**2 + d*z_test + e
        
        # Count sign changes
        sign_changes_sampling = np.sum(poly_vals[1:] * poly_vals[:-1] < 0)
        return sign_changes_sampling > 0
    
    @staticmethod
    def _count_sign_changes(coeffs):
        """Count sign changes in coefficient list"""
        coeffs = [c for c in coeffs if abs(c) > 1e-12]  # Remove near-zero coefficients
        if not coeffs:
            return 0
        
        sign_changes = 0
        current_sign = np.sign(coeffs[0])
        
        for coeff in coeffs[1:]:
            new_sign = np.sign(coeff)
            if new_sign != 0 and new_sign != current_sign:
                sign_changes += 1
                current_sign = new_sign
        
        return sign_changes
    
    def integrate(self, func, x_range, y_range, method='dblquad', **kwargs):
        """Integrate over the domain where quartic has real roots with z <= 0"""
        if method == 'dblquad':
            return self._integrate_dblquad(func, x_range, y_range, **kwargs)
        elif method == 'monte_carlo':
            return self._integrate_monte_carlo(func, x_range, y_range, **kwargs)
        elif method == 'quadrature':
            return self._integrate_quadrature(func, x_range, y_range, **kwargs)
        else:
            raise ValueError("Method must be 'dblquad', 'monte_carlo', or 'quadrature'")
    
    def _integrate_dblquad(self, func, x_range, y_range, tol=1e-6):
        """Adaptive double quadrature"""
        def integrand_wrapper(y, x):
            if self._in_domain(x, y):
                return func(x, y)
            return 0.0
        
        # Find tighter bounds for efficiency
        y_bounds = self._find_domain_bounds(x_range, y_range, 'y')
        if y_bounds[1] <= y_bounds[0]:
            return 0.0, 0.0
        
        result, error = dblquad(
            integrand_wrapper,
            y_bounds[0], y_bounds[1],
            lambda y: self._find_domain_bounds_for_y(y, x_range, 'lower'),
            lambda y: self._find_domain_bounds_for_y(y, x_range, 'upper'),
            epsabs=tol
        )
        
        return result, error
    
    def _integrate_monte_carlo(self, func, x_range, y_range, n_samples=100000):
        """Monte Carlo integration with z <= 0 constraint"""
        x_min, x_max = x_range
        y_min, y_max = y_range
        
        # Generate samples
        x_samples = np.random.uniform(x_min, x_max, n_samples)
        y_samples = np.random.uniform(y_min, y_max, n_samples)
        
        # Vectorized domain check
        in_domain_mask = np.zeros(n_samples, dtype=bool)
        for i in range(n_samples):
            in_domain_mask[i] = self._in_domain(x_samples[i], y_samples[i])
        
        n_in_domain = np.sum(in_domain_mask)
        
        if n_in_domain == 0:
            return 0.0, 0.0
        
        # Calculate function values only for points in domain
        func_vals = np.zeros(n_in_domain)
        j = 0
        for i in range(n_samples):
            if in_domain_mask[i]:
                func_vals[j] = func(x_samples[i], y_samples[i])
                j += 1
        
        # Monte Carlo estimate
        total_area = (x_max - x_min) * (y_max - y_min)
        integral = total_area * np.mean(func_vals)
        error = total_area * np.std(func_vals) / np.sqrt(n_in_domain)
        
        return integral, error
    
    def _integrate_quadrature(self, func, x_range, y_range, n_points=100):
        """Fixed quadrature integration"""
        x_min, x_max = x_range
        y_min, y_max = y_range
        
        # Create grid
        x_vals = np.linspace(x_min, x_max, n_points)
        y_vals = np.linspace(y_min, y_max, n_points)
        dx = (x_max - x_min) / (n_points - 1)
        dy = (y_max - y_min) / (n_points - 1)
        
        integral = 0.0
        
        for i, x in enumerate(x_vals):
            for j, y in enumerate(y_vals):
                if self._in_domain(x, y):
                    integral += func(x, y) * dx * dy
        
        # Rough error estimate
        error = dx * dy  # Basic error per cell
        
        return integral, error
    
    def _find_domain_bounds(self, x_range, y_range, axis='y', n_samples=100):
        """Find the actual bounds of the domain"""
        if axis == 'y':
            test_vals = np.linspace(y_range[0], y_range[1], n_samples)
        else:
            test_vals = np.linspace(x_range[0], x_range[1], n_samples)
        
        in_domain_vals = []
        
        for val in test_vals:
            if axis == 'y':
                # Check if any x for this y is in domain
                x_test = np.linspace(x_range[0], x_range[1], 20)
                for x in x_test:
                    if self._in_domain(x, val):
                        in_domain_vals.append(val)
                        break
            else:
                # Check if any y for this x is in domain
                y_test = np.linspace(y_range[0], y_range[1], 20)
                for y in y_test:
                    if self._in_domain(val, y):
                        in_domain_vals.append(val)
                        break
        
        if not in_domain_vals:
            return (0, 0)
        
        return min(in_domain_vals), max(in_domain_vals)
    
    def _find_domain_bounds_for_y(self, y, x_range, bound_type='lower', n_samples=50):
        """Find x-bounds for a given y value"""
        x_test = np.linspace(x_range[0], x_range[1], n_samples)
        domain_x = []
        
        for x in x_test:
            if self._in_domain(x, y):
                domain_x.append(x)
        
        if not domain_x:
            return x_range[0] if bound_type == 'lower' else x_range[1]
        
        if bound_type == 'lower':
            return min(domain_x)
        else:
            return max(domain_x)
    
    def visualize_domain_comparison(self, x_range, y_range, n_points=200):
        """Compare domains with and without z <= 0 constraint"""
        x = np.linspace(x_range[0], x_range[1], n_points)
        y = np.linspace(y_range[0], y_range[1], n_points)
        X, Y = np.meshgrid(x, y)
        
        # Domain with z <= 0 constraint
        Z_constrained = np.zeros_like(X)
        # Domain without constraint (any real root)
        Z_any = np.zeros_like(X)
        
        for i in range(n_points):
            for j in range(n_points):
                a, b, c, d, e = [f(X[i, j], Y[i, j]) for f in self.coeff_funcs]
                Z_constrained[i, j] = self._has_negative_real_roots_numba(a, b, c, d, e)
                
                # Check for any real root (simplified)
                z_test = np.linspace(-10, 10, 50)
                poly_vals = a*z_test**4 + b*z_test**3 + c*z_test**2 + d*z_test + e
                sign_changes = np.sum(poly_vals[1:] * poly_vals[:-1] < 0)
                Z_any[i, j] = sign_changes > 0
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot constrained domain
        im1 = ax1.contourf(X, Y, Z_constrained, levels=1, alpha=0.3)
        ax1.set_title('Domain: Real roots with z ≤ 0')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        plt.colorbar(im1, ax=ax1)
        
        # Plot unconstrained domain
        im2 = ax2.contourf(X, Y, Z_any, levels=1, alpha=0.3)
        ax2.set_title('Domain: Any real roots')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        plt.colorbar(im2, ax=ax2)
        
        plt.tight_layout()
        plt.show()
        
        # Calculate area ratio
        constrained_area = np.sum(Z_constrained) / (n_points * n_points) * np.prod([x_range[1]-x_range[0], y_range[1]-y_range[0]])
        any_area = np.sum(Z_any) / (n_points * n_points) * np.prod([x_range[1]-x_range[0], y_range[1]-y_range[0]])
        
        print(f"Constrained domain area: {constrained_area:.6f}")
        print(f"Unconstrained domain area: {any_area:.6f}")
        print(f"Ratio: {constrained_area/any_area:.3f}")

        
def compute_array(B, ky, kx, gamma, lambda_, phi_x, Delta, mu):
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
    
    # Fifth element
    elem5 = 1
    
    return np.array([elem1, elem2, elem3, elem4, elem5])

# Example usage:
# result = compute_array(B=1.0, ky=0.5, kx=0.3, gamma=0.1, lambda_=0.2, phi_x=0.4, Delta=0.6, mu=0.7)
# print(result)
# Example: Quartic that has negative roots in specific regions

B = 2
lambda_ = 1
phi_x = 1
mu = 3
gamma = 1
Delta = 1

def a4(x, y): 
    return compute_array(B, y, x, gamma, lambda_, phi_x, Delta, mu)[4]
def a3(x, y): 
    return compute_array(B, y, x, gamma, lambda_, phi_x, Delta, mu)[3]
def a2(x, y): 
    return compute_array(B, y, x, gamma, lambda_, phi_x, Delta, mu)[2]
def a1(x, y): 
    return compute_array(B, y, x, gamma, lambda_, phi_x, Delta, mu)[1]
def a0(x, y): 
    return compute_array(B, y, x, gamma, lambda_, phi_x, Delta, mu)[0]

def my_integrand(x, y):
    return np.exp(-(x**2 + y**2)/2)

# Create integrator
coeff_funcs = [a4, a3, a2, a1, a0]
integrator = NegativeZDomainIntegrator(coeff_funcs, use_jit=False)

# Define integration region
x_range = (-2, 2)
y_range = (-2, 2)

# Visualize the domain comparison
integrator.visualize_domain_comparison(x_range, y_range)

# Perform integration
print("Integration methods comparison:")

# Method 1: Adaptive
result1, error1 = integrator.integrate(my_integrand, x_range, y_range, 
                                     method='dblquad', tol=1e-6)
print(f"Adaptive: {result1:.8f} ± {error1:.2e}")

# Method 2: Monte Carlo
result2, error2 = integrator.integrate(my_integrand, x_range, y_range,
                                     method='monte_carlo', n_samples=500000)
print(f"Monte Carlo: {result2:.8f} ± {error2:.2e}")

# Method 3: Quadrature
result3, error3 = integrator.integrate(my_integrand, x_range, y_range,
                                     method='quadrature', n_points=200)
print(f"Quadrature: {result3:.8f} ± {error3:.2e}")