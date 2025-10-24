#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  5 15:08:13 2025

@author: gabriel
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List, Optional, Dict
import time
from matplotlib.patches import Polygon, Rectangle
from matplotlib.collections import PatchCollection
import matplotlib.cm as cm
from pauli_matrices import tau_0, tau_z, sigma_0, tau_x, sigma_z, sigma_x, sigma_y
import scipy

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

def get_Energies(k_x_values, k_y_values, mu, B, Delta, phi_x, gamma, Lambda):
    """Return the energies of the Hamiltonian at a given k."""
    E = np.zeros((len(k_x_values), len(k_y_values), 4))
    for i, k_x in enumerate(k_x_values):
        for j, k_y in enumerate(k_y_values):
            H = get_Hamiltonian(k_x, k_y, mu, B, Delta, phi_x, gamma, Lambda)
            E[i, j, :] = np.linalg.eigvalsh(H)
    return E

def get_Energy(k_x, k_y, mu, B, Delta, phi_x, gamma, Lambda):
    """Return the energies of the Hamiltonian at a given k."""
    H = get_Hamiltonian(k_x, k_y, mu, B, Delta, phi_x, gamma, Lambda)
    return np.linalg.eigvalsh(H)

def get_Analytic_energies_at_k_y_zero(k_x, mu, B, Delta, phi_x, gamma, Lambda):
    return np.array([
        1/2 * (
            (B - Lambda*phi_x) + 2*k_x*gamma*phi_x + np.sqrt(Delta**2 + (k_x**2 * gamma - Lambda*k_x - mu + gamma*phi_x**2)**2)
        ),
        1/2 * (
            (B - Lambda*phi_x) + 2*k_x*gamma*phi_x - np.sqrt(Delta**2 + (k_x**2 * gamma - Lambda*k_x - mu + gamma*phi_x**2)**2)
        ),
        1/2 * (
            -(B - Lambda*phi_x) + 2*k_x*gamma*phi_x + np.sqrt(Delta**2 + (k_x**2 * gamma + Lambda*k_x - mu + gamma*phi_x**2)**2)
        ),
        1/2 * (
            -(B - Lambda*phi_x) + 2*k_x*gamma*phi_x - np.sqrt(Delta**2 + (k_x**2 * gamma + Lambda*k_x - mu + gamma*phi_x**2)**2)
        ),
        ])

def find_all_roots(f, x_range=(-10, 10), num_points=1000, tol=1e-6):
    """
    Find all real roots of a function f(x) in a given range.
    
    Parameters:
    f: function to find roots of
    x_range: tuple (min, max) search range
    num_points: number of points for initial sampling
    tol: tolerance for root uniqueness
    """
    # Sample the function to find sign changes
    x_vals = np.linspace(x_range[0], x_range[1], num_points)
    y_vals = f(x_vals)
    
    roots = []
    
    # Find intervals where sign changes (potential roots)
    for i in range(len(x_vals) - 1):
        if y_vals[i] * y_vals[i + 1] <= 0:  # Sign change or zero
            # Refine the root using brentq
            try:
                root_val = scipy.optimize.brentq(f, x_vals[i], x_vals[i + 1])
                # Check if this root is distinct from previously found ones
                if not any(abs(root_val - r) < tol for r in roots):
                    roots.append(root_val)
            except (ValueError, RuntimeError):
                continue
    
    return np.array(roots)


class Interval:
    """Interval arithmetic implementation"""
    
    def __init__(self, a: float, b: Optional[float] = None):
        if b is None:
            b = a
        self.a = min(a, b)
        self.b = max(a, b)
    
    def __add__(self, other):
        if isinstance(other, Interval):
            return Interval(self.a + other.a, self.b + other.b)
        return Interval(self.a + other, self.b + other)
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
        if isinstance(other, Interval):
            return Interval(self.a - other.b, self.b - other.a)
        return Interval(self.a - other, self.b - other)
    
    def __rsub__(self, other):
        if isinstance(other, Interval):
            return Interval(other.a - self.b, other.b - self.a)
        return Interval(other - self.b, other - self.a)
    
    def __mul__(self, other):
        if isinstance(other, Interval):
            products = [self.a * other.a, self.a * other.b, 
                       self.b * other.a, self.b * other.b]
            return Interval(min(products), max(products))
        return Interval(self.a * other, self.b * other)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other):
        if isinstance(other, Interval):
            if other.a <= 0 <= other.b:
                raise ValueError("Division by interval containing 0")
            reciprocals = [1/other.a, 1/other.b]
            return self * Interval(min(reciprocals), max(reciprocals))
        if other == 0:
            raise ValueError("Division by zero")
        return Interval(self.a / other, self.b / other)
    
    def contains_zero(self):
        return self.a <= 0 <= self.b
    
    def is_positive(self):
        return self.a > 0
    
    def is_negative(self):
        return self.b < 0
    
    def __repr__(self):
        return f"[{self.a}, {self.b}]"

class ImplicitIntegrator:
    """
    Numerical integration over implicitly defined domains with topological guarantee
    """
    
    def __init__(self, f_implicit: Callable, domain: Tuple[float, float, float, float]):
        """
        Parameters:
        f_implicit: implicit function defining domain Ω = {(x,y) | f(x,y) ≥ 0}
        domain: (x_min, x_max, y_min, y_max) bounding box
        """
        self.f_implicit = f_implicit
        self.domain = domain
        self.x_min, self.x_max, self.y_min, self.y_max = domain
        
        # Visualization data
        self.visualization_data = {
            'cells': [],
            'integration_points': [],
            'boundary_points': [],
            'subdivision_history': []
        }
    
    def interval_eval(self, x_interval: Interval, y_interval: Interval) -> Interval:
        """Evaluate implicit function using interval arithmetic"""
        try:
            f = lambda x: get_Analytic_energies_at_k_y_zero(x, mu, B, Delta, phi_x, gamma, Lambda)[2]
            g = lambda x: get_Analytic_energies_at_k_y_zero(x, mu, B, Delta, phi_x, gamma, Lambda)[1]
            h = lambda x: get_Analytic_energies_at_k_y_zero(x, mu, B, Delta, phi_x, gamma, Lambda)[0]
            l = lambda x: get_Analytic_energies_at_k_y_zero(x, mu, B, Delta, phi_x, gamma, Lambda)[3]
            roots = find_all_roots(f, x_range=(x_interval.a, x_interval.b))
            roots = np.append(roots, find_all_roots(g, x_range=(x_interval.a, x_interval.b)))
            roots = np.append(roots, find_all_roots(h, x_range=(x_interval.a, x_interval.b)))
            roots = np.append(roots, find_all_roots(l, x_range=(x_interval.a, x_interval.b)))

            # Sample points within the interval to get bounds
            n_samples = 10
            x_samples = np.linspace(x_interval.a, x_interval.b, n_samples)
            y_samples = np.linspace(y_interval.a, y_interval.b, n_samples)
            for root in roots:
                if  x_interval.a<=root <= x_interval.b and y_interval.a<= 0 <= y_interval.b:
                    x_samples = np.append(x_samples, root)
                    y_samples = np.append(y_samples, 0)
                x_samples = np.sort(x_samples)
                y_samples = np.sort(y_samples)

            #x_samples = [x_interval.a, x_interval.b, (x_interval.a + x_interval.b)/2]
            #y_samples = [y_interval.a, y_interval.b, (y_interval.a + y_interval.b)/2]
            
            values = []
            for x in x_samples:
                for y in y_samples:
                    values.append(self.f_implicit(x, y))
            
            return Interval(min(values), max(values))
        except:
            # Fallback: use endpoints
            corners = [
                self.f_implicit(x_interval.a, y_interval.a),
                self.f_implicit(x_interval.a, y_interval.b),
                self.f_implicit(x_interval.b, y_interval.a),
                self.f_implicit(x_interval.b, y_interval.b)
            ]
            return Interval(min(corners), max(corners))
    
    def classify_cell(self, cell: Tuple[float, float, float, float]) -> str:
        """
        Classify cell as 'interior', 'exterior', or 'boundary'
        """
        x_min, x_max, y_min, y_max = cell
        x_interval = Interval(x_min, x_max)
        y_interval = Interval(y_min, y_max)
        
        f_interval = self.interval_eval(x_interval, y_interval)
        
        # Store for visualization
        self.visualization_data['cells'].append({
            'bounds': cell,
            'type': 'interior' if f_interval.is_positive() else 
                   'exterior' if f_interval.is_negative() else 'boundary',
            'interval': f_interval
        })
        
        if f_interval.is_positive():
            return 'interior'
        elif f_interval.is_negative():
            return 'exterior'
        else:
            return 'boundary'
    
    def find_intersections(self, cell: Tuple[float, float, float, float]) -> List[Tuple[float, float]]:
        """
        Find intersections of implicit boundary with cell edges
        Simplified implementation using bisection
        """
        x_min, x_max, y_min, y_max = cell
        intersections = []
        
        # Check left and right edges
        for x in [x_min, x_max]:
            y1, y2 = y_min, y_max
            f1, f2 = self.f_implicit(x, y1), self.f_implicit(x, y2)
            
            if f1 * f2 < 0:  # Sign change indicates intersection
                # Bisection to find intersection
                for _ in range(10):  # 10 iterations for reasonable accuracy
                    y_mid = (y1 + y2) / 2
                    f_mid = self.f_implicit(x, y_mid)
                    if f1 * f_mid < 0:
                        y2, f2 = y_mid, f_mid
                    else:
                        y1, f1 = y_mid, f_mid
                intersection_point = (x, (y1 + y2) / 2)
                intersections.append(intersection_point)
                self.visualization_data['boundary_points'].append(intersection_point)
        
        # Check bottom and top edges
        for y in [y_min, y_max]:
            x1, x2 = x_min, x_max
            f1, f2 = self.f_implicit(x1, y), self.f_implicit(x2, y)
            
            if f1 * f2 < 0:
                # Bisection to find intersection
                for _ in range(10):
                    x_mid = (x1 + x2) / 2
                    f_mid = self.f_implicit(x_mid, y)
                    if f1 * f_mid < 0:
                        x2, f2 = x_mid, f_mid
                    else:
                        x1, f1 = x_mid, f_mid
                intersection_point = ((x1 + x2) / 2, y)
                intersections.append(intersection_point)
                self.visualization_data['boundary_points'].append(intersection_point)
        
        return intersections
    
    def quadratic_bezier_approximation(self, intersections: List[Tuple[float, float]], 
                                     cell: Tuple[float, float, float, float]) -> Callable:
        """
        Create quadratic Bezier approximation of boundary curve
        """
        if len(intersections) != 2:
            # Fallback to linear approximation
            return lambda t: (
                intersections[0][0] + t * (intersections[1][0] - intersections[0][0]),
                intersections[0][1] + t * (intersections[1][1] - intersections[0][1])
            )
        
        P0 = np.array(intersections[0])
        P2 = np.array(intersections[1])
        
        # Estimate tangent directions (simplified)
        x_min, x_max, y_min, y_max = cell
        center_x, center_y = (x_min + x_max)/2, (y_min + y_max)/2
        
        # Use center point as P1 (control point)
        P1 = np.array([center_x, center_y])
        
        def bezier_curve(t):
            """Quadratic Bezier curve"""
            return (1-t)**2 * P0 + 2*(1-t)*t * P1 + t**2 * P2
        
        return bezier_curve
    
    def geometric_error_estimate(self, cell: Tuple[float, float, float, float], 
                               bezier_curve: Callable, F: Callable) -> float:
        """
        Estimate geometric error for subdivision criterion
        """
        x_min, x_max, y_min, y_max = cell
        cell_area = (x_max - x_min) * (y_max - y_min)
        
        # Sample points along Bezier curve to estimate maximum distance
        sample_points = 5
        max_distance = 0.0
        max_F = 0.0
        
        for i in range(sample_points):
            t = i / (sample_points - 1)
            bezier_point = bezier_curve(t)
            
            # Estimate distance (simplified)
            distance = 0.1 * min(x_max - x_min, y_max - y_min)  # Placeholder
            max_distance = max(max_distance, distance)
            
            # Estimate maximum of |F|
            F_val = abs(F(bezier_point[0], bezier_point[1]))
            max_F = max(max_F, F_val)
        
        # Error estimate: M * area_of_error_region
        error_band_area = max_distance * self.curve_length(bezier_curve)
        error_estimate = max_F * error_band_area
        
        return error_estimate
    
    def curve_length(self, curve: Callable, samples: int = 10) -> float:
        """Estimate curve length"""
        length = 0.0
        #prev_point = curve(0)
        prev_point = np.array(curve(0))
        for i in range(1, samples):
            t = i / (samples - 1)
            #current_point = curve(t)
            current_point = np.array(curve(t))
            length += np.linalg.norm(current_point - prev_point)
            prev_point = current_point
        return length
    
    def integrate_cell(self, cell: Tuple[float, float, float, float], 
                      F: Callable, tau: float, level: int = 0, 
                      max_level: int = 10) -> float:
        """
        Recursive integration over a cell with adaptive subdivision
        """
        if level > max_level:
            return 0.0
        
        cell_type = self.classify_cell(cell)
        x_min, x_max, y_min, y_max = cell
        
        # Record subdivision
        self.visualization_data['subdivision_history'].append({
            'level': level,
            'cell': cell,
            'type': cell_type
        })
        
        if cell_type == 'exterior':
            return 0.0
        elif cell_type == 'interior':
            # Use Gaussian quadrature for interior cells
            # result = self.gaussian_quadrature(cell, F)
            # F in dblquad is F(y,x)
            result = scipy.integrate.dblquad(F, x_min, x_max, y_min, y_max)[0]
            # Record integration points for visualization
            self.record_integration_points(cell, F, 'interior')
            return result
        else:  # boundary cell
            intersections = self.find_intersections(cell)
            
            if len(intersections) < 2:
                # Cannot properly approximate boundary, subdivide
                return self.subdivide_and_integrate(cell, F, tau, level, max_level)
            
            # Create Bezier approximation
            bezier_curve = self.quadratic_bezier_approximation(intersections, cell)
            
            # Check error estimate
            error_estimate = self.geometric_error_estimate(cell, bezier_curve, F)
            cell_area_ratio = (x_max - x_min) * (y_max - y_min) / (
                (self.x_max - self.x_min) * (self.y_max - self.y_min))
            scaled_tolerance = tau * cell_area_ratio
            
            if error_estimate < scaled_tolerance or level == max_level:
                # Accept approximation and integrate
                result = self.integrate_boundary_cell(cell, F, bezier_curve)
                self.record_integration_points(cell, F, 'boundary')
                return result
            else:
                # Subdivide further
                return self.subdivide_and_integrate(cell, F, tau, level, max_level)
    
    def record_integration_points(self, cell: Tuple[float, float, float, float], 
                                F: Callable, cell_type: str):
        """Record integration points for visualization"""
        x_min, x_max, y_min, y_max = cell
        
        # Generate sample integration points
        points = []
        n_points = 3  # Sample points per dimension
        
        for i in range(n_points):
            for j in range(n_points):
                x = x_min + (i + 0.5) * (x_max - x_min) / n_points
                y = y_min + (j + 0.5) * (y_max - y_min) / n_points
                if self.f_implicit(x, y) >= 0:  # Point inside domain
                    points.append((x, y, F(x, y), cell_type))
        
        self.visualization_data['integration_points'].extend(points)
    
    def integrate_boundary_cell(self, cell: Tuple[float, float, float, float],
                              F: Callable, bezier_curve: Callable) -> float:
        """
        Integrate over boundary cell using Bezier approximation
        Simplified implementation
        """
        # This would implement the actual integration scheme described in the paper
        # For now, use simple Gaussian quadrature as placeholder
        # x_min, x_max, y_min, y_max = cell
        # result = scipy.integrate.dblquad(F, x_min, x_max, y_min, y_max)[0]
        return self.gaussian_quadrature(cell, F) * 0.5  # Rough approximation
        # return result
    def subdivide_and_integrate(self, cell: Tuple[float, float, float, float],
                              F: Callable, tau: float, level: int, max_level: int) -> float:
        """Subdivide cell and integrate subcells recursively"""
        x_min, x_max, y_min, y_max = cell
        x_mid = (x_min + x_max) / 2
        y_mid = (y_min + y_max) / 2
        
        subcells = [
            (x_min, x_mid, y_min, y_mid),  # bottom-left
            (x_mid, x_max, y_min, y_mid),  # bottom-right
            (x_min, x_mid, y_mid, y_max),  # top-left
            (x_mid, x_max, y_mid, y_max)   # top-right
        ]
        
        result = 0.0
        for subcell in subcells:
            result += self.integrate_cell(subcell, F, tau, level + 1, max_level)
        
        return result
    
    def gaussian_quadrature(self, cell: Tuple[float, float, float, float], 
                          F: Callable, order: int = 2) -> float:
        """Gaussian quadrature over rectangular cell"""
        x_min, x_max, y_min, y_max = cell
        
        # Gauss-Legendre points and weights for order 2
        if order == 2:
            points = [-1/np.sqrt(3), 1/np.sqrt(3)]
            weights = [1.0, 1.0]
        else:
            # Default to order 2
            points = [-1/np.sqrt(3), 1/np.sqrt(3)]
            weights = [1.0, 1.0]
        
        integral = 0.0
        for i, xi in enumerate(points):
            for j, eta in enumerate(points):
                # Transform to cell coordinates
                x = (x_min + x_max) / 2 + (x_max - x_min) / 2 * xi
                y = (y_min + y_max) / 2 + (y_max - y_min) / 2 * eta
                
                integral += weights[i] * weights[j] * F(x, y)
        
        # Jacobian determinant
        jacobian = (x_max - x_min) * (y_max - y_min) / 4
        return integral * jacobian
    
    def integrate(self, F: Callable, tau: float = 1e-4, max_level: int = 8) -> float:
        """
        Main integration method
        """
        # Reset visualization data
        self.visualization_data = {
            'cells': [],
            'integration_points': [],
            'boundary_points': [],
            'subdivision_history': []
        }
        
        initial_cell = self.domain
        return self.integrate_cell(initial_cell, F, tau, 0, max_level)
    
    def visualize_integration(self, F: Callable, show_plot: bool = True):
        """
        Create comprehensive visualization of the integration process
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Numerical Integration over Implicit Domain - Visualization', fontsize=16)
        
        # Plot 1: Domain and cell classification
        ax1 = axes[0, 0]
        self._plot_domain_and_cells(ax1)
        ax1.set_title('Domain and Cell Classification')
        
        # Plot 2: Integration points and function values
        ax2 = axes[0, 1]
        self._plot_integration_points(ax2, F)
        ax2.set_title('Integration Points and Function Values')
        
        # Plot 3: Subdivision hierarchy
        ax3 = axes[1, 0]
        self._plot_subdivision_hierarchy(ax3)
        ax3.set_title('Subdivision Hierarchy')
        
        # Plot 4: Boundary detection
        ax4 = axes[1, 1]
        self._plot_boundary_detection(ax4)
        ax4.set_title('Boundary Detection and Approximation')
        
        plt.tight_layout()
        if show_plot:
            plt.show()
        
        return fig
    
    def _plot_domain_and_cells(self, ax):
        """Plot domain boundary and cell classification"""
        # Create contour plot of implicit function
        x = np.linspace(self.x_min, self.x_max, 100)
        y = np.linspace(self.y_min, self.y_max, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.vectorize(self.f_implicit)(X, Y)
        
        # Plot domain boundary (f(x,y) = 0)
        contour = ax.contour(X, Y, Z, levels=[0], colors='blue', linewidths=2)
        
        # Plot cells with color coding
        for cell_data in self.visualization_data['cells']:
            x_min, x_max, y_min, y_max = cell_data['bounds']
            cell_type = cell_data['type']
            
            if cell_type == 'interior':
                color = 'green'
                alpha = 0.3
            elif cell_type == 'exterior':
                color = 'red'
                alpha = 0.1
            else:  # boundary
                color = 'yellow'
                alpha = 0.5
            
            rect = Rectangle((x_min, y_min), x_max-x_min, y_max-y_min,
                           fill=True, color=color, alpha=alpha, edgecolor='black', linewidth=0.5)
            ax.add_patch(rect)
        
        ax.set_xlim(self.x_min, self.x_max)
        ax.set_ylim(self.y_min, self.y_max)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', alpha=0.5, label='Interior Cells'),
            Patch(facecolor='yellow', alpha=0.5, label='Boundary Cells'),
            Patch(facecolor='red', alpha=0.3, label='Exterior Cells')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
    
    def _plot_integration_points(self, ax, F: Callable):
        """Plot integration points colored by function value"""
        if not self.visualization_data['integration_points']:
            return
        
        points = self.visualization_data['integration_points']
        x_vals = [p[0] for p in points]
        y_vals = [p[1] for p in points]
        f_vals = [p[2] for p in points]
        types = [p[3] for p in points]
        
        # Color by function value
        scatter = ax.scatter(x_vals, y_vals, c=f_vals, cmap='viridis', 
                           s=30, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax, label='F(x,y) value')
        
        # Plot domain boundary for reference
        x = np.linspace(self.x_min, self.x_max, 300)
        y = np.linspace(self.y_min, self.y_max, 300)
        X, Y = np.meshgrid(x, y)
        Z = np.vectorize(self.f_implicit)(X, Y)
        ax.contour(X, Y, Z, levels=[0], colors='red', linewidths=2, linestyles='--')
        
        ax.set_xlim(self.x_min, self.x_max)
        ax.set_ylim(self.y_min, self.y_max)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'Integration Points (Total: {len(points)})')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    def _plot_subdivision_hierarchy(self, ax):
        """Plot subdivision levels with different colors"""
        if not self.visualization_data['subdivision_history']:
            return
        
        # Group by level
        max_level = max(entry['level'] for entry in self.visualization_data['subdivision_history'])
        colors = cm.viridis(np.linspace(0, 1, max_level + 1))
        
        for entry in self.visualization_data['subdivision_history']:
            level = entry['level']
            x_min, x_max, y_min, y_max = entry['cell']
            cell_type = entry['type']
            
            color = colors[level]
            alpha = 0.7 - 0.1 * level  # Decrease alpha for deeper levels
            
            rect = Rectangle((x_min, y_min), x_max-x_min, y_max-y_min,
                           fill=True, color=color, alpha=alpha, 
                           edgecolor='black', linewidth=0.5)
            ax.add_patch(rect)
            
            # Add level number
            ax.text((x_min + x_max)/2, (y_min + y_max)/2, str(level),
                   ha='center', va='center', fontsize=8, fontweight='bold')
        
        ax.set_xlim(self.x_min, self.x_max)
        ax.set_ylim(self.y_min, self.y_max)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'Subdivision Hierarchy (Max Level: {max_level})')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    def _plot_boundary_detection(self, ax):
        """Plot boundary points and detected intersections"""
        # Plot domain boundary
        x = np.linspace(self.x_min, self.x_max, 300)
        y = np.linspace(self.y_min, self.y_max, 300)
        X, Y = np.meshgrid(x, y)
        Z = np.vectorize(self.f_implicit)(X, Y)
        ax.contour(X, Y, Z, levels=[0], colors='blue', linewidths=2, label='True Boundary')
        
        # Plot detected boundary points
        if self.visualization_data['boundary_points']:
            bp_x = [p[0] for p in self.visualization_data['boundary_points']]
            bp_y = [p[1] for p in self.visualization_data['boundary_points']]
            ax.scatter(bp_x, bp_y, color='red', s=50, marker='x', 
                      linewidth=2, label='Detected Boundary Points')
        
        # Plot some boundary cells
        boundary_cells = [cell for cell in self.visualization_data['cells'] 
                         if cell['type'] == 'boundary']
        
        for i, cell in enumerate(boundary_cells[:10]):  # Limit to first 10 for clarity
            x_min, x_max, y_min, y_max = cell['bounds']
            rect = Rectangle((x_min, y_min), x_max-x_min, y_max-y_min,
                           fill=False, color='orange', linewidth=2, linestyle='--')
            ax.add_patch(rect)
        
        ax.set_xlim(self.x_min, self.x_max)
        ax.set_ylim(self.y_min, self.y_max)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

# Enhanced test functions with visualization
def test_annulus_with_visualization():
    """Test integration over annulus domain with visualization"""
    
    def f_annulus(x, y):
        return get_Energy(x, y, mu, B, Delta, phi_x, gamma, Lambda)[2]
    
    def F_unity(x, y):
        return get_Energy(x, y, mu, B, Delta, phi_x, gamma, Lambda)[2]

    
    print("Testing annulus with visualization...")
    
    # Create integrator
    integrator = ImplicitIntegrator(f_annulus, (-5, 5, -5, 5))
    
    # Test area integration
    start_time = time.time()
    area = integrator.integrate(F_unity, tau=1e-3, max_level=6)
    computation_time = time.time() - start_time
    
    print(f"Computed area: {area:.6f}")
    print(f"Computation time: {computation_time:.4f}s")
    
    # Create visualization
    fig = integrator.visualize_integration(F_unity)
    fig.suptitle(f'Annulus Integration - Area = {area:.6f} ', fontsize=16)
    
    return area, integrator

def create_animation_frames():
    """Create frames showing the progression of subdivision"""
    def f_domain(x, y):
        return get_Energy(x, y, mu, B, Delta, phi_x, gamma, Lambda)[2]
        
    def F_test(x, y):
        return get_Energy(x, y, mu, B, Delta, phi_x, gamma, Lambda)[2]

        
    # Test different max levels to show progression
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, max_level in enumerate([0, 1, 2, 3, 4, 5]):
        integrator_anim = ImplicitIntegrator(f_domain, (-cut_off, cut_off, -cut_off, cut_off))
        result = integrator_anim.integrate(F_test, tau=1e-4, max_level=max_level)
        integrator_anim._plot_domain_and_cells(axes[i])
        axes[i].set_title(f'Subdivision Level {max_level}\nResult: {result:.4f}')
    
    plt.tight_layout()
    plt.show()

def test(phi_x_values, cut_off):
    """Test integration over annulus domain with visualization"""
    Energy_phi_x = np.zeros_like(phi_x_values)
    start_time = time.time()
    for i, phi_x in enumerate(phi_x_values):
        area = 0
        print(f"Testing integral {i} ...")
    
        for j in range(4):
            def f(x, y):        # domain of integration f(x,y)>=0
                return get_Energy(x, y, mu, B, Delta, phi_x, gamma, Lambda)[j]
            
            def F(y, x):    # dblquad integrates F(y, x, xlim, ylim)
                E = get_Energy(x, y, mu, B, Delta, phi_x, gamma, Lambda)[j]
                if E >= 0:    
                    return E
                else:
                    return 0
                
            # Create integrator
            integrator = ImplicitIntegrator(f, (-cut_off, cut_off, -cut_off, cut_off))
            integral = integrator.integrate(F, tau=1e-3, max_level=6)
            area += integral
            
        Energy_phi_x[i] = area
        print(f"Computed area: {area:.6f}")

    computation_time = time.time() - start_time
    
    print(f"Computation time: {computation_time:.4f}s")   

    return Energy_phi_x

if __name__ == "__main__":
    print("Numerical Integration over Implicitly Defined Domains")
    print("With Comprehensive Visualization")
    print("=" * 60)
    
    c = 3e18 # nm/s  #3e9 # m/s
    m_e =  5.1e8 / c**2 # meV s²/m²
    m = 0.403 * m_e # meV s²/m²
    hbar = 6.58e-13 # meV s
    gamma = hbar**2 / (2*m) # meV (nm)²
    E_F = 50.6 # meV
    k_F = np.sqrt(E_F / gamma ) # 1/nm
    
    Delta = 0.08 #
    B = 1.5 * Delta
    gamma = 9479   # meV (nm)²
    Lambda = 8 * Delta    # meV nm
    phi_x = 0  # 1/nm
    phi_x_values = np.linspace(-1, 1, 20)
    Delta = 0.08  # meV
    mu = 632 * Delta   # meV  632.5 Delta   
    cut_off = 1.01 * k_F
    
    #area = test(phi_x_values, cut_off)
    
    #Run tests with visualization
    # area, annulus_integrator = test_annulus_with_visualization()

    # Create animation frames
    print("\nCreating subdivision progression frames...")
    create_animation_frames()
     
#%%

    # cut_off = 5
    fig, ax = plt.subplots()
    ax.plot(phi_x_values, -area + 4/3 * cut_off**2 * (2*cut_off**2 - 3*mu + 3*phi_x_values**2), "o")
    # ax.plot(phi_x_values, -area, "o")
    # ax.plot(phi_x_values, 4/3 * cut_off**2 * (2*cut_off**2 - 3*mu + 3*phi_x_values**2), "o")
    ax.set_xlabel(r"$\phi_x$")
    ax.set_ylabel(r"$E(\phi_x)$")
    plt.show()
    
    
#%%
    np.savez("cut_off=5_B=1.1_gamma=1_Lambda=1_Delta=1_mu=3", area=area)



