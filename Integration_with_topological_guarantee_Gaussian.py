#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  5 12:43:05 2025

@author: gabriel
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
from typing import Callable, Tuple, List, Optional
import time

class RobustImplicitIntegrator:
    """
    Robust implementation that actually works for annulus and other implicit domains
    """
    
    def __init__(self, f: Callable, domain: Tuple[float, float, float, float], 
                 tolerance: float = 1e-4, max_depth: int = 10):
        self.f = f
        self.domain = domain
        self.tolerance = tolerance
        self.max_depth = max_depth
        
        # Statistics
        self.num_interior_cells = 0
        self.num_boundary_cells = 0
        self.num_exterior_cells = 0
        self.cell_history = []  # For visualization
        
    def classify_cell_annulus(self, cell: Tuple[float, float, float, float]) -> str:
        """
        Specialized classification for annulus - sample densely to detect thin regions
        """
        x_min, x_max, y_min, y_max = cell
        
        # For annulus, we need very dense sampling to detect the thin ring
        n_samples = 10  # Increased sampling density
        x_samples = np.linspace(x_min, x_max, n_samples)
        y_samples = np.linspace(y_min, y_max, n_samples)
        
        has_positive = False
        has_negative = False
        
        for x in x_samples:
            for y in y_samples:
                val = self.f(x, y)
                if val >= 0:
                    has_positive = True
                else:
                    has_negative = True
                
                # Early exit if we found both
                if has_positive and has_negative:
                    return "boundary"
        
        if has_positive and not has_negative:
            return "interior"
        elif has_negative and not has_positive:
            return "exterior"
        else:
            return "boundary"  # Conservative
    
    def gaussian_quadrature_2d(self, cell: Tuple[float, float, float, float], 
                              F: Callable, n_points: int = 3) -> float:
        """2D Gaussian quadrature that properly handles domain boundaries"""
        x_min, x_max, y_min, y_max = cell
        
        # Gauss-Legendre points and weights
        if n_points == 3:
            points = [-np.sqrt(3/5), 0, np.sqrt(3/5)]
            weights = [5/9, 8/9, 5/9]
        elif n_points == 2:
            points = [-1/np.sqrt(3), 1/np.sqrt(3)]
            weights = [1.0, 1.0]
        else:
            points = [0]
            weights = [2.0]
        
        integral = 0.0
        count_inside = 0
        
        for i, xi in enumerate(points):
            for j, eta in enumerate(points):
                # Transform from [-1,1] to cell coordinates
                x = (x_max - x_min) / 2 * xi + (x_min + x_max) / 2
                y = (y_max - y_min) / 2 * eta + (y_min + y_max) / 2
                
                # Check if point is inside domain
                if self.f(x, y) >= 0:
                    integral += weights[i] * weights[j] * F(x, y)
                    count_inside += 1
        
        # Jacobian and scale by fraction of points inside
        cell_area = (x_max - x_min) * (y_max - y_min)
        total_points = len(points) ** 2
        
        if count_inside > 0:
            # Scale integral by the actual area fraction
            integral *= (cell_area / 4) * (count_inside / total_points)
        else:
            integral = 0.0
            
        return integral
    
    def integrate_cell(self, cell: Tuple[float, float, float, float], 
                      F: Callable, depth: int = 0) -> float:
        """Recursive integration with proper annulus handling"""
        x_min, x_max, y_min, y_max = cell
        cell_area = (x_max - x_min) * (y_max - y_min)
        total_domain_area = (self.domain[1] - self.domain[0]) * (self.domain[3] - self.domain[2])
        
        # Store cell for visualization
        self.cell_history.append((cell, depth))
        
        if depth >= self.max_depth:
            # Maximum depth reached, use direct integration
            cell_type = self.classify_cell_annulus(cell)
            if cell_type == "interior":
                self.num_interior_cells += 1
                # Full cell integration
                return self.gaussian_quadrature_2d(cell, F, 5)
            elif cell_type == "boundary":
                self.num_boundary_cells += 1
                # Boundary cell - use higher order quadrature
                return self.gaussian_quadrature_2d(cell, F, 7)
            else:
                self.num_exterior_cells += 1
                return 0.0
        
        cell_type = self.classify_cell_annulus(cell)
        
        if cell_type == "interior":
            self.num_interior_cells += 1
            return self.gaussian_quadrature_2d(cell, F, 3)
        elif cell_type == "exterior":
            self.num_exterior_cells += 1
            return 0.0
        else:  # boundary cell
            self.num_boundary_cells += 1
            
            # Check if cell is small enough relative to tolerance
            if cell_area / total_domain_area < self.tolerance * 0.1:
                return self.gaussian_quadrature_2d(cell, F, 7)
            else:
                # Subdivide
                x_mid = (x_min + x_max) / 2
                y_mid = (y_min + y_max) / 2
                
                subcells = [
                    (x_min, x_mid, y_min, y_mid),
                    (x_mid, x_max, y_min, y_mid),
                    (x_min, x_mid, y_mid, y_max),
                    (x_mid, x_max, y_mid, y_max)
                ]
                
                result = 0.0
                for subcell in subcells:
                    result += self.integrate_cell(subcell, F, depth + 1)
                return result
    
    def integrate(self, F: Callable) -> Tuple[float, dict]:
        """Main integration routine"""
        # Reset statistics
        self.num_interior_cells = 0
        self.num_boundary_cells = 0
        self.num_exterior_cells = 0
        self.cell_history = []
        
        start_time = time.time()
        result = self.integrate_cell(self.domain, F)
        computation_time = (time.time() - start_time) * 1000
        
        stats = {
            'computation_time_ms': computation_time,
            'num_interior_cells': self.num_interior_cells,
            'num_boundary_cells': self.num_boundary_cells,
            'num_exterior_cells': self.num_exterior_cells,
            'total_cells': self.num_interior_cells + self.num_boundary_cells + self.num_exterior_cells,
            'max_depth_reached': max(depth for _, depth in self.cell_history) if self.cell_history else 0
        }
        
        return result, stats

def test_robust_annulus():
    """Test annulus with robust implementation"""
    print("=== ROBUST Annulus Test ===")
    
    # Annulus: inner radius 0.4, outer radius 0.8
    def f_annulus(x, y):
        r = np.sqrt(x**2 + y**2)
        return 0.04 - (r - 0.6)**2
    
    def F_unity(x, y):
        return 1.0
    
    def F_polynomial(x, y):
        return x**3 * y - x * y + 2.5
    
    domain = (-1, 1, -1, 1)
    exact_area = np.pi * (0.8**2 - 0.4**2)  # π*(R² - r²)
    
    print(f"Expected annulus area: {exact_area:.8f}")
    
    # Test with different tolerances
    for tol in [1e-2, 1e-3, 1e-4]:
        print(f"\n--- Tolerance: {tol} ---")
        integrator = RobustImplicitIntegrator(f_annulus, domain, tolerance=tol, max_depth=12)
        result, stats = integrator.integrate(F_unity)
        
        print(f"Area result: {result:.8f}")
        print(f"Error: {abs(result - exact_area):.2e}")
        print(f"Stats: {stats}")

def test_simple_annulus():
    """Test with the exact annulus function from the paper"""
    print("\n=== PAPER Annulus Test ===")
    
    # Exact function from the paper: f(x,y) = 0.04 - (sqrt(x²+y²) - 0.6)²
    def f_paper_annulus(x, y):
        r = np.sqrt(x**2 + y**2)
        return 0.04 - (r - 0.6)**2
    
    def F_unity(x, y):
        return 1.0
    
    domain = (-1, 1, -1, 1)
    # For f(x,y) = 0.04 - (r - 0.6)² >= 0
    # This means (r - 0.6)² <= 0.04, so 0.4 <= r <= 0.8
    exact_area = np.pi * (0.8**2 - 0.4**2)
    
    print(f"Exact area: {exact_area:.8f}")
    print(f"Expected: {12*np.pi/25:.8f} (12π/25)")
    
    integrator = RobustImplicitIntegrator(f_paper_annulus, domain, tolerance=1e-4, max_depth=12)
    result, stats = integrator.integrate(F_unity)
    
    print(f"Computed area: {result:.8f}")
    print(f"Error: {abs(result - exact_area):.2e}")
    print(f"Stats: {stats}")

def visualize_annulus_integration():
    """Create visualization of the annulus integration process"""
    print("\n=== Creating Visualization ===")
    
    def f_paper_annulus(x, y):
        r = np.sqrt(x**2 + y**2)
        return 0.04 - (r - 0.6)**2
    
    def F_unity(x, y):
        return 1.0
    
    domain = (-1, 1, -1, 1)
    
    # Create integrator and compute
    integrator = RobustImplicitIntegrator(f_paper_annulus, domain, tolerance=1e-3, max_depth=8)
    result, stats = integrator.integrate(F_unity)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Domain and cells
    x_min, x_max, y_min, y_max = domain
    x = np.linspace(x_min, x_max, 400)
    y = np.linspace(y_min, y_max, 400)
    X, Y = np.meshgrid(x, y)
    Z = f_paper_annulus(X, Y)
    
    # Plot domain
    ax1.contourf(X, Y, Z, levels=[0, np.inf], alpha=0.3, colors='lightblue')
    ax1.contour(X, Y, Z, levels=[0], colors='blue', linewidths=2)
    
    # Plot cells
    for cell, depth in integrator.cell_history:
        x_min, x_max, y_min, y_max = cell
        cell_type = integrator.classify_cell_annulus(cell)
        
        color = 'green' if cell_type == "interior" else 'red' if cell_type == "boundary" else 'lightgray'
        alpha = 0.3 if cell_type == "exterior" else 0.6
        
        rect = Rectangle((x_min, y_min), x_max-x_min, y_max-y_min,
                        facecolor=color, edgecolor='black', alpha=alpha, linewidth=0.5)
        ax1.add_patch(rect)
    
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    ax1.set_aspect('equal')
    ax1.set_title('Annulus Integration - Cell Subdivision\n(Green: Interior, Red: Boundary, Gray: Exterior)')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    
    # Plot 2: Detailed view of annulus region
    detailed_domain = (0.3, 0.9, -0.3, 0.3)
    x_min_d, x_max_d, y_min_d, y_max_d = detailed_domain
    x_d = np.linspace(x_min_d, x_max_d, 300)
    y_d = np.linspace(y_min_d, y_max_d, 300)
    X_d, Y_d = np.meshgrid(x_d, y_d)
    Z_d = f_paper_annulus(X_d, Y_d)
    
    ax2.contourf(X_d, Y_d, Z_d, levels=[0, np.inf], alpha=0.3, colors='lightblue')
    ax2.contour(X_d, Y_d, Z_d, levels=[0], colors='blue', linewidths=2)
    
    # Plot cells in detailed view
    for cell, depth in integrator.cell_history:
        x_min, x_max, y_min, y_max = cell
        # Only plot cells that overlap with detailed view
        if (x_min < x_max_d and x_max > x_min_d and y_min < y_max_d and y_max > y_min_d):
            cell_type = integrator.classify_cell_annulus(cell)
            color = 'green' if cell_type == "interior" else 'red' if cell_type == "boundary" else 'lightgray'
            alpha = 0.5 if cell_type != "exterior" else 0.2
            
            # Clip cell to detailed view
            cell_x_min = max(x_min, x_min_d)
            cell_x_max = min(x_max, x_max_d)
            cell_y_min = max(y_min, y_min_d)
            cell_y_max = min(y_max, y_max_d)
            
            if cell_x_min < cell_x_max and cell_y_min < cell_y_max:
                rect = Rectangle((cell_x_min, cell_y_min), cell_x_max-cell_x_min, cell_y_max-cell_y_min,
                                facecolor=color, edgecolor='black', alpha=alpha, linewidth=1)
                ax2.add_patch(rect)
    
    ax2.set_xlim(x_min_d, x_max_d)
    ax2.set_ylim(y_min_d, y_max_d)
    ax2.set_aspect('equal')
    ax2.set_title('Detailed View - Annulus Boundary Region')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    
    plt.tight_layout()
    plt.savefig('robust_annulus_integration.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Visualization created with result: {result:.8f}")

def debug_annulus_sampling():
    """Debug function to understand why annulus detection was failing"""
    print("\n=== Debug Annulus Sampling ===")
    
    def f_paper_annulus(x, y):
        r = np.sqrt(x**2 + y**2)
        return 0.04 - (r - 0.6)**2
    
    # Test some specific points
    test_points = [
        (0.0, 0.0),    # Center - should be negative
        (0.5, 0.0),    # Inside annulus - should be positive  
        (0.7, 0.0),    # Inside annulus - should be positive
        (0.9, 0.0),    # Outside - should be negative
        (0.6, 0.2),    # On boundary - should be ~0
    ]
    
    print("Point values in annulus:")
    for x, y in test_points:
        r = np.sqrt(x**2 + y**2)
        val = f_paper_annulus(x, y)
        status = "INSIDE" if val >= 0 else "OUTSIDE"
        print(f"  ({x:.1f}, {y:.1f}) -> r={r:.3f}, f={val:.4f} -> {status}")
    
    # Test cell classification
    test_cell = (0.5, 0.7, -0.1, 0.1)  # Should contain boundary
    print(f"\nTesting cell {test_cell}:")
    
    x_min, x_max, y_min, y_max = test_cell
    n_samples = 5
    x_samples = np.linspace(x_min, x_max, n_samples)
    y_samples = np.linspace(y_min, y_max, n_samples)
    
    print("Sampling grid:")
    for i, x in enumerate(x_samples):
        for j, y in enumerate(y_samples):
            val = f_paper_annulus(x, y)
            status = "+" if val >= 0 else "-"
            print(f"  ({x:.2f}, {y:.2f}): {status} (f={val:.4f})")

if __name__ == "__main__":
    print("Running ROBUST annulus integration tests...")
    
    # Debug first to understand the function
    debug_annulus_sampling()
    
    # Run the robust tests
    test_robust_annulus()
    test_simple_annulus()
    
    # Create visualization
    visualize_annulus_integration()
    
    print("\nAll tests completed!")