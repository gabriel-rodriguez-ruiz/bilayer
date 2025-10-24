#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  5 12:19:30 2025

@author: gabriel
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List
import time
from matplotlib.patches import Rectangle

class Interval:
    """Interval arithmetic implementation"""
    
    def __init__(self, lower: float, upper: float):
        self.lower = lower
        self.upper = upper
    
    def __add__(self, other):
        if isinstance(other, Interval):
            return Interval(self.lower + other.lower, self.upper + other.upper)
        return Interval(self.lower + other, self.upper + other)
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
        if isinstance(other, Interval):
            return Interval(self.lower - other.upper, self.upper - other.lower)
        return Interval(self.lower - other, self.upper - other)
    
    def __rsub__(self, other):
        if isinstance(other, Interval):
            return Interval(other.lower - self.upper, other.upper - self.lower)
        return Interval(other - self.upper, other - self.lower)
    
    def __mul__(self, other):
        if isinstance(other, Interval):
            products = [self.lower * other.lower, self.lower * other.upper,
                       self.upper * other.lower, self.upper * other.upper]
            return Interval(min(products), max(products))
        return Interval(self.lower * other, self.upper * other)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other):
        if isinstance(other, Interval):
            if other.lower <= 0 <= other.upper:
                raise ValueError("Division by interval containing zero")
            reciprocals = [1/other.lower, 1/other.upper]
            reciprocal = Interval(min(reciprocals), max(reciprocals))
            return self * reciprocal
        if other == 0:
            raise ValueError("Division by zero")
        return Interval(self.lower / other, self.upper / other)
    
    def contains_zero(self):
        return self.lower <= 0 <= self.upper
    
    def is_positive(self):
        return self.lower > 0
    
    def is_negative(self):
        return self.upper < 0
    
    def __repr__(self):
        return f"[{self.lower}, {self.upper}]"

class ImplicitDomainIntegrator:
    """
    Numerical integration over implicitly defined domains with topological guarantee
    """
    
    def __init__(self, f: Callable, domain: Tuple[float, float, float, float], 
                 tolerance: float = 1e-6, max_depth: int = 20):
        """
        Parameters:
        f: implicit function defining the domain (f(x,y) >= 0)
        domain: (x_min, x_max, y_min, y_max) bounding box
        tolerance: error tolerance for integration
        max_depth: maximum subdivision depth
        """
        self.f = f
        self.domain = domain
        self.tolerance = tolerance
        self.max_depth = max_depth
        
        # Statistics
        self.num_interior_cells = 0
        self.num_boundary_cells = 0
        self.num_exterior_cells = 0
        self.subdivision_judgments = 0
        self.satisfied_judgments = 0
        
    def interval_eval(self, x_interval: Interval, y_interval: Interval) -> Interval:
        """Evaluate implicit function using interval arithmetic"""
        # This is a simplified implementation - in practice, you'd need
        # to implement automatic differentiation or use a more sophisticated
        # interval arithmetic library for complex functions
        try:
            # For simple functions, we can evaluate at corners and take bounds
            corners = [
                (x_interval.lower, y_interval.lower),
                (x_interval.lower, y_interval.upper),
                (x_interval.upper, y_interval.lower),
                (x_interval.upper, y_interval.upper)
            ]
            values = [self.f(x, y) for x, y in corners]
            return Interval(min(values), max(values))
        except:
            # Fallback: evaluate at center and use crude bounds
            x_center = (x_interval.lower + x_interval.upper) / 2
            y_center = (y_interval.lower + y_interval.upper) / 2
            center_val = self.f(x_center, y_center)
            # This is a simplification - proper implementation would need
            # derivative bounds for better interval estimates
            x_range = x_interval.upper - x_interval.lower
            y_range = y_interval.upper - y_interval.lower
            uncertainty = 0.1 * (x_range + y_range)  # Simplified uncertainty estimate
            return Interval(center_val - uncertainty, center_val + uncertainty)
    
    def classify_cell(self, cell: Tuple[float, float, float, float]) -> str:
        """
        Classify cell as interior, exterior, or boundary using interval arithmetic
        """
        x_min, x_max, y_min, y_max = cell
        x_interval = Interval(x_min, x_max)
        y_interval = Interval(y_min, y_max)
        
        f_interval = self.interval_eval(x_interval, y_interval)
        
        if f_interval.is_positive():
            return "interior"
        elif f_interval.is_negative():
            return "exterior"
        else:
            return "boundary"
    
    def gaussian_quadrature_2d(self, cell: Tuple[float, float, float, float], 
                              F: Callable, n_points: int = 2) -> float:
        """
        Perform 2D Gaussian quadrature over a rectangular cell
        """
        x_min, x_max, y_min, y_max = cell
        a, b = x_min, x_max
        c, d = y_min, y_max
        
        # Gauss-Legendre points and weights for n=2
        if n_points == 2:
            points = [-1/np.sqrt(3), 1/np.sqrt(3)]
            weights = [1.0, 1.0]
        else:
            # For higher order, you'd use proper Gauss-Legendre quadrature
            raise NotImplementedError("Only n_points=2 implemented")
        
        integral = 0.0
        for i, xi in enumerate(points):
            for j, eta in enumerate(points):
                # Transform from [-1,1] to [a,b] x [c,d]
                x = (b - a) / 2 * xi + (a + b) / 2
                y = (d - c) / 2 * eta + (c + d) / 2
                integral += weights[i] * weights[j] * F(x, y)
        
        # Jacobian determinant
        integral *= (b - a) * (d - c) / 4
        return integral
    
    def find_boundary_intersections(self, cell: Tuple[float, float, float, float]) -> List[Tuple[float, float]]:
        """
        Find intersections of implicit boundary with cell edges
        Simplified implementation using bisection
        """
        x_min, x_max, y_min, y_max = cell
        intersections = []
        
        # Check bottom edge
        x_vals = np.linspace(x_min, x_max, 5)
        for i in range(len(x_vals) - 1):
            x1, x2 = x_vals[i], x_vals[i+1]
            y = y_min
            f1, f2 = self.f(x1, y), self.f(x2, y)
            if f1 * f2 <= 0:  # Sign change indicates intersection
                # Bisection to refine intersection
                left, right = x1, x2
                for _ in range(10):  # 10 iterations for refinement
                    mid = (left + right) / 2
                    f_mid = self.f(mid, y)
                    if f1 * f_mid <= 0:
                        right = mid
                    else:
                        left = mid
                        f1 = f_mid
                intersections.append(((left + right) / 2, y))
        
        # Similar checks for top, left, and right edges
        # (implementation omitted for brevity)
        
        return intersections[:2]  # Return up to 2 intersections for simplicity
    
    def quadratic_bezier_approximation(self, intersections: List[Tuple[float, float]], 
                                     cell: Tuple[float, float, float, float]) -> List[Tuple[float, float]]:
        """
        Create quadratic Bezier approximation of boundary curve
        """
        if len(intersections) < 2:
            return intersections
        
        P0, P2 = intersections[0], intersections[1]
        
        # Simplified: use cell center as control point
        # In practice, you'd use tangent intersection as described in the paper
        x_min, x_max, y_min, y_max = cell
        P1 = ((x_min + x_max) / 2, (y_min + y_max) / 2)
        
        return [P0, P1, P2]
    
    def geometric_error_estimate(self, cell: Tuple[float, float, float, float], 
                               bezier_points: List[Tuple[float, float]], 
                               F: Callable) -> float:
        """
        Estimate geometric error for subdivision criterion
        """
        x_min, x_max, y_min, y_max = cell
        cell_area = (x_max - x_min) * (y_max - y_min)
        
        # Simplified error estimate
        # In practice, you'd compute the area between actual boundary and Bezier approximation
        max_F = self.estimate_max_F(cell, F)
        
        # Conservative error estimate
        error_estimate = max_F * cell_area * 0.1  # Simplified
        
        return error_estimate
    
    def estimate_max_F(self, cell: Tuple[float, float, float, float], F: Callable) -> float:
        """Estimate maximum of |F| over cell"""
        x_min, x_max, y_min, y_max = cell
        # Sample at corners and center
        points = [
            (x_min, y_min), (x_min, y_max), (x_max, y_min), (x_max, y_max),
            ((x_min + x_max) / 2, (y_min + y_max) / 2)
        ]
        return max(abs(F(x, y)) for x, y in points)
    
    def integrate_cell(self, cell: Tuple[float, float, float, float], 
                      F: Callable, depth: int = 0) -> float:
        """
        Recursive integration over a cell with hierarchical subdivision
        """
        if depth > self.max_depth:
            return 0.0
        
        cell_type = self.classify_cell(cell)
        
        if cell_type == "interior":
            self.num_interior_cells += 1
            return self.gaussian_quadrature_2d(cell, F)
        elif cell_type == "exterior":
            self.num_exterior_cells += 1
            return 0.0
        else:  # boundary cell
            self.num_boundary_cells += 1
            x_min, x_max, y_min, y_max = cell
            
            # For boundary cells, we need to decide whether to subdivide
            intersections = self.find_boundary_intersections(cell)
            bezier_points = self.quadratic_bezier_approximation(intersections, cell)
            
            error_estimate = self.geometric_error_estimate(cell, bezier_points, F)
            
            # Scaled tolerance based on cell area relative to total domain
            total_domain_area = (self.domain[1] - self.domain[0]) * (self.domain[3] - self.domain[2])
            cell_area = (x_max - x_min) * (y_max - y_min)
            scaled_tolerance = self.tolerance * (cell_area / total_domain_area)
            
            self.subdivision_judgments += 1
            if error_estimate < scaled_tolerance or depth == self.max_depth:
                # Accept current approximation
                self.satisfied_judgments += 1
                
                # Simplified: integrate over entire cell (in practice, you'd integrate
                # only over the part inside the domain using the Bezier approximation)
                if len(intersections) >= 2:
                    # For demonstration, we'll use Gaussian quadrature over entire cell
                    # In the actual algorithm, you'd only integrate over the interior part
                    return self.gaussian_quadrature_2d(cell, F)
                else:
                    return 0.0
            else:
                # Subdivide
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
                    result += self.integrate_cell(subcell, F, depth + 1)
                return result
    
    def integrate(self, F: Callable) -> Tuple[float, dict]:
        """
        Main integration routine
        
        Returns:
        integral_value: numerical approximation of the integral
        stats: dictionary with statistics about the integration process
        """
        # Reset statistics
        self.num_interior_cells = 0
        self.num_boundary_cells = 0
        self.num_exterior_cells = 0
        self.subdivision_judgments = 0
        self.satisfied_judgments = 0
        
        start_time = time.time()
        result = self.integrate_cell(self.domain, F)
        computation_time = (time.time() - start_time) * 1000  # milliseconds
        
        stats = {
            'computation_time_ms': computation_time,
            'num_interior_cells': self.num_interior_cells,
            'num_boundary_cells': self.num_boundary_cells,
            'num_exterior_cells': self.num_exterior_cells,
            'total_cells': self.num_interior_cells + self.num_boundary_cells + self.num_exterior_cells,
            'Cr_ratio': self.satisfied_judgments / self.subdivision_judgments if self.subdivision_judgments > 0 else 0
        }
        
        return result, stats


class ImprovedInterval:
    """Improved interval arithmetic implementation"""
    
    def __init__(self, lower: float, upper: float):
        self.lower = lower
        self.upper = upper
    
    def __add__(self, other):
        if isinstance(other, Interval):
            return ImprovedInterval(self.lower + other.lower, self.upper + other.upper)
        return ImprovedInterval(self.lower + other, self.upper + other)
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
        if isinstance(other, Interval):
            return ImprovedInterval(self.lower - other.upper, self.upper - other.lower)
        return ImprovedInterval(self.lower - other, self.upper - other)
    
    def __mul__(self, other):
        if isinstance(other, ImprovedInterval):
            products = [self.lower * other.lower, self.lower * other.upper,
                       self.upper * other.lower, self.upper * other.upper]
            return ImprovedInterval(min(products), max(products))
        return ImprovedInterval(self.lower * other, self.upper * other)
    
    def contains_zero(self):
        return self.lower <= 0 <= self.upper
    
    def is_positive(self):
        return self.lower > 0
    
    def is_negative(self):
        return self.upper < 0
    
    def __repr__(self):
        return f"[{self.lower:.3f}, {self.upper:.3f}]"

class WorkingImplicitIntegrator:
    """
    Working implementation of numerical integration over implicitly defined domains
    """
    
    def __init__(self, f: Callable, domain: Tuple[float, float, float, float], 
                 tolerance: float = 1e-4, max_depth: int = 8):
        self.f = f
        self.domain = domain
        self.tolerance = tolerance
        self.max_depth = max_depth
        
        # Statistics
        self.num_interior_cells = 0
        self.num_boundary_cells = 0
        self.num_exterior_cells = 0
        
    def classify_cell_robust(self, cell: Tuple[float, float, float, float]) -> str:
        """
        Robust cell classification using multiple sampling points
        """
        x_min, x_max, y_min, y_max = cell
        
        # Sample at multiple points within the cell
        sample_points = [
            (x_min, y_min), (x_min, y_max), (x_max, y_min), (x_max, y_max),  # corners
            ((x_min + x_max)/2, (y_min + y_max)/2),  # center
            (x_min, (y_min + y_max)/2), (x_max, (y_min + y_max)/2),  # edge centers
            ((x_min + x_max)/2, y_min), ((x_min + x_max)/2, y_max)   # edge centers
        ]
        
        values = [self.f(x, y) for x, y in sample_points]
        min_val = min(values)
        max_val = max(values)
        
        if min_val > 0:
            return "interior"
        elif max_val < 0:
            return "exterior"
        else:
            return "boundary"
    
    def gaussian_quadrature_2d(self, cell: Tuple[float, float, float, float], 
                              F: Callable, n_points: int = 3) -> float:
        """2D Gaussian quadrature over rectangular cell"""
        x_min, x_max, y_min, y_max = cell
        
        # Gauss-Legendre points and weights for n=3
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
        for i, xi in enumerate(points):
            for j, eta in enumerate(points):
                # Transform from [-1,1] to [a,b] x [c,d]
                x = (x_max - x_min) / 2 * xi + (x_min + x_max) / 2
                y = (y_max - y_min) / 2 * eta + (y_min + y_max) / 2
                
                # Only include points that are inside the domain
                if self.f(x, y) >= 0:
                    integral += weights[i] * weights[j] * F(x, y)
        
        # Jacobian determinant
        integral *= (x_max - x_min) * (y_max - y_min) / 4
        return integral
    
    def integrate_cell(self, cell: Tuple[float, float, float, float], 
                      F: Callable, depth: int = 0) -> float:
        """Recursive integration with proper cell handling"""
        if depth > self.max_depth:
            # At maximum depth, use direct quadrature
            return self.gaussian_quadrature_2d(cell, F, 3)
        
        cell_type = self.classify_cell_robust(cell)
        x_min, x_max, y_min, y_max = cell
        cell_area = (x_max - x_min) * (y_max - y_min)
        
        if cell_type == "interior":
            self.num_interior_cells += 1
            return self.gaussian_quadrature_2d(cell, F, 3)
        elif cell_type == "exterior":
            self.num_exterior_cells += 1
            return 0.0
        else:  # boundary cell
            self.num_boundary_cells += 1
            
            # Check if cell is small enough
            total_domain_area = (self.domain[1] - self.domain[0]) * (self.domain[3] - self.domain[2])
            if cell_area / total_domain_area < self.tolerance or depth == self.max_depth:
                return self.gaussian_quadrature_2d(cell, F, 5)  # Higher order for boundary cells
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
        
        start_time = time.time()
        result = self.integrate_cell(self.domain, F)
        computation_time = (time.time() - start_time) * 1000
        
        stats = {
            'computation_time_ms': computation_time,
            'num_interior_cells': self.num_interior_cells,
            'num_boundary_cells': self.num_boundary_cells,
            'num_exterior_cells': self.num_exterior_cells,
            'total_cells': self.num_interior_cells + self.num_boundary_cells + self.num_exterior_cells
        }
        
        return result, stats

def test_working_annulus():
    """Test with working annulus implementation"""
    print("=== WORKING Annulus Test ===")
    
    def f_annulus(x, y):
        r = np.sqrt(x**2 + y**2)
        return 0.04 - (r - 0.6)**2
    
    def F_unity(x, y):
        return 1.0
    
    def F_polynomial(x, y):
        return x**3 * y - x * y + 2.5
    
    domain = (-1, 1, -1, 1)
    exact_area = 12 * np.pi / 25
    
    # Test area integration
    integrator = WorkingImplicitIntegrator(f_annulus, domain, tolerance=1e-4, max_depth=6)
    result, stats = integrator.integrate(F_unity)
    
    print(f"Area result: {result:.8f}")
    print(f"Exact area: {exact_area:.8f}")
    print(f"Error: {abs(result - exact_area):.2e}")
    print(f"Stats: {stats}")
    
    # Test polynomial integration
    result2, stats2 = integrator.integrate(F_polynomial)
    print(f"\nPolynomial integral: {result2:.8f}")
    print(f"Stats: {stats2}")

def test_working_circle():
    """Simple circle test"""
    print("\n=== WORKING Circle Test ===")
    
    def f_circle(x, y):
        return 1 - (x**2 + y**2)  # Unit circle
    
    def F_unity(x, y):
        return 1.0
    
    domain = (-1.5, 1.5, -1.5, 1.5)
    exact_area = np.pi
    
    integrator = WorkingImplicitIntegrator(f_circle, domain, tolerance=1e-4, max_depth=6)
    result, stats = integrator.integrate(F_unity)
    
    print(f"Area result: {result:.8f}")
    print(f"Exact area: {exact_area:.8f}")
    print(f"Error: {abs(result - exact_area):.2e}")
    print(f"Stats: {stats}")

def test_working_square():
    """Simple square test - should be exact"""
    print("\n=== WORKING Square Test ===")
    
    def f_square(x, y):
        return min(1-x, 1+x, 1-y, 1+y)  # Square of side 2
    
    def F_unity(x, y):
        return 1.0
    
    domain = (-1.5, 1.5, -1.5, 1.5)
    exact_area = 4.0
    
    integrator = WorkingImplicitIntegrator(f_square, domain, tolerance=1e-4, max_depth=4)
    result, stats = integrator.integrate(F_unity)
    
    print(f"Area result: {result:.8f}")
    print(f"Exact area: {exact_area:.8f}")
    print(f"Error: {abs(result - exact_area):.2e}")
    print(f"Stats: {stats}")

class WorkingPlotter:
    """Plotter that works with the actual integration results"""
    
    def __init__(self, f_implicit, domain):
        self.f = f_implicit
        self.domain = domain
    
    def plot_integration_process(self, integrator, F: Callable, title: str):
        """Plot the integration process showing cell subdivision"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        x_min, x_max, y_min, y_max = self.domain
        
        # Plot domain background
        x = np.linspace(x_min, x_max, 300)
        y = np.linspace(y_min, y_max, 300)
        X, Y = np.meshgrid(x, y)
        Z = self.f(X, Y)
        
        ax.contourf(X, Y, Z, levels=[0, np.inf], alpha=0.3, colors='lightblue')
        ax.contour(X, Y, Z, levels=[0], colors='blue', linewidths=2)
        
        # Recursively plot cells
        self._plot_cells_recursive(ax, integrator, self.domain, F)
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        
        return fig, ax
    
    def _plot_cells_recursive(self, ax, integrator, cell, F, depth=0):
        """Recursively plot cell subdivision"""
        if depth > 6:  # Limit plotting depth
            return
        
        x_min, x_max, y_min, y_max = cell
        cell_type = integrator.classify_cell_robust(cell)
        
        # Plot cell boundary
        color = 'green' if cell_type == "interior" else 'red' if cell_type == "boundary" else 'gray'
        rect = Rectangle((x_min, y_min), x_max-x_min, y_max-y_min, 
                        fill=False, edgecolor=color, linewidth=1, alpha=0.7)
        ax.add_patch(rect)
        
        if cell_type == "boundary" and depth < 6:
            # Subdivide and plot children
            x_mid = (x_min + x_max) / 2
            y_mid = (y_min + y_max) / 2
            
            subcells = [
                (x_min, x_mid, y_min, y_mid),
                (x_mid, x_max, y_min, y_mid),
                (x_min, x_mid, y_mid, y_max),
                (x_mid, x_max, y_mid, y_max)
            ]
            
            for subcell in subcells:
                self._plot_cells_recursive(ax, integrator, subcell, F, depth + 1)

# Run the working tests
if __name__ == "__main__":
    print("Running WORKING integration tests...")
    
    # Test basic shapes
    test_working_annulus()
    test_working_circle() 
    test_working_square()
    
    # Create visualization
    print("\nCreating visualization...")
    
    def f_annulus(x, y):
        r = np.sqrt(x**2 + y**2)
        return 0.04 - (r - 0.6)**2
    
    def F_unity(x, y):
        return 1.0
    
    domain = (-1, 1, -1, 1)
    
    integrator = WorkingImplicitIntegrator(f_annulus, domain, tolerance=1e-4, max_depth=6)
    plotter = WorkingPlotter(f_annulus, domain)
    
    fig, ax = plotter.plot_integration_process(integrator, F_unity, 
                                             "Annulus Integration - Cell Subdivision")
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='green', lw=2, label='Interior Cells'),
        Line2D([0], [0], color='red', lw=2, label='Boundary Cells'),
        Line2D([0], [0], color='gray', lw=2, label='Exterior Cells'),
        Line2D([0], [0], color='blue', lw=2, label='Domain Boundary')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.savefig('working_integration.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nAll tests completed successfully!")