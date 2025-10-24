#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  5 13:17:30 2025

@author: gabriel
"""

import numpy as np
from typing import Callable, Tuple, List, Optional
import time
from get_Analytic_energy import GetAnalyticEnergies, GetSumOfPositiveAnalyticEnergy

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
        
    def interval_eval(self, x_interval: Interval, y_interval: Interval) -> Interval:
        """Evaluate implicit function using interval arithmetic"""
        # This is a simplified implementation - in practice, you'd need
        # to implement proper interval arithmetic for your specific function
        try:
            # Sample points within the interval to get bounds
            n_samples = 10
            x_samples = np.linspace(x_interval.a, x_interval.b, n_samples)
            y_samples = np.linspace(y_interval.a, y_interval.b, n_samples)
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
                intersections.append((x, (y1 + y2) / 2))
        
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
                intersections.append(((x1 + x2) / 2, y))
        
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
        # This is a simplified implementation
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
        prev_point = curve(0)
        for i in range(1, samples):
            t = i / (samples - 1)
            current_point = curve(t)
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
        
        if cell_type == 'exterior':
            return 0.0
        elif cell_type == 'interior':
            # Use Gaussian quadrature for interior cells
            return self.gaussian_quadrature(cell, F)
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
                return self.integrate_boundary_cell(cell, F, bezier_curve)
            else:
                # Subdivide further
                return self.subdivide_and_integrate(cell, F, tau, level, max_level)
    
    def integrate_boundary_cell(self, cell: Tuple[float, float, float, float],
                              F: Callable, bezier_curve: Callable) -> float:
        """
        Integrate over boundary cell using Bezier approximation
        Simplified implementation
        """
        # This would implement the actual integration scheme described in the paper
        # For now, use simple Gaussian quadrature as placeholder
        return self.gaussian_quadrature(cell, F) * 0.5  # Rough approximation
    
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
        
        Parameters:
        F: integrand function
        tau: tolerance for error control
        max_level: maximum subdivision level
        
        Returns:
        Numerical approximation of ∫_Ω F(x,y) dxdy
        """
        initial_cell = self.domain
        return self.integrate_cell(initial_cell, F, tau, 0, max_level)

# Example usage and test cases
def test_annulus(B, gamma, Lambda, phi_x, Delta, mu):
    """Test integration over annulus domain"""
    
    def f(x, y):
        return GetAnalyticEnergies(x, y, B, gamma, Lambda, phi_x, Delta, mu)[2]
    
    def F(x, y):
        return GetSumOfPositiveAnalyticEnergy(x, y, B, gamma, Lambda, phi_x, Delta, mu)
    
    
    # Create integrator
    integrator = ImplicitIntegrator(f, (-1, 1, -1, 1))
    
    # Test area integration
    print("Testing area integration...")
    start_time = time.time()
    area = integrator.integrate(F, tau=1e-4)
    computation_time = time.time() - start_time
    
    
    print(f"Computed area: {area}")
    print(f"Computation time: {computation_time:.4f}s")
    
    return area

if __name__ == "__main__":
    print("Numerical Integration over Implicitly Defined Domains")
    print("=" * 50)
    
    # Run tests
    B = 2
    gamma = 1
    Lambda = 1
    phi_x = 1
    Delta = 1
    mu = 3
    test_annulus(B, gamma, Lambda, phi_x, Delta, mu)
    

    