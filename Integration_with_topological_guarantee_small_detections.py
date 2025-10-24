#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  5 16:32:46 2025

@author: gabriel
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List, Optional, Dict
import time
from matplotlib.patches import Polygon, Rectangle
from matplotlib.collections import PatchCollection
import matplotlib.cm as cm
from scipy.optimize import brentq

class EnhancedInterval:
    """Enhanced interval arithmetic with better function evaluation"""
    
    def __init__(self, a: float, b: Optional[float] = None):
        if b is None:
            b = a
        self.a = min(a, b)
        self.b = max(a, b)
    
    def __add__(self, other):
        if isinstance(other, EnhancedInterval):
            return EnhancedInterval(self.a + other.a, self.b + other.b)
        return EnhancedInterval(self.a + other, self.b + other)
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
        if isinstance(other, EnhancedInterval):
            return EnhancedInterval(self.a - other.b, self.b - other.a)
        return EnhancedInterval(self.a - other, self.b - other)
    
    def __rsub__(self, other):
        if isinstance(other, EnhancedInterval):
            return EnhancedInterval(other.a - self.b, other.b - self.a)
        return EnhancedInterval(other - self.b, other - self.a)
    
    def __mul__(self, other):
        if isinstance(other, EnhancedInterval):
            products = [self.a * other.a, self.a * other.b, 
                       self.b * other.a, self.b * other.b]
            return EnhancedInterval(min(products), max(products))
        return EnhancedInterval(self.a * other, self.b * other)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other):
        if isinstance(other, EnhancedInterval):
            if other.a <= 0 <= other.b:
                # Handle division by interval containing zero carefully
                if other.a == 0 and other.b == 0:
                    raise ValueError("Division by zero interval")
                # Split into positive and negative parts
                return EnhancedInterval(-np.inf, np.inf)
            reciprocals = [1/other.a, 1/other.b]
            return self * EnhancedInterval(min(reciprocals), max(reciprocals))
        if other == 0:
            raise ValueError("Division by zero")
        return EnhancedInterval(self.a / other, self.b / other)
    
    def contains_zero(self):
        return self.a <= 0 <= self.b
    
    def is_positive(self):
        return self.a > 0
    
    def is_negative(self):
        return self.b < 0
    
    def width(self):
        return self.b - self.a
    
    def midpoint(self):
        return (self.a + self.b) / 2
    
    def __repr__(self):
        return f"[{self.a:.6f}, {self.b:.6f}]"

class EnhancedImplicitIntegrator:
    """
    Enhanced numerical integration with improved boundary detection for small domains
    """
    
    def __init__(self, f_implicit: Callable, domain: Tuple[float, float, float, float]):
        self.f_implicit = f_implicit
        self.domain = domain
        self.x_min, self.x_max, self.y_min, self.y_max = domain
        
        # Enhanced visualization data
        self.visualization_data = {
            'cells': [],
            'integration_points': [],
            'boundary_points': [],
            'subdivision_history': [],
            'intersection_attempts': [],
            'small_features_detected': []
        }
        
        # Statistics
        self.stats = {
            'total_intersections': 0,
            'missed_intersections': 0,
            'small_features_found': 0,
            'bisection_iterations': 0
        }
    
    def integrate(self, F: Callable, tau: float = 1e-4, max_level: int = 8) -> float:
        """
        Main integration method
        """
        """
        # Reset visualization data
        self.visualization_data = {
            'cells': [],
            'integration_points': [],
            'boundary_points': [],
            'subdivision_history': [],
            'intersection_attempts': [],
            'small_features_detected': []
        }
        """
        initial_cell = self.domain
        return self.integrate_cell(initial_cell, F, tau, 0, max_level)
    
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
    
    def enhanced_interval_eval(self, x_interval: EnhancedInterval, y_interval: EnhancedInterval, 
                             num_samples: int = 9) -> EnhancedInterval:
        """
        Enhanced interval evaluation with adaptive sampling for small intervals
        """
        # For very small intervals, use more sophisticated approach
        if x_interval.width() < 1e-10 or y_interval.width() < 1e-10:
            # For tiny intervals, evaluate at midpoint
            mid_val = self.f_implicit(x_interval.midpoint(), y_interval.midpoint())
            return EnhancedInterval(mid_val, mid_val)
        
        # Use multiple sampling strategies
        strategies = []
        
        # Strategy 1: Corner sampling
        corners = [
            (x_interval.a, y_interval.a),
            (x_interval.a, y_interval.b),
            (x_interval.b, y_interval.a),
            (x_interval.b, y_interval.b)
        ]
        corner_vals = [self.f_implicit(x, y) for x, y in corners]
        strategies.append(EnhancedInterval(min(corner_vals), max(corner_vals)))
        
        # Strategy 2: Grid sampling for better coverage
        if num_samples > 4:
            x_samples = np.linspace(x_interval.a, x_interval.b, int(np.sqrt(num_samples)))
            y_samples = np.linspace(y_interval.a, y_interval.b, int(np.sqrt(num_samples)))
            grid_vals = []
            for x in x_samples:
                for y in y_samples:
                    grid_vals.append(self.f_implicit(x, y))
            strategies.append(EnhancedInterval(min(grid_vals), max(grid_vals)))
        
        # Strategy 3: Include midpoint
        mid_val = self.f_implicit(x_interval.midpoint(), y_interval.midpoint())
        strategies.append(EnhancedInterval(mid_val, mid_val))
        
        # Combine strategies
        all_mins = [s.a for s in strategies]
        all_maxs = [s.b for s in strategies]
        
        return EnhancedInterval(min(all_mins), max(all_maxs))
    
    def robust_find_intersections(self, cell: Tuple[float, float, float, float], 
                                tolerance: float = 1e-4) -> List[Tuple[float, float]]:
        """
        Robust intersection finding using multiple methods
        """
        x_min, x_max, y_min, y_max = cell
        intersections = []
        
        # Method 1: Check all edges with enhanced sampling
        edges = [
            ('vertical', x_min, y_min, y_max),   # left edge
            ('vertical', x_max, y_min, y_max),   # right edge
            ('horizontal', y_min, x_min, x_max), # bottom edge
            ('horizontal', y_max, x_min, x_max)  # top edge
        ]
        
        for edge_type, fixed, start, end in edges:
            edge_intersections = self.find_edge_intersections(edge_type, fixed, start, end, tolerance)
            intersections.extend(edge_intersections)
        
        # Method 2: Check for intersections inside the cell (not on edges)
        # This helps with very small domains entirely inside a cell
        internal_intersections = self.find_internal_intersections(cell, tolerance)
        intersections.extend(internal_intersections)
        
        # Remove duplicates
        unique_intersections = []
        for point in intersections:
            is_duplicate = False
            for existing in unique_intersections:
                if (abs(point[0] - existing[0]) < tolerance and 
                    abs(point[1] - existing[1]) < tolerance):
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_intersections.append(point)
        
        # Check if we found a very small feature
        if len(unique_intersections) >= 2:
            # Calculate approximate size of detected feature
            points_array = np.array(unique_intersections)
            bbox_size = np.max(points_array, axis=0) - np.min(points_array, axis=0)
            feature_size = np.max(bbox_size)
            
            if feature_size < min(x_max - x_min, y_max - y_min) * 0.1:
                self.stats['small_features_found'] += 1
                self.visualization_data['small_features_detected'].append({
                    'cell': cell,
                    'intersections': unique_intersections,
                    'feature_size': feature_size
                })
        
        self.stats['total_intersections'] += len(unique_intersections)
        
        return unique_intersections
    
    def find_edge_intersections(self, edge_type: str, fixed: float, start: float, end: float,
                              tolerance: float) -> List[Tuple[float, float]]:
        """Find intersections along a single edge"""
        intersections = []
        
        # Sample the edge at multiple points
        num_samples = max(5, int((end - start) / tolerance) + 1)
        samples = np.linspace(start, end, num_samples)
        
        if edge_type == 'vertical':
            # x is fixed, vary y
            prev_val = self.f_implicit(fixed, samples[0])
            for i in range(1, len(samples)):
                current_val = self.f_implicit(fixed, samples[i])
                
                if prev_val * current_val <= 0:  # Sign change or zero
                    # Found potential intersection
                    if abs(prev_val) < tolerance:
                        # Previous point is essentially on boundary
                        intersections.append((fixed, samples[i-1]))
                    elif abs(current_val) < tolerance:
                        # Current point is essentially on boundary
                        intersections.append((fixed, samples[i]))
                    else:
                        # Proper sign change - use robust root finding
                        try:
                            if edge_type == 'vertical':
                                root_func = lambda y: self.f_implicit(fixed, y)
                                root = brentq(root_func, samples[i-1], samples[i], 
                                            xtol=tolerance, maxiter=100)
                                intersections.append((fixed, root))
                                self.stats['bisection_iterations'] += 1
                            else:
                                root_func = lambda x: self.f_implicit(x, fixed)
                                root = brentq(root_func, samples[i-1], samples[i],
                                            xtol=tolerance, maxiter=100)
                                intersections.append((root, fixed))
                                self.stats['bisection_iterations'] += 1
                        except (ValueError, RuntimeError):
                            # Fallback to bisection
                            if edge_type == 'vertical':
                                y_root = self.bisection_search(lambda y: self.f_implicit(fixed, y),
                                                             samples[i-1], samples[i], tolerance)
                                intersections.append((fixed, y_root))
                            else:
                                x_root = self.bisection_search(lambda x: self.f_implicit(x, fixed),
                                                             samples[i-1], samples[i], tolerance)
                                intersections.append((x_root, fixed))
                
                prev_val = current_val
        
        return intersections
    
    def find_internal_intersections(self, cell: Tuple[float, float, float, float],
                                  tolerance: float) -> List[Tuple[float, float]]:
        """
        Find intersections that don't lie on cell edges
        Important for very small domains entirely inside a cell
        """
        x_min, x_max, y_min, y_max = cell
        intersections = []
        
        # Create a fine grid inside the cell
        grid_size = max(5, int(min(x_max - x_min, y_max - y_min) / tolerance) + 1)
        x_samples = np.linspace(x_min, x_max, grid_size)
        y_samples = np.linspace(y_min, y_max, grid_size)
        
        # Evaluate function on grid
        Z = np.zeros((len(y_samples), len(x_samples)))
        for i, y in enumerate(y_samples):
            for j, x in enumerate(x_samples):
                Z[i, j] = self.f_implicit(x, y)
        
        # Look for sign changes in both directions
        for i in range(len(y_samples) - 1):
            for j in range(len(x_samples) - 1):
                # Check 2x2 patch for sign changes
                patch = Z[i:i+2, j:j+2]
                if np.min(patch) <= 0 <= np.max(patch):
                    # Potential intersection in this patch
                    patch_x_min, patch_x_max = x_samples[j], x_samples[j+1]
                    patch_y_min, patch_y_max = y_samples[i], y_samples[i+1]
                    
                    # Try to find intersection more precisely
                    try:
                        # Use 2D root finding for small patches
                        def func_2d(params):
                            x, y = params
                            return self.f_implicit(x, y)
                        
                        # Start from patch center
                        x0 = (patch_x_min + patch_x_max) / 2
                        y0 = (patch_y_min + patch_y_max) / 2
                        
                        # Simple gradient descent for small patches
                        point = self.gradient_descent_root(func_2d, (x0, y0), tolerance)
                        if (x_min <= point[0] <= x_max and y_min <= point[1] <= y_max):
                            intersections.append(point)
                    except:
                        # Fallback: use patch center if function is near zero
                        center_val = self.f_implicit(x0, y0)
                        if abs(center_val) < tolerance:
                            intersections.append((x0, y0))
        
        return intersections
    
    def bisection_search(self, func: Callable, a: float, b: float, tolerance: float, 
                       max_iter: int = 50) -> float:
        """Robust bisection search for root finding"""
        fa, fb = func(a), func(b)
        
        if abs(fa) < tolerance:
            return a
        if abs(fb) < tolerance:
            return b
        
        for _ in range(max_iter):
            mid = (a + b) / 2
            f_mid = func(mid)
            
            if abs(f_mid) < tolerance or (b - a) < tolerance:
                return mid
            
            if fa * f_mid <= 0:
                b, fb = mid, f_mid
            else:
                a, fa = mid, f_mid
        
        return (a + b) / 2
    
    def gradient_descent_root(self, func: Callable, x0: Tuple[float, float], 
                            tolerance: float, max_iter: int = 20) -> Tuple[float, float]:
        """Simple gradient descent to find root for 2D case"""
        x, y = x0
        h = tolerance / 10
        
        for _ in range(max_iter):
            f_current = func((x, y))
            if abs(f_current) < tolerance:
                break
            
            # Numerical gradient
            df_dx = (func((x + h, y)) - f_current) / h
            df_dy = (func((x, y + h)) - f_current) / h
            
            grad_norm = np.sqrt(df_dx**2 + df_dy**2)
            if grad_norm < 1e-15:
                break
            
            # Step in direction opposite to gradient
            step_size = min(0.1, abs(f_current) / grad_norm)
            x -= step_size * df_dx / grad_norm
            y -= step_size * df_dy / grad_norm
        
        return (x, y)
    
    def classify_cell(self, cell: Tuple[float, float, float, float]) -> str:
        """Enhanced cell classification with small feature detection"""
        x_min, x_max, y_min, y_max = cell
        x_interval = EnhancedInterval(x_min, x_max)
        y_interval = EnhancedInterval(y_min, y_max)
        
        # Use enhanced interval evaluation
        f_interval = self.enhanced_interval_eval(x_interval, y_interval)
        
        # Store for visualization
        cell_data = {
            'bounds': cell,
            'type': 'interior' if f_interval.is_positive() else 
                   'exterior' if f_interval.is_negative() else 'boundary',
            'interval': f_interval,
            'size': (x_max - x_min, y_max - y_min)
        }
        
        self.visualization_data['cells'].append(cell_data)
        
        # If interval is very wide but cell is small, we might have a small feature
        if (f_interval.width() > 10 * abs(f_interval.midpoint()) and 
            min(x_max - x_min, y_max - y_min) < 0.01):
            # Force boundary classification for potential small features
            return 'boundary'
        
        if f_interval.is_positive():
            return 'interior'
        elif f_interval.is_negative():
            return 'exterior'
        else:
            return 'boundary'
    
    def integrate_cell(self, cell: Tuple[float, float, float, float], 
                      F: Callable, tau: float, level: int = 0, 
                      max_level: int = 10) -> float:
        """
        Enhanced integration with improved small feature handling
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
            result = self.gaussian_quadrature(cell, F)
            self.record_integration_points(cell, F, 'interior')
            return result
        else:  # boundary cell
            # Use enhanced intersection finding
            intersections = self.robust_find_intersections(cell)
            
            # Special handling for cells with no edge intersections but small features
            if len(intersections) < 2 and level < max_level:
                # Check if this might contain a very small feature
                cell_size = min(x_max - x_min, y_max - y_min)
                if cell_size < 0.01:  # Small cell
                    # Force subdivision to find small features
                    return self.subdivide_and_integrate(cell, F, tau, level, max_level)
            
            if len(intersections) < 2:
                # If still no intersections, use center point approximation
                center_val = self.f_implicit((x_min + x_max)/2, (y_min + y_max)/2)
                if center_val >= 0:
                    return self.gaussian_quadrature(cell, F)
                else:
                    return 0.0
            
            # Proceed with normal boundary handling
            bezier_curve = self.quadratic_bezier_approximation(intersections, cell)
            
            error_estimate = self.geometric_error_estimate(cell, bezier_curve, F)
            cell_area_ratio = (x_max - x_min) * (y_max - y_min) / (
                (self.x_max - self.x_min) * (self.y_max - self.y_min))
            scaled_tolerance = tau * cell_area_ratio
            
            if error_estimate < scaled_tolerance or level == max_level:
                result = self.integrate_boundary_cell(cell, F, bezier_curve)
                self.record_integration_points(cell, F, 'boundary')
                return result
            else:
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

    # ... (keep the previous visualization and helper methods, but update them to use enhanced features)

    def visualize_enhanced_detection(self, F: Callable):
        """Enhanced visualization showing small feature detection"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Enhanced Integration with Small Feature Detection', fontsize=16)
        
        # Plot 1: Domain and cell classification
        ax1 = axes[0, 0]
        self._plot_domain_and_cells(ax1)
        ax1.set_title('Cell Classification with Small Features')
        
        # Plot 2: Enhanced boundary detection
        ax2 = axes[0, 1]
        self._plot_enhanced_boundary_detection(ax2)
        ax2.set_title('Enhanced Boundary Detection')
        
        # Plot 3: Small features highlight
        ax3 = axes[1, 0]
        self._plot_small_features(ax3)
        ax3.set_title('Detected Small Features')
        
        # Plot 4: Statistics
        ax4 = axes[1, 1]
        self._plot_detection_statistics(ax4)
        ax4.set_title('Detection Statistics')
        
        plt.tight_layout()
        plt.show()
        
        # Print statistics
        print("\n=== ENHANCED DETECTION STATISTICS ===")
        print(f"Total intersections found: {self.stats['total_intersections']}")
        print(f"Small features detected: {self.stats['small_features_found']}")
        print(f"Bisection iterations: {self.stats['bisection_iterations']}")
        print(f"Missed intersections: {self.stats['missed_intersections']}")
    
    def _plot_enhanced_boundary_detection(self, ax):
        """Plot enhanced boundary detection results"""
        # Plot domain boundary
        x = np.linspace(self.x_min, self.x_max, 200)
        y = np.linspace(self.y_min, self.y_max, 200)
        X, Y = np.meshgrid(x, y)
        Z = np.vectorize(self.f_implicit)(X, Y)
        ax.contour(X, Y, Z, levels=[0], colors='blue', linewidths=2, label='True Boundary')
        
        # Plot all detected boundary points
        if self.visualization_data['boundary_points']:
            bp_x = [p[0] for p in self.visualization_data['boundary_points']]
            bp_y = [p[1] for p in self.visualization_data['boundary_points']]
            ax.scatter(bp_x, bp_y, color='red', s=30, marker='o', 
                      alpha=0.7, label='Detected Points')
        
        # Highlight small features
        for feature in self.visualization_data['small_features_detected']:
            cell = feature['cell']
            x_min, x_max, y_min, y_max = cell
            rect = Rectangle((x_min, y_min), x_max-x_min, y_max-y_min,
                           fill=False, color='orange', linewidth=3, linestyle='-')
            ax.add_patch(rect)
            
            # Mark intersections
            intersections = feature['intersections']
            if intersections:
                ix_x = [p[0] for p in intersections]
                ix_y = [p[1] for p in intersections]
                ax.scatter(ix_x, ix_y, color='green', s=50, marker='*', 
                          label='Small Feature' if not ax.get_legend() else "")
        
        ax.set_xlim(self.x_min, self.x_max)
        ax.set_ylim(self.y_min, self.y_max)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    def _plot_small_features(self, ax):
        """Zoom in on detected small features"""
        if not self.visualization_data['small_features_detected']:
            ax.text(0.5, 0.5, 'No small features detected', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # Find bounding box of all small features
        all_points = []
        for feature in self.visualization_data['small_features_detected']:
            all_points.extend(feature['intersections'])
        
        if all_points:
            points_array = np.array(all_points)
            x_center, y_center = np.mean(points_array, axis=0)
            x_range = max(0.1, np.ptp(points_array[:, 0]) * 1.5)
            y_range = max(0.1, np.ptp(points_array[:, 1]) * 1.5)
            
            ax.set_xlim(x_center - x_range/2, x_center + x_range/2)
            ax.set_ylim(y_center - y_range/2, y_center + y_range/2)
            
            # Plot zoomed-in view
            x_fine = np.linspace(x_center - x_range/2, x_center + x_range/2, 100)
            y_fine = np.linspace(y_center - y_range/2, y_center + y_range/2, 100)
            X_fine, Y_fine = np.meshgrid(x_fine, y_fine)
            Z_fine = np.vectorize(self.f_implicit)(X_fine, Y_fine)
            
            ax.contour(X_fine, Y_fine, Z_fine, levels=[0], colors='blue', linewidths=2)
            
            # Plot detected points
            for feature in self.visualization_data['small_features_detected']:
                intersections = feature['intersections']
                ix_x = [p[0] for p in intersections]
                ix_y = [p[1] for p in intersections]
                ax.scatter(ix_x, ix_y, color='red', s=50, marker='o')
                
                # Draw cell boundary
                cell = feature['cell']
                x_min, x_max, y_min, y_max = cell
                rect = Rectangle((x_min, y_min), x_max-x_min, y_max-y_min,
                               fill=False, color='orange', linewidth=2)
                ax.add_patch(rect)
        
        ax.set_title('Zoom on Small Features')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    def _plot_detection_statistics(self, ax):
        """Plot detection statistics"""
        stats = self.stats
        labels = ['Total\nIntersections', 'Small\nFeatures', 'Bisection\nIterations']
        values = [stats['total_intersections'], stats['small_features_found'], 
                 stats['bisection_iterations']]
        
        bars = ax.bar(labels, values, color=['skyblue', 'lightcoral', 'lightgreen'])
        ax.set_ylabel('Count')
        ax.set_title('Detection Performance')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{value}', ha='center', va='bottom')
        
        ax.grid(True, alpha=0.3, axis='y')

# Test with very small domains
def test_tiny_domains():
    """Test the enhanced integrator with very small domains"""
    
    # Test 1: Very small circle
    def f_tiny_circle(x, y):
        return 0.01 - (x**2 + y**2)  # Radius ~0.1
    
    # Test 2: Multiple tiny circles
    def f_tiny_circles(x, y):
        circle1 = 0.005 - ((x-0.5)**2 + (y-0.5)**2)
        circle2 = 0.005 - ((x+0.5)**2 + (y+0.5)**2)
        circle3 = 0.005 - ((x-0.5)**2 + (y+0.5)**2)
        return max(circle1, circle2, circle3)
    
    # Test 3: Thin annulus
    def f_thin_annulus(x, y):
        r = np.sqrt(x**2 + y**2)
        return 0.001 - (r - 0.7)**2  # Very thin ring
    
    # Test 4: Small complex shape
    def f_complex_small(x, y):
        return 0.02 - ((x**2 + y**2 - 0.5)**2 + (x - 0.1)**2 - 0.01)
    
    tests = [
        ("Tiny Circle", f_tiny_circle, (-0.2, 0.2, -0.2, 0.2)),
        ("Multiple Tiny Circles", f_tiny_circles, (-1, 1, -1, 1)),
        ("Thin Annulus", f_thin_annulus, (-1, 1, -1, 1)),
        ("Complex Small Shape", f_complex_small, (-0.3, 0.3, -0.3, 0.3))
    ]
    
    for name, f_func, domain in tests:
        print(f"\n{'='*50}")
        print(f"Testing: {name}")
        print(f"{'='*50}")
        
        integrator = EnhancedImplicitIntegrator(f_func, domain)
        
        def F_unity(x, y):
            return 1.0
        
        start_time = time.time()
        area = integrator.integrate(F_unity, tau=1e-5, max_level=8)
        comp_time = time.time() - start_time
        
        print(f"Computed area: {area:.8f}")
        print(f"Computation time: {comp_time:.4f}s")
        
        # Show enhanced visualization
        integrator.visualize_enhanced_detection(F_unity)

if __name__ == "__main__":
    print("Enhanced Numerical Integration with Small Domain Detection")
    print("=" * 60)
    
    test_tiny_domains()