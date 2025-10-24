#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  7 17:28:55 2025

@author: gabriel
"""

import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.patches import Polygon, Rectangle
import matplotlib.cm as cm
from pauli_matrices import tau_0, tau_z, sigma_0, tau_x, sigma_z, sigma_x, sigma_y
import scipy
from Integration_with_topological_guarantee_simple_visualization import test

def Energy_vs_B(phi_x_values):
    

if __name__ == "__main__":
    print("Numerical Integration over Implicitly Defined Domains")
    print("With Comprehensive Visualization")
    print("=" * 60)
    
    B = 1.1
    gamma = 1
    Lambda = 1
    phi_x = -0.02
    phi_x_values = np.linspace(-1, 0, 10)
    Delta = 1
    mu = 3
    
    area = test(phi_x_values)
    
    #Run tests with visualization
    # area, annulus_integrator = test_annulus_with_visualization()

    # Create animation frames
    print("\nCreating subdivision progression frames...")
    # create_animation_frames()
     

    cut_off = 5
    fig, ax = plt.subplots()
    ax.plot(phi_x_values, -area + 4/3 * cut_off**2 * (2*cut_off**2 - 3*mu + 3*phi_x_values**2), "o")
    # ax.plot(phi_x_values, -area, "o")
    plt.show()


