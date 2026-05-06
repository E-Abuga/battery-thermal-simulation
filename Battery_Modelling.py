#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 10 17:17:40 2026

@author: christopherabuga
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D

# Geometry (meters)
Lx = 0.06     # width of the 2-D slice (x-direction)
Ly = 0.020    # height of the 2-D slice (y-direction)

# Grid resolution
nx = 81       # number of nodes in x
ny = 41       # number of nodes in y

dx = Lx/(nx-1)
dy = Ly/(ny-1)

# Material properties (roughly graphite/aluminum composite-like)
rho = 2200.0      # kg/m^3
cp  = 900.0       # J/kg-K
k   = 1.0         # W/m-K
alpha = k/(rho*cp) #

# Heat source and cooling
qdot  = 2.0e4     # W/m^3 (internal generation)
h     = 8.0       # W/m^2-K (convective cooling on all sides)
T_inf = 298.15    # K ambient
T0    = 298.15    # K initial

dt = 8.0783e-02
# dt = 1.1*1.7952e-01

def step_ftcs_convective(T, alpha, dx, dy, dt, h, k, T_inf, q_over_rhocp):
    """
    Advance one FTCS step with convective (Robin) boundaries on all sides,
    using array indexing as T[i, j] where:
        i = x-index (0..nx-1),  j = y-index (0..ny-1)
    and T.shape == (nx, ny).

    Interior update:
        T^{n+1}_{i,j} = T^{n}_{i,j}
            + rx*(T^{n}_{i+1,j} - 2 T^{n}_{i,j} + T^{n}_{i-1,j})
            + ry*(T^{n}_{i,j+1} - 2 T^{n}_{i,j} + T^{n}_{i,j-1})
            + q_over_rhocp*dt

    Convective boundaries use the standard ghost-node elimination that yields:
        left/right:  + 2*rx*(neighbor - boundary) - conv_x*(T_boundary - T_inf)
        bottom/top:  + 2*ry*(neighbor - boundary) - conv_y*(T_boundary - T_inf)
    """
    nx, ny = T.shape            # NOTE: shape is (nx, ny)
    Tn = T.copy()               # true copy (do not alias)

    ###########################################################################
    ### WARNING: Assigning Tn to T (as in Tn = T) does not create a new variable (Tn) in the memory ###
    ### It only creates a reference. Meaning, one variable, two names. Hence we use the .copy() ###
    ###########################################################################

    # Coefficients
    rx = alpha*dt/dx**2
    ry = alpha*dt/dy**2
    conv_x = 2*alpha*dt*h/(k*dx)    # convection factor in x
    conv_y = 2*alpha*dt*h/(k*dy)    # convection factor in y

    Tnew = Tn.copy()
    
    # ------------------
    # Interior (1..nx-2, 1..ny-2)
    # x-second-deriv uses axis 0 neighbors; y-second-deriv uses axis 1 neighbors
    
    # we can use this first or the for loop below
    #Tnew[1:-1, 1:-1] = (
        #Tn[1:-1, 1:-1]
        #+ rx*(Tn[2:,   1:-1] - 2*Tn[1:-1, 1:-1] + Tn[0:-2, 1:-1])
        #+ ry*(Tn[1:-1, 2:  ] - 2*Tn[1:-1, 1:-1] + Tn[1:-1, 0:-2])
        #+ dt*q_over_rhocp
   # )

    for i in range(1, nx-1):
        for j in range(1, ny-1):
            Tnew[i, j] = (
                Tn[i, j]
                + rx * (Tn[i+1, j] - 2*Tn[i, j] + Tn[i-1, j])
                + ry * (Tn[i, j+1] - 2*Tn[i, j] + Tn[i, j-1])
                + q_over_rhocp * dt
            )
            
    #Tnew[0, 1:-1] = (
        #Tn[0, 1:-1]
        #+ 2*rx*(Tn[1, 1:-1] - Tn[0, 1:-1])      # x ghost-node elimination
        #- conv_x*(Tn[0, 1:-1] - T_inf)
        #+ ry*(Tn[0, 2:  ] - 2*Tn[0, 1:-1] + Tn[0, 0:-2])
        #+ dt*q_over_rhocp
    #)
    
    # Left Boundary (i = 0)
    for j in range(1, ny-1):
        Tnew[0, j] = Tn[0, j] + 2*rx*(Tn[1, j] - Tn[0, j]) - \
            conv_x*(Tn[0, j] - T_inf) + ry*(Tn[0, j+1] - 2*Tn[0, j] + Tn[0, j-1]) \
            + q_over_rhocp*dt
            
    # Right Boundary (x = Ly)
    for j in range(1, ny-1):
        Tnew[-1, j] = (
            Tn[-1, j]
            + 2*rx * (Tn[-2, j] - Tn[-1, j])
            - conv_x * (Tn[-1, j] - T_inf)
            + ry * (Tn[-1, j+1] - 2*Tn[-1, j] + Tn[-1, j-1])
            + q_over_rhocp * dt
        )


    # Bottom boundary (j = 0)
    for i in range(1, nx-1):
        Tnew[i, 0] = (
            Tn[i, 0]
            + rx * (Tn[i+1, 0] - 2*Tn[i, 0] + Tn[i-1, 0])
            + 2*ry * (Tn[i, 1] - Tn[i, 0])
            - conv_y * (Tn[i, 0] - T_inf)
            + q_over_rhocp * dt
        )

    # Top boundary (J = Ly)
    for i in range(1, nx-1):
        Tnew[i, -1] = (
            Tn[i, -1]
            + rx * (Tn[i+1, -1] - 2*Tn[i, -1] + Tn[i-1, -1])
            + 2*ry * (Tn[i, -2] - Tn[i, -1])
            - conv_y * (Tn[i, -1] - T_inf)
            + q_over_rhocp * dt
        )

    # ------------------
    # Corners (apply both x & y boundary logic)
    
    # bottom-left (i=0, j=0)
    Tnew[0, 0] = (
        Tn[0, 0]
        + 2*rx*(Tn[1, 0] - Tn[0, 0]) - conv_x*(Tn[0, 0] - T_inf)
        + 2*ry*(Tn[0, 1] - Tn[0, 0]) - conv_y*(Tn[0, 0] - T_inf)
        + dt*q_over_rhocp
    )
    
    # bottom-right (i=nx-1, j=0)
    Tnew[-1, 0] = (
        Tn[-1, 0]
        + 2*rx*(Tn[-2, 0] - Tn[-1, 0]) - conv_x*(Tn[-1, 0] - T_inf)
        + 2*ry*(Tn[-1, 1] - Tn[-1, 0]) - conv_y*(Tn[-1, 0] - T_inf)
        + dt*q_over_rhocp
    )
    
    # top-left (i=0, j=ny-1)
    Tnew[0, -1] = (
        Tn[0, -1]
        + 2*rx*(Tn[1, -1] - Tn[0, -1]) - conv_x*(Tn[0, -1] - T_inf)
        + 2*ry*(Tn[0, -2] - Tn[0, -1]) - conv_y*(Tn[0, -1] - T_inf)
        + dt*q_over_rhocp
    )
    
    # top-right (i=nx-1, j=ny-1)
    Tnew[-1, -1] = (
        Tn[-1, -1]
        + 2*rx*(Tn[-2, -1] - Tn[-1, -1]) - conv_x*(Tn[-1, -1] - T_inf)
        + 2*ry*(Tn[-1, -2] - Tn[-1, -1]) - conv_y*(Tn[-1, -1] - T_inf)
        + dt*q_over_rhocp
    )
    
    return Tnew

T = np.full((ny, nx), T0, dtype=float)
t_end = 50.0              # total simulated time (s)
# Prep time integration
q_over_rhocp = qdot/(rho*cp)
nsteps = int(np.ceil(t_end/dt))
save_times = np.linspace(0, t_end, 5)         # 5 snapshots including t=0 and t=t_end
save_ids = set(np.round(save_times/dt).astype(int))

Ts = []
times = []

for n in range(nsteps+1):
    if n in save_ids:
        Ts.append(T.copy()); times.append(n*dt)
    if n == nsteps:
        break
    T = step_ftcs_convective(T, alpha, dx, dy, dt, h, k, T_inf, q_over_rhocp)

print(f"Saved snapshots at: {[f'{t:.2f}' for t in times]} s")

# Contour plots
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)

for t, Tplot in zip(times, Ts):
    plt.figure(figsize=(6,3))
    cs = plt.contourf(x, y, Tplot, levels=20)
    plt.colorbar(cs, label='T [K]')
    plt.xlabel('x [m]'); plt.ylabel('y [m]')
    plt.title(f'Temperature field at t = {t:.2f} s')
    plt.tight_layout()
    plt.show()
