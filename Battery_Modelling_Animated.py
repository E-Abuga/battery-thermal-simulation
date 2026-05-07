#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 06 17:46:30 2026

@author: christopherabuga

Battery Thermal Simulation — 2D Finite Difference Method (FTCS)

Simulates transient heat conduction across a 2D lithium-ion battery
cross-section with internal heat generation and convective cooling
on all four boundaries.

Outputs:
    - 5 static contour snapshots
    - animated temperature evolution
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize

# Geometry (meters)
Lx = 0.06     # width  (x-direction)
Ly = 0.020    # height (y-direction)

# Grid resolution
nx = 81
ny = 41

dx = Lx / (nx - 1)
dy = Ly / (ny - 1)

# Material properties (graphite/aluminium composite-like)

rho   = 2200.0        # kg/m³
cp    = 900.0         # J/kg·K
k     = 1.0           # W/m·K
alpha = k / (rho * cp)

# Operating conditions

qdot  = 2.0e4    # W/m³  internal heat generation
h     = 8.0      # W/m²·K convective cooling coefficient
T_inf = 298.15   # K  ambient temperature
T0    = 298.15   # K  initial temperature

# Stable time step (0.45 safety factor)

dt_stable = 1.0 / (2 * alpha * (1/dx**2 + 1/dy**2))
dt = 0.45 * dt_stable

print(f"Stable dt = {dt_stable:.5f} s  →  using dt = {dt:.5f} s")


def step_ftcs_convective(T, alpha, dx, dy, dt, h, k, T_inf, q_over_rhocp):
    """
    Advance the temperature field one FTCS time step.
    Convective (Robin) boundary conditions on all four edges.
    T.shape == (ny, nx)
    """
    ny, nx = T.shape
    Tn   = T.copy()
    Tnew = T.copy()

    rx     = alpha * dt / dx**2
    ry     = alpha * dt / dy**2
    conv_x = 2 * alpha * dt * h / (k * dx)
    conv_y = 2 * alpha * dt * h / (k * dy)

    # Interior (vectorised)
    Tnew[1:-1, 1:-1] = (
        Tn[1:-1, 1:-1]
        + rx * (Tn[1:-1, 2:]   - 2*Tn[1:-1, 1:-1] + Tn[1:-1, 0:-2])
        + ry * (Tn[2:,   1:-1] - 2*Tn[1:-1, 1:-1] + Tn[0:-2, 1:-1])
        + dt  * q_over_rhocp
    )

    # Left boundary  (j = 0)
    Tnew[1:-1, 0] = (
        Tn[1:-1, 0]
        + 2*rx * (Tn[1:-1, 1] - Tn[1:-1, 0])
        - conv_x * (Tn[1:-1, 0] - T_inf)
        + ry * (Tn[2:, 0] - 2*Tn[1:-1, 0] + Tn[0:-2, 0])
        + dt * q_over_rhocp
    )

    # Right boundary (j = nx-1)
    Tnew[1:-1, -1] = (
        Tn[1:-1, -1]
        + 2*rx * (Tn[1:-1, -2] - Tn[1:-1, -1])
        - conv_x * (Tn[1:-1, -1] - T_inf)
        + ry * (Tn[2:, -1] - 2*Tn[1:-1, -1] + Tn[0:-2, -1])
        + dt * q_over_rhocp
    )

    # Bottom boundary (i = 0)
    Tnew[0, 1:-1] = (
        Tn[0, 1:-1]
        + rx * (Tn[0, 2:] - 2*Tn[0, 1:-1] + Tn[0, 0:-2])
        + 2*ry * (Tn[1, 1:-1] - Tn[0, 1:-1])
        - conv_y * (Tn[0, 1:-1] - T_inf)
        + dt * q_over_rhocp
    )

    # Top boundary (i = ny-1)
    Tnew[-1, 1:-1] = (
        Tn[-1, 1:-1]
        + rx * (Tn[-1, 2:] - 2*Tn[-1, 1:-1] + Tn[-1, 0:-2])
        + 2*ry * (Tn[-2, 1:-1] - Tn[-1, 1:-1])
        - conv_y * (Tn[-1, 1:-1] - T_inf)
        + dt * q_over_rhocp
    )

    # Corners
    Tnew[0, 0] = (
        Tn[0, 0]
        + 2*rx*(Tn[0, 1]  - Tn[0, 0])  - conv_x*(Tn[0, 0]  - T_inf)
        + 2*ry*(Tn[1, 0]  - Tn[0, 0])  - conv_y*(Tn[0, 0]  - T_inf)
        + dt * q_over_rhocp
    )
    Tnew[0, -1] = (
        Tn[0, -1]
        + 2*rx*(Tn[0, -2] - Tn[0, -1]) - conv_x*(Tn[0, -1] - T_inf)
        + 2*ry*(Tn[1, -1] - Tn[0, -1]) - conv_y*(Tn[0, -1] - T_inf)
        + dt * q_over_rhocp
    )
    Tnew[-1, 0] = (
        Tn[-1, 0]
        + 2*rx*(Tn[-1, 1] - Tn[-1, 0]) - conv_x*(Tn[-1, 0] - T_inf)
        + 2*ry*(Tn[-2, 0] - Tn[-1, 0]) - conv_y*(Tn[-1, 0] - T_inf)
        + dt * q_over_rhocp
    )
    Tnew[-1, -1] = (
        Tn[-1, -1]
        + 2*rx*(Tn[-1, -2] - Tn[-1, -1]) - conv_x*(Tn[-1, -1] - T_inf)
        + 2*ry*(Tn[-2, -1] - Tn[-1, -1]) - conv_y*(Tn[-1, -1] - T_inf)
        + dt * q_over_rhocp
    )

    return Tnew



# Run simulation — collect frames for animation

T             = np.full((ny, nx), T0, dtype=float)
t_end         = 50.0
q_over_rhocp  = qdot / (rho * cp)
nsteps        = int(np.ceil(t_end / dt))

# Save every N steps for the animation (target ~60 frames)
anim_every    = max(1, nsteps // 60)

# Also save 5 static snapshots
save_times    = np.linspace(0, t_end, 5)
save_ids      = set(np.round(save_times / dt).astype(int))

frames        = []   # (time, T_snapshot) for animation
Ts_static     = []   # for static plots
times_static  = []

print(f"Running {nsteps} steps...")

for n in range(nsteps + 1):
    # Collect animation frame
    if n % anim_every == 0:
        frames.append((n * dt, T.copy()))

    # Collect static snapshot
    if n in save_ids:
        Ts_static.append(T.copy())
        times_static.append(n * dt)

    if n == nsteps:
        break

    T = step_ftcs_convective(T, alpha, dx, dy, dt, h, k, T_inf, q_over_rhocp)

print(f"Collected {len(frames)} animation frames.")
print(f"Static snapshots at: {[f'{t:.1f}s' for t in times_static]}")

# Static contour plots (5 snapshots)

x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)

for t, Tplot in zip(times_static, Ts_static):
    plt.figure(figsize=(7, 3))
    cs = plt.contourf(x, y, Tplot, levels=20, cmap='hot')
    plt.colorbar(cs, label='Temperature [K]')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title(f'Battery Temperature Field  —  t = {t:.1f} s')
    plt.tight_layout()
    plt.show()

# Build & save animated GIF

T_min = T0
T_max = max(f[1].max() for f in frames)

fig, ax = plt.subplots(figsize=(7, 3))
cf = ax.contourf(x, y, frames[0][1], levels=20, cmap='hot',
                 vmin=T_min, vmax=T_max)
cbar = fig.colorbar(cf, ax=ax, label='Temperature [K]')
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
title = ax.set_title(f't = {frames[0][0]:.2f} s')
plt.tight_layout()

def update(frame_idx):
    t_frame, T_frame = frames[frame_idx]
    ax.clear()
    ax.contourf(x, y, T_frame, levels=20, cmap='hot',
                vmin=T_min, vmax=T_max)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title(f'Battery Temperature Field  —  t = {t_frame:.1f} s')
    return ax,

ani = animation.FuncAnimation(
    fig,
    update,
    frames=len(frames),
    interval=80,
    blit=False
)

gif_path = 'battery_thermal.gif'
ani.save(gif_path, writer='pillow', fps=12, dpi=100)
print(f"\nAnimation saved → {gif_path}")
plt.close()