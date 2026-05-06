# battery-thermal-simulation
Battery Thermal Management — 2D Finite Difference Simulation

This project simulates the transient temperature distribution across a 2D lithium-ion battery cross-section using the Forward Time Centered Space (FTCS) explicit finite difference method, implemented from scratch in Python.
The simulation solves the 2D heat conduction equation with internal heat generation (representing battery discharge) and convective cooling on all four boundaries (Newton's law of cooling). The numerical grid consists of 81×41 nodes across a 6cm × 2cm cross-section, with the time integration running over 50 seconds using a stability-constrained time step.
Key concepts implemented:

Finite difference discretisation of a 2D PDE
Robin (convective) boundary conditions with ghost node elimination
Explicit time-stepping with CFL stability constraint
NumPy vectorised array operations for computational efficiency
Matplotlib contour visualisation of the evolving temperature field

Tools & Libraries: Python, NumPy, Matplotlib
