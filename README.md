# Discontinuous_Galerkin
This is a practice assignment performed in the scope of my honours thesis in the McGill Fluids Aerodynamics Group.

DG:
This module implements the strong form of the discontinuous Galerkin method in 1D to discretize the space component of the linear advection differential operator. The time component is solved using explicit RK4. The basis functions used are the Lagrange polynomials taken at the Gauss-Lobatto points. The form of the flux used can be specified by the user.

error:
An error analysis is performed for schemes of different orders and two types of flux are tested (central and pure upwind).

diffusion:
A diffusion error analysis is performed for schemes of different orders. Dissipation and dispersion errors are characterized.
