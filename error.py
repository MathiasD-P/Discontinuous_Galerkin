#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Honours Thesis Assignment 2

Created on Tue Sep 17 15:00:11 2024

@author: mathiasdufresne-piche

Description: Analysis of error on DG.
"""

import DG
import numpy as np
from matplotlib import pyplot as plt

#%% Solving the linear advection equation for average flux    
A = 0.5
N = 2000
J = 41
order = 2
alpha = 1
bcs = lambda t : np.array([0,1.0]) # could be a function of time
ics = lambda x : 0.5 * (1 + np.tanh(250*(x - 20))) # function of space only

x = np.linspace(0, 40, J)
t = np.linspace(0, 10, N)

fe = DG.finite_elements(x, order)
problem = DG.galerkin(fe, A, alpha)
L = lambda u,t : problem.space_operator(u, bcs(t))

u = DG.RK4(L, ics(fe.elements), t)

#%%
DG.plot_u(fe.elements, t, u)
plt.ylim([-0.3,1.3])
plt.title("Discontinuous Galerkin $\gamma = 1$")
plt.xlabel("x")
plt.ylabel("u")
plt.savefig("figures/time_evol_gamma1.png", dpi=300)
plt.show()

#%% Solving the linear advection equation for upwind flux    
A = 0.5
N = 2000
J = 1000
order = 2
alpha = 0
bcs = lambda t : np.array([0,1.0]) # could be a function of time
ics = lambda x : 0.5 * (1 + np.tanh(0.5*(x - 20))) # function of space only

x = np.linspace(0, 40, J)
t = np.linspace(0, 10, N)

fe = DG.finite_elements(x, order)
problem = DG.galerkin(fe, A, alpha)
L = lambda u,t : problem.space_operator(u, bcs(t))

u = DG.RK4(L, ics(fe.elements), t)

#%%
DG.plot_u(fe.elements, t, u)
plt.ylim([-0.3,1.3])
plt.title("Discontinuous Galerkin $\gamma = 0$")
plt.xlabel("x")
plt.ylabel("u")
plt.savefig("figures/time_evol_gamma0.png", dpi=300)
plt.show()

#%% Error Analysis K = 2
size = np.arange(41, 100)
A = 0.5
N = 2000
J = 2 ** np.arange(2,10,1)
order = 2
alpha = 0
bcs = lambda t : np.array([0,1.0]) # could be a function of time
ics = lambda x : 0.5 * (1 + np.tanh(0.5*(x - 20))) # function of space only

t = np.linspace(0, 10, N)
exact_solution = lambda x,t : ics(x - A * t)
error = np.zeros(np.shape(J))

for j in range(0,len(J)):
    x = np.linspace(0, 40, J[j])
    fe = DG.finite_elements(x, order)
    problem = DG.galerkin(fe, A, alpha)
    L = lambda u,t : problem.space_operator(u, bcs(t))
    u = DG.RK4(L, ics(fe.elements), t)
    u = u[:,:,N-1]
    
    u_exact = exact_solution(fe.elements, t[N-1])
    
    error[j] = DG.L2(u, u_exact, fe.elements)

#%%
DOF = order * J
begin = 3
end = len(J)

lsqr = np.vstack([np.log10(DOF[begin:end]), np.ones(end - begin)]).T
fit = (np.linalg.lstsq(lsqr, np.log10(error[begin:end])))[0]

plt.plot(np.log10(DOF), np.log10(error), 'o')
plt.plot([(np.log10(error[0]) - fit[1]) / fit[0], (np.log10(error[len(J)-1]) - fit[1]) / fit[0]], [np.log10(error[0]), np.log10(error[len(J)-1])], 'r--')
plt.plot([np.log10(DOF[begin]), np.log10(DOF[end-1])], [np.log10(error[begin]), np.log10(error[end-1])], '*')
plt.legend(["RMS Error (logarithmic)", "Least Squares Fit (slope = -2.99)", "Fit Interval"])
plt.title("RMS Error for the Galerkin Scheme for $K=2$")
plt.xlabel("$\log_{10} DOF$")
plt.ylabel("$\log_{10} \epsilon_{RMS}$")
plt.ylim([-10,0])
plt.savefig("figures/error_K2.png", dpi=300)
plt.show()

#%% Error Analysis K = 3
size = np.arange(41, 100)
A = 0.5
N = 2000
J = 2 ** np.arange(2,10,1)
order = 3
alpha = 0
bcs = lambda t : np.array([0,1.0]) # could be a function of time
ics = lambda x : 0.5 * (1 + np.tanh(0.5*(x - 20))) # function of space only

t = np.linspace(0, 10, N)
exact_solution = lambda x,t : ics(x - A * t)
error = np.zeros(np.shape(J))

for j in range(0,len(J)):
    x = np.linspace(0, 40, J[j])
    fe = DG.finite_elements(x, order)
    problem = DG.galerkin(fe, A, alpha)
    L = lambda u,t : problem.space_operator(u, bcs(t))
    u = DG.RK4(L, ics(fe.elements), t)
    u = u[:,:,N-1]
    
    u_exact = exact_solution(fe.elements, t[N-1])
    
    error[j] = DG.L2(u, u_exact, fe.elements)

#%%
DOF = order * J
begin = 2
end = len(J)

lsqr = np.vstack([np.log10(DOF[begin:end]), np.ones(end - begin)]).T
fit = (np.linalg.lstsq(lsqr, np.log10(error[begin:end])))[0]

plt.plot(np.log10(DOF), np.log10(error), 'o')
plt.plot([(np.log10(error[0]) - fit[1]) / fit[0], (np.log10(error[len(J)-1]) - fit[1]) / fit[0]], [np.log10(error[0]), np.log10(error[len(J)-1])], 'r--')
plt.plot([np.log10(DOF[begin]), np.log10(DOF[end-1])], [np.log10(error[begin]), np.log10(error[end-1])], '*')
plt.legend(["RMS Error (logarithmic)", "Least Squares Fit (slope = -3.97)", "Fit Interval"])
plt.title("RMS Error for the Galerkin Scheme for $K=3$")
plt.xlabel("$\log_{10} DOF$")
plt.ylabel("$\log_{10} \epsilon_{RMS}$")
plt.ylim([-10,0])
plt.savefig("figures/error_K3.png", dpi=300)
plt.show()

#%% Error Analysis K = 4
size = np.arange(41, 100)
A = 0.5
N = 2000
J = 2 ** np.arange(2,10,1)
order = 4
alpha = 0
bcs = lambda t : np.array([0,1.0]) # could be a function of time
ics = lambda x : 0.5 * (1 + np.tanh(0.5*(x - 20))) # function of space only

t = np.linspace(0, 10, N)
exact_solution = lambda x,t : ics(x - A * t)
error = np.zeros(np.shape(J))

for j in range(0,len(J)):
    x = np.linspace(0, 40, J[j])
    fe = DG.finite_elements(x, order)
    problem = DG.galerkin(fe, A, alpha)
    L = lambda u,t : problem.space_operator(u, bcs(t))
    u = DG.RK4(L, ics(fe.elements), t)
    u = u[:,:,N-1]
    
    u_exact = exact_solution(fe.elements, t[N-1])
    
    error[j] = DG.L2(u, u_exact, fe.elements)

#%%
DOF = order * J
begin = 2
end = len(J)-2

lsqr = np.vstack([np.log10(DOF[begin:end]), np.ones(end - begin)]).T
fit = (np.linalg.lstsq(lsqr, np.log10(error[begin:end])))[0]

plt.plot(np.log10(DOF), np.log10(error), 'o')
plt.plot([(np.log10(error[0]) - fit[1]) / fit[0], (np.log10(error[len(J)-1]) - fit[1]) / fit[0]], [np.log10(error[0]), np.log10(error[len(J)-1])], 'r--')
plt.plot([np.log10(DOF[begin]), np.log10(DOF[end-1])], [np.log10(error[begin]), np.log10(error[end-1])], '*')
plt.legend(["RMS Error (logarithmic)", "Least Squares Fit (slope = -5.10)", "Fit Interval"])
plt.title("RMS Error for the Galerkin Scheme for $K=4$")
plt.xlabel("$\log_{10} DOF$")
plt.ylabel("$\log_{10} \epsilon_{RMS}$")
plt.ylim([-10,0])
plt.savefig("figures/error_K4.png", dpi=300)
plt.show()

#%% Error Analysis K = 5
size = np.arange(41, 100)
A = 0.5
N = 2000
J = 2 ** np.arange(2,10,1)
order = 5
alpha = 0
bcs = lambda t : np.array([0,1.0]) # could be a function of time
ics = lambda x : 0.5 * (1 + np.tanh(0.5*(x - 20))) # function of space only

t = np.linspace(0, 1, N)
exact_solution = lambda x,t : ics(x - A * t)
error = np.zeros(np.shape(J))

for j in range(0,len(J)):
    x = np.linspace(0, 40, J[j])
    fe = DG.finite_elements(x, order)
    problem = DG.galerkin(fe, A, alpha)
    L = lambda u,t : problem.space_operator(u, bcs(t))
    u = DG.RK4(L, ics(fe.elements), t)
    u = u[:,:,N-1]
    
    u_exact = exact_solution(fe.elements, t[N-1])
    
    error[j] = DG.L2(u, u_exact, fe.elements)

#%%
DOF = order * J
begin = 3
end = len(J)-3

lsqr = np.vstack([np.log10(DOF[begin:end]), np.ones(end - begin)]).T
fit = (np.linalg.lstsq(lsqr, np.log10(error[begin:end])))[0]

plt.plot(np.log10(DOF), np.log10(error), 'o')
plt.plot([(np.log10(error[0]) - fit[1]) / fit[0], (np.log10(error[len(J)-1]) - fit[1]) / fit[0]], [np.log10(error[0]), np.log10(error[len(J)-1])], 'r--')
plt.plot([np.log10(DOF[begin]), np.log10(DOF[end-1])], [np.log10(error[begin]), np.log10(error[end-1])], '*')
plt.legend(["RMS Error (logarithmic)", "Least Squares Fit (slope = -6.08)", "Fit Interval"])
plt.title("RMS Error for the Galerkin Scheme for $K=5$")
plt.xlabel("$\log_{10} DOF$")
plt.ylabel("$\log_{10} \epsilon_{RMS}$")
plt.ylim([-10,0])
plt.savefig("figures/error_K5.png", dpi=300)
plt.show()