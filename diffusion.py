#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Honours Thesis Assignment 2

Created on Tue Sep 17 15:00:11 2024

@author: mathiasdufresne-piche

Description: Analysis of dissipation and dispersion.
"""

import DG
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt

def Q_mat(problem, theta):
    order = problem.finite_elements.order
    h = problem.finite_elements.h[0]
    A = problem.A
    Dr = problem.Dr
    M_inv = problem.M_inv
    
    e0 = np.zeros((order+1,1))
    e0[0] = 1
    eN = np.zeros((order+1,1))
    eN[order] = 1
    
    Q = (-2 * A * 1j / h) * (Dr + M_inv @ e0 @ (e0.T - np.exp(-1j * theta) * eN.T))
    
    return Q

def diffusion(theta_delta_range, problem):
    order = problem.finite_elements.order
    vand_inv = np.linalg.inv(problem.finite_elements.vandermonde()) # Please forgive me...

    theta_out = np.zeros((order+1, len(theta_delta_range)))
    omega_out = np.zeros((order+1, len(theta_delta_range))) + 1j

    for i in range(0, len(theta_delta_range)):
        theta_delta = theta_delta_range[i]
        
        Q = Q_mat(problem, theta_delta)
        omega, V = np.linalg.eig(Q)
        leg_modes = vand_inv @ V
        leg_energy = abs(leg_modes) ** 2
        total_energy = np.sum(leg_energy, axis=0)
        leg_energy = leg_energy / np.tile(total_energy, (order+1, 1))
        
        theta_possible = np.arange(-(order+1), order+2) * 2 * np.pi + theta_delta
        theta = theta_possible[abs(theta_possible) <= np.pi * (order+1)]
        theta = sorted(theta, key = abs)
        
        for j in range(0,order+1):
            pol, mode = np.unravel_index(leg_energy.argmax(), leg_energy.shape)
            omega_out[j,i] = omega[mode]
            theta_out[j,i] = theta[pol]
            leg_energy[pol,:] = -1
            leg_energy[:,mode] = -1

    theta_out = np.ravel(theta_out)
    omega_out = np.ravel(omega_out)
    j = np.unravel_index(np.argsort(theta_out), theta_out.shape)
    theta_out = theta_out[j]
    omega_out = omega_out[j]
    
    return theta_out, omega_out

#%% TEST
A = 1
N = 2000
J = 41
order = 10
alpha = 0
theta_delta = 1
x = np.array([-0.5,0.5])

fe = DG.finite_elements(x, order)
problem = DG.galerkin(fe, A, alpha)
vand_inv = np.linalg.inv(problem.finite_elements.vandermonde())


Q = Q_mat(problem, theta_delta)
omega, V = np.linalg.eig(Q)
plt.plot(fe.ri, V)
plt.show()
leg_modes = vand_inv @ V
leg_energy = abs(leg_modes) ** 2
total_energy = np.sum(leg_energy, axis=0)
leg_energy = leg_energy / np.tile(total_energy, (order+1, 1))
theta_possible = np.arange(-(order+1), order+2) * 2 * np.pi
theta = theta_possible[abs(theta_possible) <= np.pi * (order+1)]
theta = sorted(theta, key = abs)

X = np.linspace(-0.5,0.5,order+1)
V_new = np.polynomial.legendre.legvander(X, order) @ leg_modes
fourier = np.fft.fft(V_new,axis=0)
fourier = abs(fourier) ** 2 / np.sum(abs(fourier)**2, axis=0)
freq = np.fft.fftfreq(X.shape[-1], 1 / (order-1))
plt.plot(freq, fourier, '.')
plt.ylim([0,2])
plt.show()

#%%

A = 1
N = 2000
J = 41
order = np.arange(2,6)
alpha = 0
theta_delta_range = np.linspace(-np.pi, np.pi, 21)

x = np.array([-0.5,0.5])
theta = []
omega = []

for i in range(0, len(order)):
    fe = DG.finite_elements(x, order[i])
    problem = DG.galerkin(fe, A, alpha)
    theta_add, omega_add = diffusion(theta_delta_range, problem)
    theta.append(theta_add)
    omega.append(omega_add)

# plot results
color_map = plt.get_cmap('jet', len(order))

for i in range(len(order)-1,-1,-1):
    plt.plot(abs(theta[i]) / (order[i]+1), abs(np.real(omega[i])) / (order[i]+1), color=color_map(i))
plt.plot([0,np.pi],[0,np.pi],'k--')
plt.ylabel("$Re(\omega) / (K+1)$")
plt.xlabel("$\eta / (K+1)$")
plt.show()

for i in range(len(order)-1,-1,-1):
    plt.plot(abs(theta[i]) / (order[i]+1), np.imag(omega[i]) / (order[i]+1), color=color_map(i))
plt.plot([0,np.pi],[0,0],'k--')
plt.ylabel("$Im(\omega) / (K+1)$")
plt.xlabel("$\eta / (K+1)$")
plt.show()