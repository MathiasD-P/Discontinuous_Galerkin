#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Honours Thesis Assignment 2

Created on Tue Sep 17 15:00:11 2024

@author: mathiasdufresne-piche

Description: Galerkin method implementation
"""
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt

class finite_elements:
    def __init__(self, mesh, order):
        self.N_el = len(mesh) - 1
        self.order = order
        
        ri = np.zeros((order+1,))
        ri[0] = -1
        ri[order] = 1
        ri[1:order] = sp.special.roots_jacobi(order-1,1,1)[0]
        self.ri = ri
        self.h = mesh[1:self.N_el+1] - mesh[0:self.N_el]
        self.xl = mesh[0:self.N_el]
        
        Ri = (np.tile(ri, (self.N_el, 1))).T
        Xl = np.tile(self.xl, (order+1, 1))
        H = np.tile(self.h, (order+1, 1))
        elements = Xl + (1 + Ri) / 2 * H
        self.elements = elements
    
    def vandermonde(self):
        V = np.polynomial.legendre.legvander(self.ri, self.order)
        N = np.linspace(0, self.order, self.order+1)
        N = np.tile(N, (self.order+1,1))
        norm = np.sqrt(2 / (2 * N + 1))
        return V / norm
    
    def differentiation(self):
        Vr = np.zeros((self.order+1, self.order+1))
        for j in range(1,self.order+1):
            norm = np.sqrt(8 / (2 * (j-1) + 3) * (sp.special.gamma(j+1) ** 2) / sp.special.gamma(j+2) / sp.special.factorial(j-1))
            Vr[:,j] = np.sqrt(j * (j + 1)) * sp.special.jacobi(j-1,1,1)(self.ri) / norm
        return Vr
    
    def operators(self):
        V = self.vandermonde()
        V_inv = np.linalg.inv(V)
        Vr = self.differentiation()
        Dr = Vr @ V_inv
        M_inv = V @ V.T
        return M_inv, Dr

class galerkin:
    def __init__(self, fe, A, alpha):
        self.finite_elements = fe
        self.A = A
        self.alpha = alpha
        self.M_inv, self.Dr = fe.operators()
    
    def space_operator(self, u, bcs):
        H = np.tile(self.finite_elements.h, (self.finite_elements.order+1,1))
        
        u_l = np.zeros((self.finite_elements.N_el+1,))
        u_r = np.zeros((self.finite_elements.N_el+1,))
        u_l[self.finite_elements.N_el] = bcs[1]
        u_r[0] = bcs[0]
        u_l[0:self.finite_elements.N_el] = u[0,:]
        u_r[1:self.finite_elements.N_el+1] = u[self.finite_elements.order,:]
        
        flux1 = -self.A * u_l[0:self.finite_elements.N_el] \
                + 0.5 * self.A * (u_l[0:self.finite_elements.N_el] + u_r[0:self.finite_elements.N_el]) \
                + 0.5 * abs(self.A) * (1 - self.alpha) * (-u_l[0:self.finite_elements.N_el] + u_r[0:self.finite_elements.N_el])
        flux2 = self.A * u_r[1:self.finite_elements.N_el+1] \
                - 0.5 * self.A * (u_r[1:self.finite_elements.N_el+1] + u_l[1:self.finite_elements.N_el+1]) \
                - 0.5 * abs(self.A) * (1 - self.alpha) * (u_r[1:self.finite_elements.N_el+1] - u_l[1:self.finite_elements.N_el+1])
        flux = np.zeros((self.finite_elements.order+1, self.finite_elements.N_el))
        flux[0,:] = flux1
        flux[self.finite_elements.order,:] = flux2
        
        u_dot = 2 / H * (-self.A * self.Dr @ u + self.M_inv @ flux)
        
        return u_dot

def RK4(F, X, t): # for a matrix differential equation
    N = len(t)
    h = t[1:N] - t[0:N-1] # stepsize
    Y = np.copy(X) # initialize point
    sol = np.zeros(np.shape(X) + (N,))
    sol[:,:,0] = X
    
    for i in range(1, N):
        k1 = h[i-1] * F(Y, t[i])
        k2 = h[i-1] * F(Y + k1 / 2, t[i] + h[i-1] / 2)
        k3 = h[i-1] * F(Y + k2 / 2, t[i] + h[i-1] / 2)
        k4 = h[i-1] * F(Y + k3, t[i] + h[i-1])
        
        Y = Y + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        sol[:,:,i] = Y
    
    return sol

def L2(u, u_exact, problem, order):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    
    Xl = np.tile(problem.finite_elements.xl, (order,1))
    H = np.tile(problem.finite_elements.h, (order, 1))
    Ri = np.tile(nodes, (problem.finite_elements.N_el, 1)).T
    points = Xl + (1 + Ri) / 2 * H
    
    u_samp = np.zeros(np.shape(points))
    for i in range(0, problem.finite_elements.N_el):
        u_pol = sp.interpolate.lagrange(problem.finite_elements.ri, u[:,i])
        u_samp[:,i] = u_pol(nodes)
        
    weights = np.tile(weights, (problem.finite_elements.N_el, 1)).T
    L2_error = 0.5 * weights * H * (u_samp - u_exact(points)) ** 2
    L2_error = np.sqrt(np.sum(L2_error))
    
    return L2_error

def plot_u(x, t, u):
    color_map = plt.get_cmap('jet', len(t)-1)
    for i in range(0,len(t)):
        plt.plot(np.ravel(x.T), np.ravel(u[:,:,i].T), color = color_map(i))
