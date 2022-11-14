# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 17:36:16 2022

@author: farez
"""
import numpy as np

# A = np.array([[3,7,-1],[7,4,1],[-1,1,2]])

A = np.array([[7,4,1],[3,7,-1],[-1,1,2]],'float')
A_copy = A.copy()
P = np.tril(A)
N = -np.triu(A)
np.fill_diagonal(N,0)
P_inv = np.linalg.inv(P)
B = np.matmul(P_inv,N)
np.sqrt(np.linalg.eigvals(B).real**2 + np.linalg.eigvals(B).imag**2)

P = np.array([[np.diag(A)[0],0,0],[0,np.diag(A)[1],0],[0,0,np.diag(A)[2]]])
np.fill_diagonal(A_copy,0)
N = -A_copy
B = np.matmul(P_inv,N)
np.sqrt(np.linalg.eigvals(B).real**2 + np.linalg.eigvals(B).imag**2)

b = np.array([2,3,1],'float')
f = np.matmul(P_inv,b)

x_prev = np.array([1,1,1],'float')
x_new = np.array([100,100,100],'float')
i = 0
while abs(sum(x_new-x_prev))>.000000000000000000001:
    i += 1
    x_prev = x_new
    x_new = np.matmul(B,x_prev) + f
    
    
np.matmul(A,x_new)

# A = np.array([[7,3,-1],[4,7,1],[1,-1,2]])
np.linalg.eigvals(B)
