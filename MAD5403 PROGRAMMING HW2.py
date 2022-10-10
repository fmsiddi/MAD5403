from tqdm import tqdm
import numpy as np
import random as rnd

def generate_L(n):
    L = np.zeros((n,n))
    np.fill_diagonal(L,1)
    for i in range(1,n):
        for j in range(i):
            L[i,j] = rnd.random()
    return L

def generate_U(n):
    L = generate_L(10)
    U = L.T
    return U

L = generate_L(10)
U = generate_U(10)

def mat_mult(L,U):
    A = np.zeros((L.shape[0],U.shape[1]))
    for i in range(L.shape[0]):
        for j in range(U.shape[1]):
            A[i,j] = sum(L[i] * U.T[j])
    return A

L = np.array([[1,2],[3,4]])
U = np.array([[2,3],[4,5]])

A = mat_mult(L,U)