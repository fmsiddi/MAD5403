#%%
from tqdm import tqdm
import numpy as np
import random as rnd

# Step 1:
def generate_L(n):
    L = np.zeros((n,n))
    np.fill_diagonal(L,1)
    for i in range(1,n):
        for j in range(i):
            L[i,j] = rnd.random()
    return L

def generate_U(n):
    L = generate_L(n)
    np.fill_diagonal(L,rnd.choices(range(2,5), k=n))
    # np.fill_diagonal(L,np.diagonal(L)+1)
    U = L.T
    return U

def mat_mult(L,U):
    if len(U.shape) > 1:
        A = np.zeros((L.shape[0],U.shape[1]))
        for i in range(L.shape[0]):
            for j in range(U.shape[1]):
                A[i,j] = sum(L[i] * U.T[j])
    else:
        A = np.zeros(L.shape[0])
        for i in range(L.shape[1]):
            A[i] = sum(L[i] * U)
    return A

n = 10
L = generate_L(n)
U = generate_U(n)

A = mat_mult(L,U)
# np.fill_diagonal(A,np.diagonal(A)+5) # making diagonally dominant

#%%

# Step 2:
def generate_b(A,x):
    b = mat_mult(A,x)
    return b

x = np.zeros(n)
for i in range(n):
    x[i] = rnd.random()
    
b = generate_b(A,x)

# testing multiplying a matrix by a vector
# A = np.array([[4,3],[6,3]])
# x = np.array([3,3])
# b = generate_b(A,x)

#%%

# Step 3:
def index_flip(array,i,j):
    out = array
    out[[i,j]] = out[[j,i]]
    return out

def LU_factorization(A, pivot='none'):
    n = A.shape[0]
    if pivot == 'none':
        for k in range(n-1):
            if A[k,k] == 0:
                print('Error: diagonal element of 0 encountered')
                print('Try partial pivoting')
                break
            A[k+1:, k] = A[k+1:, k]/A[k,k]
            for j in range(k+1, n):
                for i in range(k+1, n):
                    A[i,j] = A[i,j] - A[i,k]*A[k,j]
        return A
    if pivot == 'partial':
        P = np.arange(n)
        for k in range(n-1):
            p_index = abs(A[k:,k]).argmax() + k
            if p_index - k > 0:
                P[[k,p_index]] = P[[p_index,k]]
                A = A[index_flip(np.arange(n),k,p_index)]
            if A[k,k] == 0:
                print('Error: diagonal element of 0 encountered')
                print('Try complete pivoting')
                break
            A[k+1:, k] = A[k+1:, k]/A[k,k]
            for j in range(k+1, n):
                for i in range(k+1, n):
                    A[i,j] = A[i,j] - A[i,k]*A[k,j]
            P = np.array([list(P).index(i) for i in range(n)])
        return A, P
    if pivot == 'complete':
        P = np.arange(n)
        Q = np.arange(n)
        for k in range(n-1):
            p_index = abs(A[k:,k:]).argmax() // A[k:,k:].shape[0] + k
            q_index = abs(A[k:,k:].T).argmax() // A[k:,k:].shape[1] + k
            if p_index - k > 0:
                P[[k,p_index]] = P[[p_index,k]]
                A = A[index_flip(np.arange(n),k,p_index)]
            if q_index - k > 0:
                Q[[k,q_index]] = Q[[q_index,k]]
                A = A.T[index_flip(np.arange(n),k,q_index)].T
            if A[k,k] == 0:
                print('Error: matrix is singular, LU factorization does not exist')
                break
            A[k+1:, k] = A[k+1:, k]/A[k,k]
            for j in range(k+1, n):
                for i in range(k+1, n):
                    A[i,j] = A[i,j] - A[i,k]*A[k,j]
            P = np.array([list(P).index(i) for i in range(n)])
            Q = np.array([list(Q).index(i) for i in range(n)])
        return A, P, Q

#%%
# test matrix from https://www.geeksforgeeks.org/l-u-decomposition-system-linear-equations/
A = np.array([[1,1,1],[4,3,-1],[3,5,3]],'float')
# solution: [  1   1   1]
#           [  4  -1  -5]
#           [  3  -2 -10]
# LU = LU_factorization(A, pivot='none')
# print(LU)
# LU, P = LU_factorization(A, pivot='partial')
LU, P, Q = LU_factorization(A, pivot='complete')
A = np.array([[1,1,1],[4,3,-1],[3,5,3]],'float')

def get_A_from_LU(LU,P=None,Q=None):  
    L = np.tril(LU)
    np.fill_diagonal(L,1)
    U = np.triu(LU)
    A = mat_mult(L,U)
    if P.any() != None:
        A = A[P]
    if Q.any() != None:
        A = A.T[Q].T
    print(A)
    return

get_A_from_LU(LU,P,Q)

#%%

# Step 4:
def solver(b, LU, orientation_method, P=None, Q=None):
    x = None
    return x
    