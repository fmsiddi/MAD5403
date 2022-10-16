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
# solution: [1   1   1]
#           [4  -1  -5]
#           [3  -2 -10]
pivot='complete'
# A = LU_factorization(A, pivot)
# print(LU)
# A, P = LU_factorization(A, pivot)
A, P, Q = LU_factorization(A, pivot)
# A = np.array([[1,1,1],[4,3,-1],[3,5,3]],'float')

def get_A_from_LU(LU,pivot=False,P=None,Q=None):  
    L = np.tril(LU)
    np.fill_diagonal(L,1)
    U = np.triu(LU)
    A = mat_mult(L,U)
    if pivot!= 'none':
        A = A[P]
        if pivot=='complete':
            A = A.T[Q].T
    print(A)
    return

get_A_from_LU(A,pivot,P,Q)

#%%

# Step 4:
def get_LU_vector(LU, lower_or_upper, row_or_col, i):
    if lower_or_upper == 'L':
        if row_or_col == 'row':
            v = LU[i].copy()
            v[i] = 1
            v[i+1:] = 0
        elif row_or_col == 'col':
            v = LU.T[i].copy()
            v[i] = 1
            v[:i] = 0
    elif lower_or_upper == 'U':
        if row_or_col == 'row':
            v = LU[i].copy()
            v[:i] = 0
        elif row_or_col == 'col':
            v = LU.T[i].copy()
            v[i+1:] = 0
    return v
    
def forward_sub(b, LU, orientation_method):
    n = b.shape[0]
    if orientation_method == 'row':
        y = np.ndarray(n)
        y[0] = b[0]
        for i in range(1,n):
            L_i = get_LU_vector(LU,'L','row',i)
            y[i] = (b[i]-sum(L_i[:i]*y[:i]))/L_i[i]
        return y
    if orientation_method == 'col':
        for j in range(n-1):
            L_j = get_LU_vector(LU,'L','col',j)
            b[j] = b[j]/L_j[j]
            b[j+1:] = b[j+1:]-(b[j]*L_j[j+1:])
        b[n-1] = b[n-1]
        return b
    
def backward_sub(b, LU, orientation_method):
    n = b.shape[0]
    if orientation_method == 'row':
        x = np.ndarray(n)
        x[n-1] = b[n-1]/get_LU_vector(LU,'U','row',n-1)[n-1]
        for i in reversed(range(n-1)):
            U_i = get_LU_vector(LU,'U','row',i)
            x[i] = (b[i]-sum(U_i[i+1:]*x[i+1:]))/LU[i,i]
        return x
    if orientation_method == 'col':
        for j in reversed(range(1,n)):
            U_j = get_LU_vector(LU,'U','col',j)
            b[j] = b[j]/U_j[j]
            b[:j] = b[:j]-(b[j]*U_j[:j])
        b[0] = b[0]/LU[0,0]
        return b

def solver(b, LU, orientation_method, pivot, P=None, Q=None):
    if pivot != 'none':
        b = b[P].copy()
    y = forward_sub(b, LU, orientation_method)
    x = backward_sub(y, LU, orientation_method)
    if pivot == 'complete':
        x = x[Q]
    return x

    
#%%
# test matrix from https://www.mathsisfun.com/algebra/systems-linear-equations-matrices.html
# A = np.array([[1,1,1],[0,2,5],[2,5,-1]],'float')
# b = np.array([6,-4,27],'float')
# solution: [5   3   -2]

# test matrix from https://www.geeksforgeeks.org/l-u-decomposition-system-linear-equations/
# A = np.array([[1,1,1],[4,3,-1],[3,5,3]],'float')
# b = np.array([1,6,4],'float')
# solution: [1   .5   -.5]

# A = np.array([[2,1,0],[-4,0,4],[2,5,10]],'float')
# b = np.array([3,0,17],'float')
# solution: [1   1   1]

orientation_method = 'row'

# pivot = 'none'
# A = LU_factorization(A, pivot)
# x = solver(b, A, orientation_method, pivot)

pivot = 'partial'
A, P = LU_factorization(A, pivot)
# b = b[P].copy()
# y = forward_sub(b, A, orientation_method)
# x = backward_sub(y, A, orientation_method)
x = solver(b, A, orientation_method, pivot, P)

A = np.array([[1,1,1],[0,2,5],[2,5,-1]],'float')
P = np.array([[0,0,1],[1,0,0],[0,1,0]],'float')
Q = np.array([[0,0,1],[1,0,0],[0,1,0]],'float')
test = mat_mult(mat_mult(P,A),Q)
b = np.array([6,-4,27],'float')
pivot = 'complete'
A, P, Q = LU_factorization(A, pivot)

# L = np.tril(A)
# np.fill_diagonal(L,1)
# U = np.triu(A)
# check = mat_mult(L,U)
# if pivot != 'none':
#     b = b[P].copy()
# y = forward_sub(b, A, orientation_method)
# x = backward_sub(y, A, orientation_method)
# if pivot == 'complete':
#     x = x[Q]
x = solver(b, A, orientation_method, pivot, P, Q)

print(x)
# [[0,1,2]]
# [1,2,0]
