#%%
from tqdm import tqdm
import numpy as np
import random as rnd
import matplotlib.pyplot as plt
plt.style.use('seaborn')

# Step 1:
def generate_L(n):
    L = np.zeros((n,n))
    np.fill_diagonal(L,1) # UNITARY DIAGONAL
    for i in range(1,n):
        for j in range(i):
            L[i,j] = rnd.choice([1,-1]) * rnd.random() # rnd.random() ONLY GENERATES FLOATS < |1|
    return L

def generate_positive_L(n):
    L = np.zeros((n,n))
    np.fill_diagonal(L,1) # UNITARY DIAGONAL
    for i in range(1,n):
        for j in range(i):
            L[i,j] = rnd.random() # rnd.random() ONLY GENERATES FLOATS < |1|
    return L

def generate_U(n):
    L = generate_L(n)
    np.fill_diagonal(L,rnd.choices(range(2,5), k=n)) # ENSURES DIAGONAL ELEMENTS AREN'T TOO SMALL COMPARED TO NON-ZERO ELEMENTS
    U = L.T
    return U

#%%

# Step 3:
def mat_mult(A,B): # MATRIX MULTIPLICATION ROUTINE
    if len(B.shape) > 1:
        M = np.zeros((A.shape[0],B.shape[1]))
        for i in range(A.shape[0]):
            for j in range(B.shape[1]):
                M[i,j] = sum(A[i] * B.T[j])
    else:
        M = np.zeros(A.shape[0])
        for i in range(A.shape[1]):
            M[i] = sum(A[i] * B)
    return M

def index_flip(array,i,j): # TAKES ARRAY AND SWAPS iTH ELEMENT WITH jTH ELEMENT
    out = array
    out[[i,j]] = out[[j,i]]
    return out

def LU_factorization(A, pivot='none'): # FACTORIZATION TAKES MATRIX A AND PIVOT METHOD
    n = A.shape[0] # EXTRACT NUMBER OF ROWS
    if pivot == 'none': # WITHOUT PIVOTING
        for k in range(n-1):
            if A[k,k] == 0: # EXIT ROUTINE IF DIAGONAL ELEMENT IS 0
                print('Error: diagonal element of 0 encountered')
                print('Try partial pivoting')
                break
            A[k+1:, k] = A[k+1:, k]/A[k,k] # COMPUTE MULTIPLIERS IN PLACE
            for j in range(k+1, n):
                for i in range(k+1, n):
                    A[i,j] = A[i,j] - A[i,k]*A[k,j] # COMPUTE REMAINING ENTRIES OF SUBMATRIX
        return A # RETURN COMPOSITE LU MATRIX
    
    if pivot == 'partial': # WITH PARTIAL PIVOTING
        P = np.arange(n) # INITIALIZE ROW PERMUTATION ARRAY
        for k in range(n-1):
            p_index = abs(A[k:,k]).argmax() + k # FIND INDEX OF ROW WITH LARGEST PIVOT VALUE
            if p_index - k > 0:
                P[[k,p_index]] = P[[p_index,k]] # RECORD ROW PERMUTATION
                A = A[index_flip(np.arange(n),k,p_index)] # PERMUTE A BY P
            if A[k,k] == 0: # EXIT ROUTINE IF DIAGONAL ELEMENT IS 0
                print('Error: diagonal element of 0 encountered')
                print('Try complete pivoting')
                break
            A[k+1:, k] = A[k+1:, k]/A[k,k] # COMPUTE MULTIPLIERS IN PLACE
            for j in range(k+1, n):
                for i in range(k+1, n):
                    A[i,j] = A[i,j] - A[i,k]*A[k,j] # COMPUTE REMAINING ENTRIES OF SUBMATRIX
        return A, P # RETURN COMPOSITE LU MATRIX AND PERMUTATION ARRAY
    
    if pivot == 'complete': # WITH COMPLETE PIVOTING
        P = np.arange(n) # INITIALIZE ROW PERMUTATION ARRAY
        Q = np.arange(n) # INITIALIZE COLUMN PERMUTATION ARRAY
        for k in range(n-1):
            p_index = abs(A[k:,k:]).argmax() // A[k:,k:].shape[0] + k # FIND INDEX OF ROW WITH LARGEST PIVOT VALUE
            q_index = abs(A[k:,k:].T).argmax() // A[k:,k:].shape[1] + k # FIND INDEX OF COLUMN WITH LARGEST PIVOT VALUE
            if p_index - k > 0:
                P[[k,p_index]] = P[[p_index,k]] # RECORD ROW PERMUTATION
                A = A[index_flip(np.arange(n),k,p_index)] # PERMUTE A BY P
            if q_index - k > 0:
                Q[[k,q_index]] = Q[[q_index,k]] # RECORD COLUMN PERMUTATION
                A = A.T[index_flip(np.arange(n),k,q_index)].T # PERMUTE A BY Q
            if A[k,k] == 0: # EXIT ROUTINE IF DIAGONAL ELEMENT IS 0
                print('Error: matrix is singular, LU factorization does not exist')
                break
            A[k+1:, k] = A[k+1:, k]/A[k,k] # COMPUTE MULTIPLIERS IN PLACE
            for j in range(k+1, n):
                for i in range(k+1, n):
                    A[i,j] = A[i,j] - A[i,k]*A[k,j] # COMPUTE REMAINING ENTRIES OF SUBMATRIX
        return A, P, Q # RETURN COMPOSITE LU MATRIX AND PERMUTATION ARRAYS

#%%

# Step 4:
# TAKES COMPOSITE LU MATRIX AND EXTRACTS WHAT THE iTH ROW OR COLUMN WOULD BE FOR L OR U
# THIS METHOD IS NEEDED SO WE CAN PERFORM THE NECESSARY COMPUTATIONS WITH L AND U WITHOUT EVER
# ALLOCATING MEMORY FOR A FULL VERSION OF L OR U
def get_LU_vector(LU, lower_or_upper, row_or_col, i):
    if lower_or_upper == 'L':
        if row_or_col == 'row':
            v = LU[i].copy() # EXTRACT iTH ROW OF COMPOSITE LU
            v[i] = 1 # DIAGONAL ELEMENTS OF L ARE 1
            v[i+1:] = 0 # SUPER DIAGONAL ELEMENTS OF L ARE 0
        elif row_or_col == 'col':
            v = LU.T[i].copy() # EXTRACT iTH COLUMN OF COMPOSITE LU
            v[i] = 1 # DIAGONAL ELEMENTS OF L ARE 1
            v[:i] = 0 # SUPER DIAGONAL ELEMENTS OF L ARE 0
    elif lower_or_upper == 'U':
        if row_or_col == 'row':
            v = LU[i].copy() # EXTRACT iTH ROW OF COMPOSITE LU
            v[:i] = 0 # SUB DIAGONAL ELEMENTS OF U ARE 0
        elif row_or_col == 'col':
            v = LU.T[i].copy() # EXTRACT iTH COLUMN OF COMPOSITE LU
            v[i+1:] = 0 # SUB DIAGONAL ELEMENTS OF U ARE 0
    return v
    
def forward_sub(b, LU, orientation_method): # SOLVES Ly = b FOR BOTH ROW/COLUMN ORIENTED METHODS
    n = b.shape[0]
    if orientation_method == 'row':
        y = np.ndarray(n)
        y[0] = b[0]
        for i in range(1,n):
            L_i = get_LU_vector(LU,'L','row',i)
            y[i] = (b[i]-sum(L_i[:i]*y[:i]))/L_i[i]
        return y
    if orientation_method == 'col':
        b_copy = b.copy()
        for j in range(n-1):
            L_j = get_LU_vector(LU,'L','col',j)
            b_copy[j] = b_copy[j]/L_j[j]
            b_copy[j+1:] = b_copy[j+1:]-(b_copy[j]*L_j[j+1:])
        b_copy[n-1] = b_copy[n-1]
        return b_copy
    
def backward_sub(b, LU, orientation_method): # SOLVES Ux = y FOR BOTH ROW/COLUMN ORIENTED METHODS
    n = b.shape[0]
    if orientation_method == 'row':
        x = np.ndarray(n)
        x[n-1] = b[n-1]/get_LU_vector(LU,'U','row',n-1)[n-1]
        for i in reversed(range(n-1)):
            U_i = get_LU_vector(LU,'U','row',i)
            x[i] = (b[i]-sum(U_i[i+1:]*x[i+1:]))/LU[i,i]
        return x
    if orientation_method == 'col':
        b_copy = b.copy()
        for j in reversed(range(1,n)):
            U_j = get_LU_vector(LU,'U','col',j)
            b_copy[j] = b_copy[j]/U_j[j]
            b_copy[:j] = b_copy[:j]-(b_copy[j]*U_j[:j])
        b_copy[0] = b_copy[0]/LU[0,0]
        return b_copy

def solver(b, LU, orientation_method, pivot, P=None, Q=None): # SOLVES LINEAR SYSTEM USING LU FACTORIZATION AND PERMUTATION ARRAYS
    if pivot != 'none':
        b = b[P].copy()
    y = forward_sub(b, LU, orientation_method)
    x = backward_sub(y, LU, orientation_method)
    if pivot == 'complete':
        Q_T = np.array([list(Q).index(i) for i in range(len(Q))])
        x = x[Q_T]
    return x

#%%

# Step 5:
def M_one_norm(A):
    m = A.shape[1]
    max_col_sum = max([sum(abs(A.T[j])) for j in range(m)])
    return max_col_sum

def M_F_norm(A):
    n = A.shape[0]
    F = np.sqrt(sum([sum(A[i]**2) for i in range(n)]))
    return F

def PAQ(P, A, Q):
    return A[P].T[Q].T

def mat_mult_LU(LU):
    M = LU.copy()
    for i in range(LU.shape[0]):
        for j in range(LU.shape[1]):
            M[i,j] = sum(get_LU_vector(LU,'L','row',i)*get_LU_vector(LU,'U','col',j))
    return M

def v_1_norm(v):
    return sum(abs(v))

def v_2_norm(v):
    return np.sqrt(sum(v**2))

def generate_x(n):
    x = np.zeros(n)
    for i in range(n):
        x[i] = rnd.random()
    while v_1_norm(x) < 1 or v_2_norm(x) < 1:
        x = x*1.5 
    return x

def generate_b(A,x):
    b = mat_mult(A,x)
    return b

def get_A_from_LU(LU,pivot,P=None,Q=None):  
    M = mat_mult_LU(LU)
    if pivot != 'none':
        P_T = P = np.array([list(P).index(i) for i in range(len(P))])
        M = M[P_T]
        if pivot =='complete':
            Q_T = np.array([list(Q).index(i) for i in range(len(Q))])
            M = M.T[Q_T].T
    return M

#%%
def steepest_descent(A,P,b,tol):
    n = len(b)
    x = np.ndarray((n))
    LU, p, q = LU_factorization(P.copy(), 'complete')
    for i in range(n):
        x[i] = rnd.choice([1,-1]) * rnd.random()
    r = b - mat_mult(A,x)
    i = 0
    while sum(abs(r)) > tol:
       i += 1
       z = solver(r,LU,'col','complete',p, q)
       ω = mat_mult(A,z)
       α = sum(r*z)/sum(z*mat_mult(A,z))
       x = x + α*z
       r = r - α*ω
    return x

A = np.array([[7,4,1],[3,7,-1],[-1,1,2]],'float')
b = np.array([2,3,1],'float')
P = np.array([[np.diag(A)[0],0,0],[0,np.diag(A)[1],0],[0,0,np.diag(A)[2]]])
# solution: .705782, -.493197, 1.69898

test = steepest_descent(A,P,b,.0000001)

#%%

A = np.array([[2,1,0],[-4,0,4],[2,5,10]], dtype='float')
b = np.array([3,0,17],'float')
pivot = 'complete'
orientation = 'row'
LU_c, p, q = LU_factorization(A, pivot)
# print(get_A_from_LU(LU_c,pivot,p,q))
x_c = solver(b, LU_c, orientation, pivot, p, q)
print('Composite LU Matrix with complete pivoting:')
print(LU_c,'\n')
print('Pivot vector p:')
print(p,'\n')
print('Pivot vector q:')
print(q,'\n')
print('Solution x_c:')
print(x_c)