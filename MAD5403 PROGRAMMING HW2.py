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
        for j in range(n-1):
            L_j = get_LU_vector(LU,'L','col',j)
            b[j] = b[j]/L_j[j]
            b[j+1:] = b[j+1:]-(b[j]*L_j[j+1:])
        b[n-1] = b[n-1]
        return b
    
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
        for j in reversed(range(1,n)):
            U_j = get_LU_vector(LU,'U','col',j)
            b[j] = b[j]/U_j[j]
            b[:j] = b[:j]-(b[j]*U_j[:j])
        b[0] = b[0]/LU[0,0]
        return b

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

def accuracy_test(trials,pivot):
    M_F = np.ndarray((trials,2))
    M_1 = np.ndarray((trials,2))
    x_1 = np.ndarray((trials,2))
    x_2 = np.ndarray((trials,2))
    r_1 = np.ndarray((trials,2))
    r_2 = np.ndarray((trials,2))
    for i in tqdm(range(trials), desc="Running accuracy tests for pivot type: '{}'".format(pivot)):
        for n in [10,100]:
            L = generate_L(n)
            U = generate_U(n)
            A = mat_mult(L,U)
            A_copy = A.copy()
            x = generate_x(n)
            b = generate_b(A,x)
            b_copy = b.copy()
            if pivot == 'none':
                A_copy = LU_factorization(A_copy, pivot)
                P = np.arange(n)
                Q = np.arange(n)
            elif pivot == 'partial':
                A_copy, P = LU_factorization(A_copy, pivot)
                Q = np.arange(n)
            elif pivot == 'complete':
                A_copy, P, Q = LU_factorization(A_copy, pivot)
            x̃ = solver(b_copy, A_copy, 'row', pivot, P, Q)
            r = b - mat_mult(A,x̃)
            if n == 10:
                M_F[i,0] = M_F_norm(PAQ(P, A, Q) - mat_mult_LU(A_copy))/M_F_norm(A)
                M_1[i,0] = M_one_norm(PAQ(P, A, Q) - mat_mult_LU(A_copy))/M_one_norm(A)
                x_1[i,0] = v_1_norm(x - x̃)/v_1_norm(x)
                x_2[i,0] = v_2_norm(x - x̃)/v_2_norm(x)
                r_1[i,0] = v_1_norm(r)/v_1_norm(b)
                r_2[i,0] = v_2_norm(r)/v_2_norm(b)
            elif n == 100:
                M_F[i,1] = M_F_norm(PAQ(P, A, Q) - mat_mult_LU(A_copy))/M_F_norm(A)
                M_1[i,1] = M_one_norm(PAQ(P, A, Q) - mat_mult_LU(A_copy))/M_one_norm(A)
                x_1[i,1] = v_1_norm(x - x̃)/v_1_norm(x)
                x_2[i,1] = v_2_norm(x - x̃)/v_2_norm(x)
                r_1[i,1] = v_1_norm(r)/v_1_norm(b)
                r_2[i,1] = v_2_norm(r)/v_2_norm(b)
    return M_F, M_1, x_1, x_2, r_1, r_2

trials = 500
bins = int(trials/20)
M_F_none, M_1_none, x_1_none, x_2_none, r_1_none, r_2_none = accuracy_test(trials,'none')
M_F_partial, M_1_partial, x_1_partial, x_2_partial, r_1_partial, r_2_partial = accuracy_test(trials,'partial')
M_F_complete, M_1_complete, x_1_complete, x_2_complete, r_1_complete, r_2_complete = accuracy_test(trials,'complete')
print('\n')

#%%
# GRAPHING HISTOGRAM USING PYPLOT
f1, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2, sharey=True, sharex=True)
f1.suptitle('Relative Factorization Error Without Pivoting', fontsize=16)
ax1.hist(M_F_none[:,0],bins)
ax1.set_ylabel('Frequency')
ax1.set_title('F-norm (n = 10)')

ax2.hist(M_F_none[:,1],bins)
ax2.set_title('F-norm (n = 100)')

ax3.hist(M_1_none[:,0],bins)
ax3.set_xlabel('Error')
ax3.set_ylabel('Frequency')
ax3.set_title('1-norm (n = 10)')

ax4.hist(M_1_none[:,1],bins)
ax4.set_xlabel('Error')
ax4.set_title('1-norm (n = 100)')

plt.show()

# GRAPHING HISTOGRAM USING PYPLOT
f1, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2, sharey=True, sharex=True)
f1.suptitle('Relative Factorization Error With Partial Pivoting', fontsize=16)
ax1.hist(M_F_partial[:,0],bins)
ax1.set_ylabel('Frequency')
ax1.set_title('F-norm (n = 10)')

ax2.hist(M_F_partial[:,1],bins)
ax2.set_title('F-norm (n = 100)')

ax3.hist(M_1_partial[:,0],bins)
ax3.set_xlabel('Error')
ax3.set_ylabel('Frequency')
ax3.set_title('1-norm (n = 10)')

ax4.hist(M_1_partial[:,1],bins)
ax4.set_xlabel('Error')
ax4.set_title('1-norm (n = 100)')

plt.show()

# GRAPHING HISTOGRAM USING PYPLOT
f1, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2, sharey=True, sharex=True)
f1.suptitle('Relative Factorization Error With Complete Pivoting', fontsize=16)
ax1.hist(M_F_complete[:,0],bins)
ax1.set_ylabel('Frequency')
ax1.set_title('F-norm (n = 10)')

ax2.hist(M_F_complete[:,1],bins)
ax2.set_title('F-norm (n = 100)')

ax3.hist(M_1_complete[:,0],bins)
ax3.set_xlabel('Error')
ax3.set_ylabel('Frequency')
ax3.set_title('1-norm (n = 10)')

ax4.hist(M_1_complete[:,1],bins)
ax4.set_xlabel('Error')
ax4.set_title('1-norm (n = 100)')

plt.show()



# GRAPHING HISTOGRAM USING PYPLOT
f1, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2, sharey=True)
f1.suptitle('Relative Solution Error Without Pivoting', fontsize=16)
ax1.hist(x_2_none[:,0],bins)
ax1.set_ylabel('Frequency')
ax1.set_title('2-norm (n = 10)')

ax2.hist(x_2_none[:,1],bins)
ax2.set_title('2-norm (n = 100)')

ax3.hist(x_1_none[:,0],bins)
ax3.set_xlabel('Error')
ax3.set_ylabel('Frequency')
ax3.set_title('1-norm (n = 10)')

ax4.hist(x_1_none[:,1],bins)
ax4.set_xlabel('Error')
ax4.set_title('1-norm (n = 100)')

plt.show()

# GRAPHING HISTOGRAM USING PYPLOT
f1, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2, sharey=True)
f1.suptitle('Relative Solution Error With Partial Pivoting', fontsize=16)
ax1.hist(x_2_partial[:,0],bins)
ax1.set_ylabel('Frequency')
ax1.set_title('2-norm (n = 10)')

ax2.hist(x_2_partial[:,1],bins)
ax2.set_title('2-norm (n = 100)')

ax3.hist(x_1_partial[:,0],bins)
ax3.set_xlabel('Error')
ax3.set_ylabel('Frequency')
ax3.set_title('1-norm (n = 10)')

ax4.hist(x_1_partial[:,1],bins)
ax4.set_xlabel('Error')
ax4.set_title('1-norm (n = 100)')

plt.show()

# GRAPHING HISTOGRAM USING PYPLOT
f1, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2, sharey=True)
f1.suptitle('Relative Solution Error With Complete Pivoting', fontsize=16)
ax1.hist(x_2_complete[:,0],bins)
ax1.set_ylabel('Frequency')
ax1.set_title('2-norm (n = 10)')

ax2.hist(x_2_complete[:,1],bins)
ax2.set_title('2-norm (n = 100)')

ax3.hist(x_1_complete[:,0],bins)
ax3.set_xlabel('Error')
ax3.set_ylabel('Frequency')
ax3.set_title('1-norm (n = 10)')

ax4.hist(x_1_complete[:,1],bins)
ax4.set_xlabel('Error')
ax4.set_title('1-norm (n = 100)')

plt.show()



# GRAPHING HISTOGRAM USING PYPLOT
f1, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2, sharey=True, sharex=True)
f1.suptitle('Relative Residual Error Without Pivoting', fontsize=16)
ax1.hist(r_2_none[:,0],bins)
ax1.set_ylabel('Frequency')
ax1.set_title('2-norm (n = 10)')

ax2.hist(r_2_none[:,1],bins)
ax2.set_title('2-norm (n = 100)')

ax3.hist(r_1_none[:,0],bins)
ax3.set_xlabel('Error')
ax3.set_ylabel('Frequency')
ax3.set_title('1-norm (n = 10)')

ax4.hist(r_1_none[:,1],bins)
ax4.set_xlabel('Error')
ax4.set_title('1-norm (n = 100)')

plt.show()

# GRAPHING HISTOGRAM USING PYPLOT
f1, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2, sharey=True, sharex=True)
f1.suptitle('Relative Residual Error With Partial Pivoting', fontsize=16)
ax1.hist(r_2_partial[:,0],bins)
ax1.set_ylabel('Frequency')
ax1.set_title('2-norm (n = 10)')

ax2.hist(r_2_partial[:,1],bins)
ax2.set_title('2-norm (n = 100)')

ax3.hist(r_1_partial[:,0],bins)
ax3.set_xlabel('Error')
ax3.set_ylabel('Frequency')
ax3.set_title('1-norm (n = 10)')

ax4.hist(r_1_partial[:,1],bins)
ax4.set_xlabel('Error')
ax4.set_title('1-norm (n = 100)')

plt.show()

# GRAPHING HISTOGRAM USING PYPLOT
f1, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2, sharey=True, sharex=True)
f1.suptitle('Relative Residual Error With Complete Pivoting', fontsize=16)
ax1.hist(r_2_complete[:,0],bins)
ax1.set_ylabel('Frequency')
ax1.set_title('2-norm (n = 10)')

ax2.hist(r_2_complete[:,1],bins)
ax2.set_title('2-norm (n = 100)')

ax3.hist(r_1_complete[:,0],bins)
ax3.set_xlabel('Error')
ax3.set_ylabel('Frequency')
ax3.set_title('1-norm (n = 10)')

ax4.hist(r_1_complete[:,1],bins)
ax4.set_xlabel('Error')
ax4.set_title('1-norm (n = 100)')

plt.show()

#%%

def SPD_accuracy_test(trials,pivot):
    M_F = np.ndarray((trials,2))
    M_1 = np.ndarray((trials,2))
    x_1 = np.ndarray((trials,2))
    x_2 = np.ndarray((trials,2))
    r_1 = np.ndarray((trials,2))
    r_2 = np.ndarray((trials,2))
    for i in tqdm(range(trials), desc="Running SPD accuracy tests for pivot type: '{}'".format(pivot)):
        for n in [10,100]:
            L = generate_positive_L(n)
            A = mat_mult(L,L.T)
            A_copy = A.copy()
            x = generate_x(n)
            b = generate_b(A,x)
            b_copy = b.copy()
            if pivot == 'none':
                A_copy = LU_factorization(A_copy, pivot)
                P = np.arange(n)
                Q = np.arange(n)
            elif pivot == 'partial':
                A_copy, P = LU_factorization(A_copy, pivot)
                Q = np.arange(n)
            elif pivot == 'complete':
                A_copy, P, Q = LU_factorization(A_copy, pivot)
            x̃ = solver(b_copy, A_copy, 'row', pivot, P, Q)
            r = b - mat_mult(A,x̃)
            if n == 10:
                M_F[i,0] = M_F_norm(PAQ(P, A, Q) - mat_mult_LU(A_copy))/M_F_norm(A)
                M_1[i,0] = M_one_norm(PAQ(P, A, Q) - mat_mult_LU(A_copy))/M_one_norm(A)
                x_1[i,0] = v_1_norm(x - x̃)/v_1_norm(x)
                x_2[i,0] = v_2_norm(x - x̃)/v_2_norm(x)
                r_1[i,0] = v_1_norm(r)/v_1_norm(b)
                r_2[i,0] = v_2_norm(r)/v_2_norm(b)
            elif n == 100:
                M_F[i,1] = M_F_norm(PAQ(P, A, Q) - mat_mult_LU(A_copy))/M_F_norm(A)
                M_1[i,1] = M_one_norm(PAQ(P, A, Q) - mat_mult_LU(A_copy))/M_one_norm(A)
                x_1[i,1] = v_1_norm(x - x̃)/v_1_norm(x)
                x_2[i,1] = v_2_norm(x - x̃)/v_2_norm(x)
                r_1[i,1] = v_1_norm(r)/v_1_norm(b)
                r_2[i,1] = v_2_norm(r)/v_2_norm(b)
    return M_F, M_1, x_1, x_2, r_1, r_2

trials = 500
bins = int(trials/20)
SPD_M_F_none, SPD_M_1_none, SPD_x_1_none, SPD_x_2_none, SPD_r_1_none, SPD_r_2_none = SPD_accuracy_test(trials,'none')
SPD_M_F_partial, SPD_M_1_partial, SPD_x_1_partial, SPD_x_2_partial, SPD_r_1_partial, SPD_r_2_partial = SPD_accuracy_test(trials,'partial')
SPD_M_F_complete, SPD_M_1_complete, SPD_x_1_complete, SPD_x_2_complete, SPD_r_1_complete, SPD_r_2_complete = SPD_accuracy_test(trials,'complete')
print('\n')

#%%
# GRAPHING HISTOGRAM USING PYPLOT
f1, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2, sharey=True, sharex=True)
f1.suptitle('Relative Factorization Error Without Pivoting (SPD)', fontsize=16)
ax1.hist(SPD_M_F_none[:,0],bins)
ax1.set_ylabel('Frequency')
ax1.set_title('F-norm (n = 10)')

ax2.hist(SPD_M_F_none[:,1],bins)
ax2.set_title('F-norm (n = 100)')

ax3.hist(SPD_M_1_none[:,0],bins)
ax3.set_xlabel('Error')
ax3.set_ylabel('Frequency')
ax3.set_title('1-norm (n = 10)')

ax4.hist(SPD_M_1_none[:,1],bins)
ax4.set_xlabel('Error')
ax4.set_title('1-norm (n = 100)')

plt.show()

# GRAPHING HISTOGRAM USING PYPLOT
f1, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2, sharey=True, sharex=True)
f1.suptitle('Relative Factorization Error With Partial Pivoting (SPD)', fontsize=16)
ax1.hist(SPD_M_F_partial[:,0],bins)
ax1.set_ylabel('Frequency')
ax1.set_title('F-norm (n = 10)')

ax2.hist(SPD_M_F_partial[:,1],bins)
ax2.set_title('F-norm (n = 100)')

ax3.hist(SPD_M_1_partial[:,0],bins)
ax3.set_xlabel('Error')
ax3.set_ylabel('Frequency')
ax3.set_title('1-norm (n = 10)')

ax4.hist(SPD_M_1_partial[:,1],bins)
ax4.set_xlabel('Error')
ax4.set_title('1-norm (n = 100)')

plt.show()

# GRAPHING HISTOGRAM USING PYPLOT
f1, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2, sharey=True, sharex=True)
f1.suptitle('Relative Factorization Error With Complete Pivoting (SPD)', fontsize=16)
ax1.hist(SPD_M_F_complete[:,0],bins)
ax1.set_ylabel('Frequency')
ax1.set_title('F-norm (n = 10)')

ax2.hist(SPD_M_F_complete[:,1],bins)
ax2.set_title('F-norm (n = 100)')

ax3.hist(SPD_M_1_complete[:,0],bins)
ax3.set_xlabel('Error')
ax3.set_ylabel('Frequency')
ax3.set_title('1-norm (n = 10)')

ax4.hist(SPD_M_1_complete[:,1],bins)
ax4.set_xlabel('Error')
ax4.set_title('1-norm (n = 100)')

plt.show()



# GRAPHING HISTOGRAM USING PYPLOT
f1, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2, sharey=True)
f1.suptitle('Relative Solution Error Without Pivoting (SPD)', fontsize=16)
ax1.hist(SPD_x_2_none[:,0],bins)
ax1.set_ylabel('Frequency')
ax1.set_title('2-norm (n = 10)')

ax2.hist(SPD_x_2_none[:,1],bins)
ax2.set_title('2-norm (n = 100)')

ax3.hist(SPD_x_1_none[:,0],bins)
ax3.set_xlabel('Error')
ax3.set_ylabel('Frequency')
ax3.set_title('1-norm (n = 10)')

ax4.hist(SPD_x_1_none[:,1],bins)
ax4.set_xlabel('Error')
ax4.set_title('1-norm (n = 100)')

plt.show()

# GRAPHING HISTOGRAM USING PYPLOT
f1, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2, sharey=True)
f1.suptitle('Relative Solution Error With Partial Pivoting (SPD)', fontsize=16)
ax1.hist(SPD_x_2_partial[:,0],bins)
ax1.set_ylabel('Frequency')
ax1.set_title('2-norm (n = 10)')

ax2.hist(SPD_x_2_partial[:,1],bins)
ax2.set_title('2-norm (n = 100)')

ax3.hist(SPD_x_1_partial[:,0],bins)
ax3.set_xlabel('Error')
ax3.set_ylabel('Frequency')
ax3.set_title('1-norm (n = 10)')

ax4.hist(SPD_x_1_partial[:,1],bins)
ax4.set_xlabel('Error')
ax4.set_title('1-norm (n = 100)')

plt.show()

# GRAPHING HISTOGRAM USING PYPLOT
f1, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2, sharey=True)
f1.suptitle('Relative Solution Error With Complete Pivoting (SPD)', fontsize=16)
ax1.hist(SPD_x_2_complete[:,0],bins)
ax1.set_ylabel('Frequency')
ax1.set_title('2-norm (n = 10)')

ax2.hist(SPD_x_2_complete[:,1],bins)
ax2.set_title('2-norm (n = 100)')

ax3.hist(SPD_x_1_complete[:,0],bins)
ax3.set_xlabel('Error')
ax3.set_ylabel('Frequency')
ax3.set_title('1-norm (n = 10)')

ax4.hist(SPD_x_1_complete[:,1],bins)
ax4.set_xlabel('Error')
ax4.set_title('1-norm (n = 100)')

plt.show()



# GRAPHING HISTOGRAM USING PYPLOT
f1, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2, sharey=True, sharex=True)
f1.suptitle('Relative Residual Error Without Pivoting (SPD)', fontsize=16)
ax1.hist(SPD_r_2_none[:,0],bins)
ax1.set_ylabel('Frequency')
ax1.set_title('2-norm (n = 10)')

ax2.hist(SPD_r_2_none[:,1],bins)
ax2.set_title('2-norm (n = 100)')

ax3.hist(SPD_r_1_none[:,0],bins)
ax3.set_xlabel('Error')
ax3.set_ylabel('Frequency')
ax3.set_title('1-norm (n = 10)')

ax4.hist(SPD_r_1_none[:,1],bins)
ax4.set_xlabel('Error')
ax4.set_title('1-norm (n = 100)')

plt.show()

# GRAPHING HISTOGRAM USING PYPLOT
f1, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2, sharey=True, sharex=True)
f1.suptitle('Relative Residual Error With Partial Pivoting (SPD)', fontsize=16)
ax1.hist(SPD_r_2_partial[:,0],bins)
ax1.set_ylabel('Frequency')
ax1.set_title('2-norm (n = 10)')

ax2.hist(SPD_r_2_partial[:,1],bins)
ax2.set_title('2-norm (n = 100)')

ax3.hist(SPD_r_1_partial[:,0],bins)
ax3.set_xlabel('Error')
ax3.set_ylabel('Frequency')
ax3.set_title('1-norm (n = 10)')

ax4.hist(SPD_r_1_partial[:,1],bins)
ax4.set_xlabel('Error')
ax4.set_title('1-norm (n = 100)')

plt.show()

# GRAPHING HISTOGRAM USING PYPLOT
f1, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2, sharey=True, sharex=True)
f1.suptitle('Relative Residual Error With Complete Pivoting (SPD)', fontsize=16)
ax1.hist(SPD_r_2_complete[:,0],bins)
ax1.set_ylabel('Frequency')
ax1.set_title('2-norm (n = 10)')

ax2.hist(SPD_r_2_complete[:,1],bins)
ax2.set_title('2-norm (n = 100)')

ax3.hist(SPD_r_1_complete[:,0],bins)
ax3.set_xlabel('Error')
ax3.set_ylabel('Frequency')
ax3.set_title('1-norm (n = 10)')

ax4.hist(SPD_r_1_complete[:,1],bins)
ax4.set_xlabel('Error')
ax4.set_title('1-norm (n = 100)')

plt.show()

#%%
# A = np.array([[2,1,0],[-4,0,4],[2,5,10]], dtype='float')
# b = np.array([3,0,17])
# pivot = 'none'
# LU = LU_factorization(A, pivot)
# x = solver(b, LU, 'row', pivot)
# print('Composite LU Matrix without pivoting:')
# print(LU,'\n')
# print('Solution x:')
# print(x,'\n')

A = np.array([[2,1,0],[-4,0,4],[2,5,10]], dtype='float')
b = np.array([3,0,17])
pivot = 'partial'
LU_p, p = LU_factorization(A, pivot)
x_p = solver(b, LU_p, 'row', pivot, p)
print('Composite LU Matrix with partial pivoting:')
print(LU_p,'\n')
print('Pivot vector p:')
print(p,'\n')
print('Solution x_p:')
print(x_p,'\n')

A = np.array([[2,1,0],[-4,0,4],[2,5,10]], dtype='float')
b = np.array([3,0,17])
pivot = 'complete'
LU_c, p, q = LU_factorization(A, pivot)
# print(get_A_from_LU(LU_c,pivot,p,q))
x_c = solver(b, LU_c, 'row', pivot, p, q)
print('Composite LU Matrix with complete pivoting:')
print(LU_c,'\n')
print('Pivot vector p:')
print(p,'\n')
print('Pivot vector q:')
print(q,'\n')
print('Solution x_c:')
print(x_c)