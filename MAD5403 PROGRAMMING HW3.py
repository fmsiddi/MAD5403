#%%
from tqdm import tqdm
import numpy as np
import random as rnd
import matplotlib.pyplot as plt
plt.style.use('seaborn')

def generate_positive_L(n,a):
    L = np.zeros((n,n))
    np.fill_diagonal(L,1) # UNITARY DIAGONAL
    for i in range(1,n):
        for j in range(i):
            # rnd.random() ONLY GENERATES FLOATS < |1|, and we divide by 'a' to improve condition number of L*L.T
            L[i,j] = rnd.random()/a 
    return L

def v_2_norm(v):
    return np.sqrt(sum(v**2))

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

def generate_x(n):
    x = np.zeros(n)
    for i in range(n):
        x[i] = rnd.random()
    while v_2_norm(x) < 1:
        x = x*1.5 
    return x

def generate_b(A,x):
    b = mat_mult(A,x)
    return b

def forward_sub(b, L, orientation_method): # SOLVES Ly = b FOR BOTH ROW/COLUMN ORIENTED METHODS
    n = b.shape[0]
    if orientation_method == 'row':
        y = np.ndarray(n)
        y[0] = b[0]/L[0,0]
        for i in range(1,n):
            L_i = L[i]
            y[i] = (b[i]-sum(L_i[:i]*y[:i]))/L_i[i]
        return y
    if orientation_method == 'col':
        b_copy = b.copy()
        for j in range(n-1):
            L_j = L[:,j]
            b_copy[j] = b_copy[j]/L_j[j]
            b_copy[j+1:] = b_copy[j+1:]-(b_copy[j]*L_j[j+1:])
        b_copy[n-1] = b_copy[n-1]/L[n-1,n-1]
        return b_copy
    
def backward_sub(b, U, orientation_method): # SOLVES Ux = y FOR BOTH ROW/COLUMN ORIENTED METHODS
    n = b.shape[0]
    if orientation_method == 'row':
        x = np.ndarray(n)
        x[n-1] = b[n-1]/U[n-1,n-1]
        for i in reversed(range(n-1)):
            U_i = U[i]
            x[i] = (b[i]-sum(U_i[i+1:]*x[i+1:]))/U[i,i]
        return x
    if orientation_method == 'col':
        b_copy = b.copy()
        for j in reversed(range(1,n)):
            U_j = U[:,j]
            b_copy[j] = b_copy[j]/U_j[j]
            b_copy[:j] = b_copy[:j]-(b_copy[j]*U_j[:j])
        b_copy[0] = b_copy[0]/U[0,0]
        return b_copy


#%%
def steepest_descent(A,b,P_choice,tol,sol,record_trend=False):
    n = len(b)
    if record_trend:
        sol_norm = np.sqrt(sum(sol**2))
        b_norm = np.sqrt(sum(b**2))
    if P_choice == 'J':
        P = np.diag(A)
    elif P_choice == 'SGS':
        C = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                if j<=i:
                    C[i,j] = A[i,j]/(np.sqrt(np.diag(A))[j])
    elif P_choice != 'I':
        print('Error: Please indicate proper choice of preconditioner matrix')
        exit
    # x = np.random.random(n)
    x = np.tile(0.,n)
    r = b - mat_mult(A,x)
    err_res = np.ndarray((1,2))
    i = 0
    while np.sqrt(sum(r**2)) > tol:
        if P_choice == 'I':
            z = r.copy()
        elif P_choice == 'J':
            z = r/P
        else:
            y = forward_sub(r,C,'col')
            z = backward_sub(y,C.T,'col')
        ω = mat_mult(A,z)
        α = sum(r*z)/sum(z*ω)
        x = x + α*z
        r = r - α*ω
        if record_trend:
            error_norm = np.sqrt(sum((x - sol)**2))/sol_norm
            resid_norm = np.sqrt(sum(r**2))/b_norm
            if i == 0:
                err_res[i,0] = error_norm
                err_res[i,1] = resid_norm
            else:
                err_res = np.concatenate((err_res,np.array([error_norm,resid_norm]).reshape(1,2)))
        i += 1
    if record_trend:
        return x, err_res
    else:
        return x, i

def conjugate_gradient(A,b,P_choice,tol,sol,record_trend=False):
    n = len(b)
    if P_choice == 'J':
        P = np.diag(A)
    elif P_choice == 'SGS':
        C = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                if j<=i:
                    C[i,j] = A[i,j]/(np.sqrt(np.diag(A))[j])
    elif P_choice != 'I':
        print('Error: Please indicate proper choice of preconditioner matrix')
        exit
    # x = np.random.random(n)
    x = np.tile(0.,n)
    r = b - mat_mult(A,x)
    if record_trend:
        err_res = np.ndarray((1,2))
        sol_norm = np.sqrt(sum(sol**2))
        b_norm = np.sqrt(sum(b**2))
        error_norm = np.sqrt(sum((x - sol)**2))/sol_norm
        resid_norm = np.sqrt(sum(r**2))/b_norm
        err_res[0,0] = error_norm
        err_res[0,1] = resid_norm
    if P_choice == 'I':
        z = r.copy()
    elif P_choice == 'J':
        z = r/P
    else:
        y = forward_sub(r,C,'col')
        z = backward_sub(y,C.T,'col')
    p = z.copy()
    i = 0
    while np.sqrt(sum(r**2)) > tol:
        v = mat_mult(A,p)
        dot_r_z = sum(r*z)
        α = dot_r_z/sum(p*v)
        x = x + α*p
        r = r - α*v
        if record_trend:
            error_norm = np.sqrt(sum((x - sol)**2))/sol_norm
            resid_norm = np.sqrt(sum(r**2))/b_norm
            err_res = np.concatenate((err_res,np.array([error_norm,resid_norm]).reshape(1,2)))
        if P_choice == 'I':
            z = r.copy()
        elif P_choice == 'J':
            z = r/P
        else:
            y = forward_sub(r,C,'col')
            z = backward_sub(y,C.T,'col')
        β = sum(r*z)/dot_r_z
        p = z + β*p
        i += 1
    if record_trend:
        return x, err_res
    else:
        return x, i
    
#%%
# TESTING
# n = 100
# tol = 1e-6
# L = generate_positive_L(n,10)
# A = mat_mult(L,L.T)
# eigs = np.linalg.eigvals(A)
# ratio = max(eigs)/min(eigs)
# print('condition number:',ratio)
    
# x = generate_x(n)
# b = generate_b(A,x)

# x̃, err_res = steepest_descent(A,b,'J',tol,x,record_trend=True)
# x̃, err_res = conjugate_gradient(A,b,'I',tol,x,record_trend=True)
# error_norm = np.sqrt(sum((I_x-x)**2))





# eigs = np.linalg.eigvals(A)
# ratio = max(eigs)/min(eigs)
# SPD = np.all(eigs > 0)
# print('condition number:', np.linalg.cond(A))
# print('error norm:',error_norm)
# print('number of iterations:',i)
# print('number of iterations:',err_res.shape[0])

#%%

# A = np.array([[5,7,6,5],[7,10,8,7],[6,8,10,9],[5,7,9,10]], dtype='float')
# b = np.array([23,32,33,31],'float')

# tol = 1e-6
# x̃, i = steepest_descent(A,b,'J',tol,record_trend=True)
# x̃, i = conjugate_gradient(A,b,'I',tol,record_trend=False)

#%%
def SD_CG_P_analysis(trials,tol,a):
    # first dimension is trial,
    # second dimension is matrix size 
    # third dimension is method
    # fourth dimension is preconditioner
    # fifth dimension is error/resid/iteration count 
    out = np.ndarray((trials,2,2,3,3))
    K = np.ndarray((trials,2))
    for i in tqdm(range(trials), desc="Running accuracy tests"):
        for n in [10,100]:
            L = generate_positive_L(n,a)
            A = mat_mult(L,L.T)
            if n == 10:
                K[i,0] = np.linalg.cond(A)
            else:
                K[i,1] = np.linalg.cond(A)
            x = generate_x(n)
            b = generate_b(A,x)
            for method in ['SD','CG']:
                if method == 'SD':
                    I_x, I_i = steepest_descent(A,b,'I',tol,x,record_trend=False)
                    J_x, J_i = steepest_descent(A,b,'J',tol,x,record_trend=False)
                    SGS_x, SGS_i = steepest_descent(A,b,'SGS',tol,x,record_trend=False)
                elif method == 'CG':
                    I_x, I_i = conjugate_gradient(A,b,'I',tol,x,record_trend=False)
                    J_x, J_i = conjugate_gradient(A,b,'J',tol,x,record_trend=False)
                    SGS_x, SGS_i = conjugate_gradient(A,b,'SGS',tol,x,record_trend=False)
                I_r = b - mat_mult(A,I_x)
                J_r = b - mat_mult(A,J_x)
                SGS_r = b - mat_mult(A,SGS_x)
                I_r_norm = np.sqrt(sum(I_r**2))
                J_r_norm = np.sqrt(sum(J_r**2))
                SGS_r_norm = np.sqrt(sum(SGS_r**2))
                I_err_norm = np.sqrt(sum((I_x-x)**2))
                J_err_norm = np.sqrt(sum((J_x-x)**2))
                SGS_err_norm = np.sqrt(sum((SGS_x-x)**2))
                if n == 10 and method == 'SD':
                    out[i,0,0,0] = np.array([I_err_norm, I_r_norm, I_i])
                    out[i,0,0,1] = np.array([J_err_norm, J_r_norm, J_i])
                    out[i,0,0,2] = np.array([SGS_err_norm, SGS_r_norm, SGS_i])
                elif n == 10 and method == 'CG':
                    out[i,0,1,0] = np.array([I_err_norm, I_r_norm, I_i])
                    out[i,0,1,1] = np.array([J_err_norm, J_r_norm, J_i])
                    out[i,0,1,2] = np.array([SGS_err_norm, SGS_r_norm, SGS_i])
                elif n == 100 and method == 'SD':
                    out[i,1,0,0] = np.array([I_err_norm, I_r_norm, I_i])
                    out[i,1,0,1] = np.array([J_err_norm, J_r_norm, J_i])
                    out[i,1,0,2] = np.array([SGS_err_norm, SGS_r_norm, SGS_i])
                elif n == 100 and method == 'CG':
                    out[i,1,1,0] = np.array([I_err_norm, I_r_norm, I_i])
                    out[i,1,1,1] = np.array([J_err_norm, J_r_norm, J_i])
                    out[i,1,1,2] = np.array([SGS_err_norm, SGS_r_norm, SGS_i])
    return out, K

trials = 500
tol = 1e-6
results, K = SD_CG_P_analysis(trials,tol,4)

resid_10 = results[:,0,:,:,0]
error_10 = results[:,0,:,:,1]
iter_10 = results[:,0,:,:,2]
resid_100 = results[:,1,:,:,0]
error_100 = results[:,1,:,:,1]
iter_100 = results[:,1,:,:,2]


#%%
# Gram-Schmidt
# def generate_Q(n):
#     Q = np.ndarray((n,n))
#     V = np.random.random((n,n))
#     # V = np.array([[1.,2.],[3.,4.]])
#     Q[0] = V[0]
#     for i in range(1,n):
#         subtract = 0
#         for j in range(i):
#             subtract -= (sum(V[i]*Q[j])/sum(Q[j]**2))*Q[j]
#         Q[i] = V[i] + subtract
#     return Q
      
# n = 5
# step = .5/5
# Q = generate_Q(n)
# lamb = np.diag(np.arange(.5,1,step))
# A = mat_mult(Q.T,mat_mult(lamb,Q))
# eigs = np.linalg.eigvals(A)
# ratio = max(eigs)/min(eigs)
# print('condition number:',ratio)
# for i in range(n):
#     for j in range(n):
#         if i != j:
#             print(round(sum(Q[i]*Q[j])))
            
