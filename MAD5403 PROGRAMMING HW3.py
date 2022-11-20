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
    while v_2_norm(x) < 1: # make sure the 2-norm is larger than 1
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
# A: matrix
# b: vector we use to compute residual
# P_choice: which preconditioner to use
# sol: actual solution x to Ax = b to use in computing error
# record_trend: boolean used to determine whether to store error and residual at each iteration
def steepest_descent(A,b,P_choice,tol,sol,record_trend=False):
    n = len(b)
    if record_trend:
        sol_norm = np.sqrt(sum(sol**2))
        b_norm = np.sqrt(sum(b**2))
    if P_choice == 'J':
        P = np.diag(A) # only diagonal elements of A are stored in an array
    elif P_choice == 'SGS':
        C = np.zeros((n,n))
        # we proceed to only compute lower trianguar matrix C that is factor of P_SGS
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
            z = r.copy() # no preconditioner = residual is left alone
        elif P_choice == 'J':
            z = r/P # Jacobi preconditioner = divide each element of r with each element of P
        else:
            # for SGS, solve Cy = b then C^Tz=y
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

# A: matrix
# b: vector we use to compute residual
# P_choice: which preconditioner to use
# sol: actual solution x to Ax = b to use in computing error
# record_trend: boolean used to determine whether to store error and residual at each iteration
def conjugate_gradient(A,b,P_choice,tol,sol,record_trend=False):
    n = len(b)
    if P_choice == 'J':
        P = np.diag(A) # only diagonal elements of A are stored in an array
    elif P_choice == 'SGS':
        C = np.zeros((n,n))
        # we proceed to only compute lower trianguar matrix C that is factor of P_SGS
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
        z = r.copy() # no preconditioner = residual is left alone
    elif P_choice == 'J':
        z = r/P # Jacobi preconditioner = divide each element of r with each element of P
    else:
        # for SGS, solve Cy = b then C^Tz=y
        y = forward_sub(r,C,'col')
        z = backward_sub(y,C.T,'col')
    p = z.copy()
    i = 0
    while np.sqrt(sum(r**2)) > tol:
        v = mat_mult(A,p)
        dot_r_z = sum(r*z) # we compute this scalar and store in variable so we don't have to recompute later
        α = dot_r_z/sum(p*v)
        x = x + α*p
        r = r - α*v
        if record_trend:
            error_norm = np.sqrt(sum((x - sol)**2))/sol_norm
            resid_norm = np.sqrt(sum(r**2))/b_norm
            err_res = np.concatenate((err_res,np.array([error_norm,resid_norm]).reshape(1,2)))
        if P_choice == 'I':
            z = r.copy() # no preconditioner = residual is left alone
        elif P_choice == 'J':
            z = r/P # Jacobi preconditioner = divide each element of r with each element of P
        else:
            # for SGS, solve Cy = b then C^Tz=y
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
            A = generate_positive_L(n,a)
            A = mat_mult(A,A.T)
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
bins = int(trials/20)

# GRAPHING HISTOGRAM USING PYPLOT
f1, ((ax1,ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(3,2, sharey=True, sharex=True,figsize=(12.8,9.6))
f1.suptitle('Iterations to Convergence for 10x10', fontsize=16)
ax1.hist(iter_10[:,0,0],label='None')
ax1.set_ylabel('Frequency',labelpad=10)
ax1.set_title('Steepest Descent')
ax1.annotate('μ: {}'.format(iter_10[:,0,0].mean()),xy=(7,300))
ax1.annotate('σ: {}'.format(round(iter_10[:,0,0].std(),2)),xy=(7,250))
ax1.legend()

ax2.hist(iter_10[:,1,0],label='None')
ax2.set_title('Conjugate Gradient')
ax2.annotate('μ: {}'.format(iter_10[:,1,0].mean()),xy=(15,300))
ax2.annotate('σ: {}'.format(round(iter_10[:,1,0].std(),2)),xy=(15,250))
ax2.legend()

ax3.hist(iter_10[:,0,1],label='Jacobi')
ax3.set_ylabel('Frequency',labelpad=10)
ax3.annotate('μ: {}'.format(iter_10[:,0,1].mean()),xy=(7,300))
ax3.annotate('σ: {}'.format(round(iter_10[:,0,1].std(),2)),xy=(7,250))
ax3.legend()

ax4.hist(iter_10[:,1,1],label='Jacobi')
ax4.annotate('μ: {}'.format(iter_10[:,1,1].mean()),xy=(15,300))
ax4.annotate('σ: {}'.format(round(iter_10[:,1,1].std(),2)),xy=(15,250))
ax4.legend()

ax5.hist(iter_10[:,0,2],label='SGS')
ax5.set_xlabel('Iterations')
ax5.set_ylabel('Frequency',labelpad=10)
ax5.annotate('μ: {}'.format(iter_10[:,0,2].mean()),xy=(7,300))
ax5.annotate('σ: {}'.format(round(iter_10[:,0,2].std(),2)),xy=(7,250))
ax5.legend()

ax6.hist(iter_10[:,1,2],label='SGS')
ax6.set_xlabel('Iterations')
ax6.annotate('μ: {}'.format(iter_10[:,1,2].mean()),xy=(15,300))
ax6.annotate('σ: {}'.format(round(iter_10[:,1,2].std(),2)),xy=(15,250))
ax6.legend()

plt.show()

#%%

# GRAPHING HISTOGRAM USING PYPLOT
f1, ((ax1,ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(3,2, sharey=True, sharex=True,figsize=(12.8,9.6))
f1.suptitle('Relative Error After Convergence for 10x10', fontsize=16)
ax1.hist(error_10[:,0,0],label='None')
ax1.set_ylabel('Frequency',labelpad=10)
ax1.set_title('Steepest Descent')
ax1.annotate('μ: {}'.format(error_10[:,0,0].mean()),xy=(0,200))
ax1.annotate('σ: {}'.format(error_10[:,0,0].std()),xy=(0,175))
ax1.legend()

ax2.hist(error_10[:,1,0],label='None')
ax2.set_title('Conjugate Gradient')
ax2.annotate('μ: {}'.format(error_10[:,1,0].mean()),xy=(0,200))
ax2.annotate('σ: {}'.format(error_10[:,1,0].std()),xy=(0,175))
ax2.legend()

ax3.hist(error_10[:,0,1],label='Jacobi')
ax3.set_ylabel('Frequency',labelpad=10)
ax3.annotate('μ: {}'.format(error_10[:,0,1].mean()),xy=(0,200))
ax3.annotate('σ: {}'.format(error_10[:,0,1].std()),xy=(0,175))
ax3.legend()

ax4.hist(error_10[:,1,1],label='Jacobi')
ax4.annotate('μ: {}'.format(error_10[:,1,1].mean()),xy=(0,200))
ax4.annotate('σ: {}'.format(error_10[:,1,1].std()),xy=(0,175))
ax4.legend()

ax5.hist(error_10[:,0,2],label='SGS')
ax5.set_xlabel('Error')
ax5.set_ylabel('Frequency',labelpad=10)
ax5.annotate('μ: {}'.format(error_10[:,0,2].mean()),xy=(0,200))
ax5.annotate('σ: {}'.format(error_10[:,0,2].std()),xy=(0,175))
ax5.legend()

ax6.hist(error_10[:,1,2],label='SGS')
ax6.set_xlabel('Error')
ax6.annotate('μ: {}'.format(error_10[:,1,2].mean()),xy=(5e-7,175))
ax6.annotate('σ: {}'.format(error_10[:,1,2].std()),xy=(5e-7,150))
ax6.legend()

plt.show()

#%%

# GRAPHING HISTOGRAM USING PYPLOT
f1, ((ax1,ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(3,2, sharey=True, sharex=True,figsize=(12.8,9.6))
f1.suptitle('Relative Residual After Convergence for 10x10', fontsize=16)
ax1.hist(resid_10[:,0,0],label='None')
ax1.set_ylabel('Frequency',labelpad=10)
ax1.set_title('Steepest Descent')
ax1.annotate('μ: {}'.format(resid_10[:,0,0].mean()),xy=(0,200))
ax1.annotate('σ: {}'.format(resid_10[:,0,0].std()),xy=(0,175))
ax1.legend()

ax2.hist(resid_10[:,1,0],label='None')
ax2.set_title('Conjugate Gradient')
ax2.annotate('μ: {}'.format(resid_10[:,1,0].mean()),xy=(0,200))
ax2.annotate('σ: {}'.format(resid_10[:,1,0].std()),xy=(0,175))
ax2.legend()

ax3.hist(resid_10[:,0,1],label='Jacobi')
ax3.set_ylabel('Frequency',labelpad=10)
ax3.annotate('μ: {}'.format(resid_10[:,0,1].mean()),xy=(0,200))
ax3.annotate('σ: {}'.format(resid_10[:,0,1].std()),xy=(0,175))
ax3.legend()

ax4.hist(resid_10[:,1,1],label='Jacobi')
ax4.annotate('μ: {}'.format(resid_10[:,1,1].mean()),xy=(0,200))
ax4.annotate('σ: {}'.format(resid_10[:,1,1].std()),xy=(0,175))
ax4.legend()

ax5.hist(resid_10[:,0,2],label='SGS')
ax5.set_xlabel('Residual')
ax5.set_ylabel('Frequency',labelpad=10)
ax5.annotate('μ: {}'.format(resid_10[:,0,2].mean()),xy=(0,200))
ax5.annotate('σ: {}'.format(resid_10[:,0,2].std()),xy=(0,175))
ax5.legend()

ax6.hist(resid_10[:,1,2],label='SGS')
ax6.set_xlabel('Residual')
ax6.annotate('μ: {}'.format(resid_10[:,1,2].mean()),xy=(75e-8,175))
ax6.annotate('σ: {}'.format(resid_10[:,1,2].std()),xy=(75e-8,150))
ax6.legend()

plt.show()
#%%

# GRAPHING HISTOGRAM USING PYPLOT
f1, ((ax1,ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(3,2, sharey=True, sharex='col',figsize=(12.8,9.6))
f1.suptitle('Iterations to Convergence for 100x100', fontsize=16)
ax1.hist(iter_100[:,0,0],label='None')
ax1.set_ylabel('Frequency',labelpad=10)
ax1.set_title('Steepest Descent')
ax1.annotate('μ: {}'.format(iter_100[:,0,0].mean()),xy=(500,175))
ax1.annotate('σ: {}'.format(round(iter_100[:,0,0].std(),2)),xy=(500,150))
ax1.legend()

bins = range(int(min(iter_100[:,1,0])),int(max(iter_100[:,1,0])+1))
ax2.hist(iter_100[:,1,0],bins,label='None')
ax2.set_title('Conjugate Gradient')
ax2.annotate('μ: {}'.format(iter_100[:,1,0].mean()),xy=(34,175))
ax2.annotate('σ: {}'.format(round(iter_100[:,1,0].std(),2)),xy=(34,150))
ax2.legend()

ax3.hist(iter_100[:,0,1],label='Jacobi')
ax3.set_ylabel('Frequency',labelpad=10)
ax3.annotate('μ: {}'.format(iter_100[:,0,1].mean()),xy=(2000,150))
ax3.annotate('σ: {}'.format(round(iter_100[:,0,1].std(),2)),xy=(2000,125))
ax3.legend()

bins = range(int(min(iter_100[:,1,1])),int(max(iter_100[:,1,1])+1))
ax4.hist(iter_100[:,1,1],bins,label='Jacobi')
ax4.annotate('μ: {}'.format(iter_100[:,1,1].mean()),xy=(40,150))
ax4.annotate('σ: {}'.format(round(iter_100[:,1,1].std(),2)),xy=(40,125))
ax4.legend()

ax5.hist(iter_100[:,0,2],label='SGS')
ax5.set_xlabel('Iterations')
ax5.set_ylabel('Frequency',labelpad=10)
ax5.annotate('μ: {}'.format(iter_100[:,0,2].mean()),xy=(1000,150))
ax5.annotate('σ: {}'.format(round(iter_100[:,0,2].std(),2)),xy=(1000,125))
ax5.legend()

bins = range(int(min(iter_100[:,1,2])),int(max(iter_100[:,1,2])+1))
ax6.hist(iter_100[:,1,2],bins,label='SGS')
ax6.set_xlabel('Iterations')
ax6.annotate('μ: {}'.format(iter_100[:,1,2].mean()),xy=(34,150))
ax6.annotate('σ: {}'.format(round(iter_100[:,1,2].std(),2)),xy=(34,125))
ax6.legend()

plt.show()

#%%

# GRAPHING HISTOGRAM USING PYPLOT
f1, ((ax1,ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(3,2, sharey=True, sharex=True,figsize=(12.8,9.6))
f1.suptitle('Relative Error After Convergence for 100x100', fontsize=16)
ax1.hist(error_100[:,0,0],label='None')
ax1.set_ylabel('Frequency',labelpad=10)
ax1.set_title('Steepest Descent')
ax1.annotate('μ: {}'.format(error_100[:,0,0].mean()),xy=(3e-7,75))
ax1.annotate('σ: {}'.format(error_100[:,0,0].std()),xy=(3e-7,65))
ax1.legend()

ax2.hist(error_100[:,1,0],label='None')
ax2.set_title('Conjugate Gradient')
ax2.annotate('μ: {}'.format(error_100[:,1,0].mean()),xy=(3e-7,80))
ax2.annotate('σ: {}'.format(error_100[:,1,0].std()),xy=(3e-7,70))
ax2.legend()

ax3.hist(error_100[:,0,1],label='Jacobi')
ax3.set_ylabel('Frequency',labelpad=10)
ax3.annotate('μ: {}'.format(error_100[:,0,1].mean()),xy=(3e-7,80))
ax3.annotate('σ: {}'.format(error_100[:,0,1].std()),xy=(3e-7,70))
ax3.legend()

ax4.hist(error_100[:,1,1],label='Jacobi')
ax4.annotate('μ: {}'.format(error_100[:,1,1].mean()),xy=(3e-7,80))
ax4.annotate('σ: {}'.format(error_100[:,1,1].std()),xy=(3e-7,70))
ax4.legend()

ax5.hist(error_100[:,0,2],label='SGS')
ax5.set_xlabel('Error')
ax5.set_ylabel('Frequency',labelpad=10)
ax5.annotate('μ: {}'.format(error_100[:,0,2].mean()),xy=(3e-7,80))
ax5.annotate('σ: {}'.format(error_100[:,0,2].std()),xy=(3e-7,70))
ax5.legend()

ax6.hist(error_100[:,1,2],label='SGS')
ax6.set_xlabel('Error')
ax6.annotate('μ: {}'.format(error_100[:,1,2].mean()),xy=(7e-7,80))
ax6.annotate('σ: {}'.format(error_100[:,1,2].std()),xy=(7e-7,70))
ax6.legend()

plt.show()

#%%

# GRAPHING HISTOGRAM USING PYPLOT
f1, ((ax1,ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(3,2, sharey=True, sharex=True,figsize=(12.8,9.6))
f1.suptitle('Relative Residual After Convergence for 100x100', fontsize=16)
ax1.hist(resid_100[:,0,0],label='None')
ax1.set_ylabel('Frequency',labelpad=10)
ax1.set_title('Steepest Descent')
ax1.annotate('μ: {}'.format(resid_100[:,0,0].mean()),xy=(5e-7,125))
ax1.annotate('σ: {}'.format(resid_100[:,0,0].std()),xy=(5e-7,110))
ax1.legend()

ax2.hist(resid_100[:,1,0],label='None')
ax2.set_title('Conjugate Gradient')
ax2.annotate('μ: {}'.format(resid_100[:,1,0].mean()),xy=(3e-6,110))
ax2.annotate('σ: {}'.format(resid_100[:,1,0].std()),xy=(3e-6,95))
ax2.legend()

ax3.hist(resid_100[:,0,1],label='Jacobi')
ax3.set_ylabel('Frequency',labelpad=10)
ax3.annotate('μ: {}'.format(resid_100[:,0,1].mean()),xy=(5e-7,125))
ax3.annotate('σ: {}'.format(resid_100[:,0,1].std()),xy=(5e-7,110))
ax3.legend()

ax4.hist(resid_100[:,1,1],label='Jacobi')
ax4.annotate('μ: {}'.format(resid_100[:,1,1].mean()),xy=(3e-6,110))
ax4.annotate('σ: {}'.format(resid_100[:,1,1].std()),xy=(3e-6,95))
ax4.legend()

ax5.hist(resid_100[:,0,2],label='SGS')
ax5.set_xlabel('Residual')
ax5.set_ylabel('Frequency',labelpad=10)
ax5.annotate('μ: {}'.format(resid_100[:,0,2].mean()),xy=(1e-6,125))
ax5.annotate('σ: {}'.format(resid_100[:,0,2].std()),xy=(1e-6,110))
ax5.legend()

ax6.hist(resid_100[:,1,2],label='SGS')
ax6.set_xlabel('Residual')
ax6.annotate('μ: {}'.format(resid_100[:,1,2].mean()),xy=(3e-6,110))
ax6.annotate('σ: {}'.format(resid_100[:,1,2].std()),xy=(3e-6,95))
ax6.legend()

plt.show()

#%%
# TESTING
tol = 1e-6

n = 10
L = generate_positive_L(n,4)
A = mat_mult(L,L.T)
x = generate_x(n)
b = generate_b(A,x)

SD_I_x_10, SD_I_err_res_10 = steepest_descent(A,b,'I',tol,x,record_trend=True)
SD_J_x_10, SD_J_err_res_10 = steepest_descent(A,b,'J',tol,x,record_trend=True)
SD_SGS_x_10, SD_SGS_err_res_10 = steepest_descent(A,b,'SGS',tol,x,record_trend=True)
CG_I_x_10, CG_I_err_res_10 = conjugate_gradient(A,b,'I',tol,x,record_trend=True)
CG_J_x_10, CG_J_err_res_10 = conjugate_gradient(A,b,'J',tol,x,record_trend=True)
CG_SGS_x_10, CG_SGS_err_res_10 = conjugate_gradient(A,b,'SGS',tol,x,record_trend=True)


n = 100
L = generate_positive_L(n,4)
A = mat_mult(L,L.T)
x = generate_x(n)
b = generate_b(A,x)

SD_I_x_100, SD_I_err_res_100 = steepest_descent(A,b,'I',tol,x,record_trend=True)
SD_J_x_100, SD_J_err_res_100 = steepest_descent(A,b,'J',tol,x,record_trend=True)
SD_SGS_x_100, SD_SGS_err_res_100 = steepest_descent(A,b,'SGS',tol,x,record_trend=True)
CG_I_x_100, CG_I_err_res_100 = conjugate_gradient(A,b,'I',tol,x,record_trend=True)
CG_J_x_100, CG_J_err_res_100 = conjugate_gradient(A,b,'J',tol,x,record_trend=True)
CG_SGS_x_100, CG_SGS_err_res_100 = conjugate_gradient(A,b,'SGS',tol,x,record_trend=True)

#%%

f1, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2, sharey='row', sharex='col',figsize=(12.8,9.6))
f1.suptitle('Error and Residual at each Iteration', fontsize=16)

ax1.set_title('10x10: Error')
ax1.plot(SD_I_err_res_10[:,0], label='SD I')
ax1.plot(SD_J_err_res_10[:,0], label='SD J')
ax1.plot(SD_SGS_err_res_10[:,0], label='SD SGS')
ax1.plot(CG_I_err_res_10[:,0], label='CG I')
ax1.plot(CG_J_err_res_10[:,0], label='CG J')
ax1.plot(CG_SGS_err_res_10[:,0], label='CG SGS')
ax1.set_ylabel('Error',labelpad=10)
ax1.legend()

ax2.set_title('100x100: Error')
ax2.plot(SD_I_err_res_100[:,0], label='SD I')
ax2.plot(SD_J_err_res_100[:,0], label='SD J')
ax2.plot(SD_SGS_err_res_100[:,0], label='SD SGS')
ax2.plot(CG_I_err_res_100[:,0], label='CG I')
ax2.plot(CG_J_err_res_100[:,0], label='CG J')
ax2.plot(CG_SGS_err_res_100[:,0], label='CG SGS')
ax2.legend()

ax3.set_title('10x10: Residual')
ax3.plot(SD_I_err_res_10[:,1], label='SD I')
ax3.plot(SD_J_err_res_10[:,1], label='SD J')
ax3.plot(SD_SGS_err_res_10[:,1], label='SD SGS')
ax3.plot(CG_I_err_res_10[:,1], label='CG I')
ax3.plot(CG_J_err_res_10[:,1], label='CG J')
ax3.plot(CG_SGS_err_res_10[:,1], label='CG SGS')
ax3.set_ylabel('Residual',labelpad=10)
ax3.set_xlabel('Iteration')
ax3.legend()

ax4.set_title('100x100: Residual')
ax4.plot(SD_I_err_res_100[:,1], label='SD I')
ax4.plot(SD_J_err_res_100[:,1], label='SD J')
ax4.plot(SD_SGS_err_res_100[:,1], label='SD SGS')
ax4.plot(CG_I_err_res_100[:,1], label='CG I')
ax4.plot(CG_J_err_res_100[:,1], label='CG J')
ax4.plot(CG_SGS_err_res_100[:,1], label='CG SGS')
ax4.set_xlabel('Iteration')
ax4.legend()

plt.show()

#%%

f1, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2, sharey='row', sharex='col',figsize=(12.8,9.6))
f1.suptitle('Log Error and Residual at each Iteration', fontsize=16)

ax1.set_title('10x10: Error')
ax1.plot(np.log(SD_I_err_res_10[:,0]), label='SD I')
ax1.plot(np.log(SD_J_err_res_10[:,0]), label='SD J')
ax1.plot(np.log(SD_SGS_err_res_10[:,0]), label='SD SGS')
ax1.plot(np.log(CG_I_err_res_10[:,0]), label='CG I')
ax1.plot(np.log(CG_J_err_res_10[:,0]), label='CG J')
ax1.plot(np.log(CG_SGS_err_res_10[:,0]), label='CG SGS')
ax1.set_ylabel('Log Error',labelpad=10)
ax1.legend()

ax2.set_title('100x100: Error')
ax2.plot(np.log(SD_I_err_res_100[:,0]), label='SD I')
ax2.plot(np.log(SD_J_err_res_100[:,0]), label='SD J')
ax2.plot(np.log(SD_SGS_err_res_100[:,0]), label='SD SGS')
ax2.plot(np.log(CG_I_err_res_100[:,0]), label='CG I')
ax2.plot(np.log(CG_J_err_res_100[:,0]), label='CG J')
ax2.plot(np.log(CG_SGS_err_res_100[:,0]), label='CG SGS')
ax2.legend()

ax3.set_title('10x10: Residual')
ax3.plot(np.log(SD_I_err_res_10[:,1]), label='SD I')
ax3.plot(np.log(SD_J_err_res_10[:,1]), label='SD J')
ax3.plot(np.log(SD_SGS_err_res_10[:,1]), label='SD SGS')
ax3.plot(np.log(CG_I_err_res_10[:,1]), label='CG I')
ax3.plot(np.log(CG_J_err_res_10[:,1]), label='CG J')
ax3.plot(np.log(CG_SGS_err_res_10[:,1]), label='CG SGS')
ax3.set_ylabel('Log Residual',labelpad=10)
ax3.set_xlabel('Iteration')
ax3.legend()

ax4.set_title('100x100: Residual')
ax4.plot(np.log(SD_I_err_res_100[:,1]), label='SD I')
ax4.plot(np.log(SD_J_err_res_100[:,1]), label='SD J')
ax4.plot(np.log(SD_SGS_err_res_100[:,1]), label='SD SGS')
ax4.plot(np.log(CG_I_err_res_100[:,1]), label='CG I')
ax4.plot(np.log(CG_J_err_res_100[:,1]), label='CG J')
ax4.plot(np.log(CG_SGS_err_res_100[:,1]), label='CG SGS')
ax4.set_xlabel('Iteration')
ax4.legend()

plt.show()

#%%

A = np.array([[5,7,6,5],[7,10,8,7],[6,8,10,9],[5,7,9,10]], dtype='float')
b = np.array([23,32,33,31],'float')

tol = 1e-6
SD_I_x, SD_I_i = steepest_descent(A,b,'I',tol,x,record_trend=False)
SD_J_x, SD_J_i = steepest_descent(A,b,'J',tol,x,record_trend=False)
SD_SGS_x, SD_SGS_i = steepest_descent(A,b,'SGS',tol,x,record_trend=False)
CG_I_x, CG_I_i = conjugate_gradient(A,b,'I',tol,x,record_trend=False)
CG_J_x, CG_J_i = conjugate_gradient(A,b,'J',tol,x,record_trend=False)
CG_SGS_x, CG_SGS_i = conjugate_gradient(A,b,'SGS',tol,x,record_trend=False)
print('\nNumber of iterations required for convergence:\n')
print('Steepest Descent')
print('No Preconditioning:',SD_I_i)
print('Jacobi:',SD_J_i)
print('Symmetric Gauss-Seidel:',SD_SGS_i)
print('\nConjugate Gradient')
print('No Preconditioning:',CG_I_i)
print('Jacobi:',CG_J_i)
print('Symmetric Gauss-Seidel:',CG_SGS_i)

#%%
# Gram-Schmidt
def generate_Q(n):
    Q = np.ndarray((n,n))
    V = np.random.random((n,n))
    Q[0] = V[0]
    for i in range(1,n):
        subtract = 0
        for j in range(i):
            subtract -= (sum(V[i]*Q[j])/sum(Q[j]**2))*Q[j]
        Q[i] = V[i] + subtract
    for i in range(n):
        Q[i] = Q[i]/v_2_norm(Q[i])
    return Q

def K_analysis(tol,n,K):
    # first dimension is trial,
    # second dimension is method
    # third dimension is error/resid/iteration count 
    out = np.ndarray((len(K),3,3))
    for i in tqdm(range(len(K)), desc="Running spectrum analysis"):
        Q = generate_Q(n)
        lamb = np.diag(np.linspace(1/K[i],1,num=n))
        A = mat_mult(Q.T,mat_mult(lamb,Q))
        x = generate_x(n)
        b = generate_b(A,x)
        
        R_x, R_i = steepest_descent(A,b,'I',tol,x,record_trend=False)
        SD_x, SD_i = steepest_descent(A,b,'SGS',tol,x,record_trend=False)
        CG_x, CG_i  = conjugate_gradient(A,b,'I',tol,x,record_trend=False)
        
        R_r = b - mat_mult(A,R_x)
        SD_r = b - mat_mult(A,SD_x)
        CG_r = b - mat_mult(A,CG_x)
        R_r_norm = np.sqrt(sum(R_r**2))
        SD_r_norm = np.sqrt(sum(SD_r**2))
        CG_r_norm = np.sqrt(sum(CG_r**2))
        R_err_norm = np.sqrt(sum((R_x-x)**2))
        SD_err_norm = np.sqrt(sum((SD_x-x)**2))
        CG_err_norm = np.sqrt(sum((CG_x-x)**2))

        out[i,0] = np.array([R_err_norm, R_r_norm, R_i])
        out[i,1] = np.array([SD_err_norm, SD_r_norm, SD_i])
        out[i,2] = np.array([CG_err_norm, CG_r_norm, CG_i])
    return out

n = 50
tol = 1e-6
K = [1,2,3,4,5,6,7,8,9,10,50,100,500,1000,10000,100000]
K = [1,2,3,4,5,6,7,8,9,10,50]
K = [1,2,3,4,5,6,7,8,9,10]
results = K_analysis(tol,n,K)

f1, ax1 = plt.subplots(1)
f1.suptitle('Number of iterations to Convergence at each Condition Number K', fontsize=16)

ax1.set_title('100x100')
ax1.plot(K,results[:,0,2], label='Non-stationary Richardson')
ax1.plot(K,results[:,1,2], label='Steepest Descent with SGS Preconditioner')
ax1.plot(K,results[:,2,2], label='Conjugate Gradient')

ax1.set_ylabel('Iterations',labelpad=10)
ax1.set_xlabel('Condition Number',labelpad=10)
ax1.legend()

plt.show()




