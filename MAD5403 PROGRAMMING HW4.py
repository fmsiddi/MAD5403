import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
plt.style.use('seaborn')

#%%
def test1(x):
    return x*np.exp(-x)-.06064

line = np.linspace(0,6,1000)

f1, ax1 = plt.subplots()
f1.suptitle('Plot of First Test Function', fontsize=16)

ax1.plot(line,test1(line))
ax1.axhline(y=0, color='black',linestyle='--')
ax1.set_ylabel('y',labelpad=10)
ax1.set_xlabel('x')

plt.show()

#%%
def bisection(func,a,b,tol=1e-6,nmax=1e12,sol=None):
    if a > b:
        print("ERROR: Please ensure 'a' is less than 'b' when entering search interval")
        print('a:',a)
        print('b:',b)
        return 'error', 'error', 'error'
    if func(a)*func(b) >= 0:
        print('ERROR: Please pick different search interval, function evaluated at each endpoint is the same sign:')
        print('f(a):',func(a))
        print('f(b):',func(b))
        return 'error', 'error', 'error', 'error', 'error'
    interval_size = np.array(abs(b-a))
    i = 0
    error = tol + 1
    while i < nmax and abs(error) > tol:
        c = (a + b)/2
        fc = func(c)
        if fc*func(a) < 0:
            b = c
        elif fc*func(b) < 0:
            a = c
        else:
            print('ERROR: Bisection method has failed for given search interval.')
            return 'error', 'error', 'error', 'error', 'error'
        interval_size = np.append(interval_size,abs(b-a))
        if sol == None:
            error = abs(fc)
        else:
            error = abs(c - sol)
        if i == 0:
            error_i = np.array(error)
            x_k = np.array(c)
            fx_k = np.array(fc)
        else:
            error_i = np.append(error_i,error)
            x_k = np.append(x_k,c)
            fx_k = np.append(fx_k,fc)
        i += 1
    if i == nmax:
        print('ERROR: Maximum number of iterations met')
        return 'error', 'error', 'error', 'error', 'error'
    return c, interval_size, error_i, x_k, fx_k

#%%

# function has a maximum at x = 1, so try intervals from left of max and right of max
x1, interval_size1, error_i1, x_k1, fx_k1 = bisection(test1,0,1)
iterations1 = len(interval_size1)
x2, interval_size2, error_i2, x_k2, fx_k2 = bisection(test1,1,5)
iterations2 = len(interval_size2)

f2, ((ax2,ax3),(ax4,ax5)) = plt.subplots(2,2, sharex=True,figsize=(12.8,9.6))
f2.suptitle('Bisection Method Results for First Test Function\nInitial Search Intervals: Blue = [0,1]  Green = [1,5]', fontsize=14)

ax2.set_title('Log of Size of Interval at Each Iteration')
x_ticks = np.arange(max(iterations1,iterations2))
ax2.plot(np.log(interval_size1), label='First Root')
ax2.plot(np.log(interval_size2), label='Second Root')
ax2.set_ylabel('log(interval size)')
ax2.legend()

ax3.set_title('Log Error at Each Iteration')
ax3.plot(np.log(error_i1), label='First Root')
ax3.plot(np.log(error_i2), label='Second Root')
ax3.set_ylabel('Log Error')
ax3.legend()

ax4.set_title('x_k at Each Iteration')
ax4.plot(x_k1, label='First Root')
ax4.plot(x_k2, label='Second Root')
ax4.set_ylabel('x_k')
ax4.set_xlabel('Iteration')
ax4.legend()

ax5.set_title('f(x_k) at Each Iteration')
ax5.plot(fx_k1, label='First Root')
ax5.plot(fx_k2, label='Second Root')
ax5.set_ylabel('f(x_k)')
ax5.set_xlabel('Iteration')
ax5.legend()

plt.tight_layout()

#%%

def newton(func,x_0,tol=1e-6,nmax=1e12):
    f = sp.lambdify(x, func, "numpy")
    df = sp.lambdify(x, sp.diff(func), "numpy")
    if round(sp.diff(func).evalf(subs={x: x_0}),120) == 0: # need to write this mess so 1/sqrt 3 evals to 0
        print("ERROR: Derivative of f(x) at x_0 is 0, please pick a different initial guess.")
        return 'error', 'error', 'error', 'error', 'error'
    x_k_track = np.array(x_0)
    fx_k_track = np.array(f(x_0))
    error_i = np.array(abs(f(x_0)))
    x_k = x_0
    i = 0
    error = tol + 1
    while i < nmax and abs(error) > tol:
        if round(sp.diff(func).evalf(subs={x: x_k}),120) == 0:
            print("ERROR: Derivative of f(x) at x_k is 0, Newton's method fails for this choice of initial guess")
            return 'error', 'error', 'error', 'error', 'error'
        x_k = x_k - f(x_k)/df(x_k)
        x_k_track = np.append(x_k_track,x_k)
        error = abs(f(x_k))
        fx_k_track = np.append(fx_k_track,f(x_k))
        error_i = np.append(error_i,abs(f(x_k)))
        i += 1
    if i == nmax:
        print('ERROR: tolerance was not reached in maximum number of iterations met')
        return 'error', 'error', 'error', 'error', 'error'
    return x_k, i, error_i, x_k_track, fx_k_track

#%%
x = sp.symbols('x')
_, _, _, _, _ = newton(x*sp.exp(-x)-.06064,1) # will result in error so not storing variables
x3, iterations3, error_i3, x_k3, fx_k3 = newton(x*sp.exp(-x)-.06064,.99)
x4, iterations4, error_i4, x_k4, fx_k4 = newton(x*sp.exp(-x)-.06064,-1) # better guess of first root due to slope
x5, iterations5, error_i5, x_k5, fx_k5 = newton(x*sp.exp(-x)-.06064,-.5) # better guess of first root due to slope
x6, iterations6, error_i6, x_k6, fx_k6 = newton(x*sp.exp(-x)-.06064,5)
x7, iterations7, error_i7, x_k7, fx_k7 = newton(x*sp.exp(-x)-.06064,3)

f3, ((ax6,ax7,ax8)) = plt.subplots(1,3,figsize=(12.8,4.8))
f3.suptitle('Newton Method Results for First Test Function\nInitial guess: 0.99', fontsize=16)

ax6.set_title('Error at Each Interval')
# x_ticks = np.arange(max(iterations1,iterations2))
ax6.plot(error_i3)
ax6.set_ylabel('Error')
ax6.set_xlabel('Iteration')

ax7.set_title('x_k at Each Iteration')
ax7.plot(x_k3)
ax7.set_ylabel('x_k')
ax7.set_xlabel('Iteration')

ax8.set_title('f(x_k) at Each Iteration')
ax8.plot(fx_k3)
ax8.set_ylabel('f(x_k)')
ax8.set_xlabel('Iteration')

plt.tight_layout()


f4, ((ax9,ax10,ax11)) = plt.subplots(1,3,figsize=(12.8,4.8))
f4.suptitle('(Better) Newton Method Results for First Test Function', fontsize=16)

ax9.set_title('Error at Each Interval')
# x_ticks = np.arange(max(iterations1,iterations2))
ax9.plot(error_i4, label='Initial guess: -1')
ax9.plot(error_i5, label='Initial guess: -.5')
ax9.plot(error_i6, label='Initial guess: 5')
ax9.plot(error_i7, label='Initial guess: 3')
ax9.set_ylabel('Error',labelpad=10)
ax9.set_xlabel('Iteration')
ax9.legend()

ax10.set_title('x_k at Each Iteration')
ax10.plot(x_k4, label='Initial guess: -1')
ax10.plot(x_k5, label='Initial guess: -.5')
ax10.plot(x_k6, label='Initial guess: 5')
ax10.plot(x_k7, label='Initial guess: 3')
ax10.set_ylabel('x_k')
ax10.set_xlabel('Iteration')
ax10.legend()

ax11.set_title('f(x_k) at Each Iteration')
ax11.plot(fx_k4, label='Initial guess: -1')
ax11.plot(fx_k5, label='Initial guess: -.5')
ax11.plot(fx_k6, label='Initial guess: 5')
ax11.plot(fx_k7, label='Initial guess: 3')
ax11.set_ylabel('f(x_k)')
ax11.set_xlabel('Iteration')
ax11.legend()

plt.tight_layout()

#%%
def fixed_point(func,Φ,x_0,tol=1e-6,nmax=1e12):
    f = sp.lambdify(x, func, "numpy")
    dΦ = sp.lambdify(x, sp.diff(Φ), "numpy")
    Φ = sp.lambdify(x, Φ, "numpy")
    x_k_track = np.array(x_0)
    fx_k_track = np.array(f(x_0))
    error_i = np.array(abs(f(x_0)))
    x_k = x_0
    i = 0
    error = tol + 1
    while i < nmax and abs(error) > tol:
        if abs(dΦ(x_k)) >= 1:
            print('ERROR: selected Φ or initial guess is not appropriate for this problem')
            print('x_k:', x_k)
            print("|Φ'(x_k)|=",dΦ(x_k),">= 1")
            return 'error', 'error', 'error', 'error', 'error'
        x_k = Φ(x_k)
        x_k_track = np.append(x_k_track,x_k)
        error = abs(f(x_k))
        fx_k_track = np.append(fx_k_track,f(x_k))
        error_i = np.append(error_i,error)
        i += 1
    if i == nmax:
        print('ERROR: maximum number of iterations met')
        return 'error', 'error', 'error', 'error', 'error'
    return x_k, i , error_i, x_k_track, fx_k_track

#%%
def test_phi_prime1(x):
    return .06064*np.exp(x)

def test_phi_prime2(x):
    return 1/x

line = np.linspace(.05,5,1000)

f5, ax12 = plt.subplots()
f5.suptitle("Checking Where |Φ'|<1", fontsize=16)

ax12.plot(line,test_phi_prime1(line), label="Φ'(x) = 0.06064e^x")
ax12.plot(line,test_phi_prime2(line), label="Φ'(x) = 1/x")
ax12.axhline(y=1, color='black',linestyle='--')
ax12.axhline(y=-1, color='black',linestyle='--')
ax12.set_ylabel("Φ'(x)",labelpad=10)
ax12.set_xlabel('x')
ax12.legend()

plt.show()

x = sp.symbols('x')
func = x*sp.exp(-x)-.06064

Φ1 = .06064*sp.exp(x) # for first root
# dΦ1 = sp.diff(Φ1)

Φ2 = sp.log(x) - sp.log(.06064) # for second root
# dΦ2 = sp.diff(Φ2)

x8, iterations8, error_i8, x_k8, fx_k8 = fixed_point(func,Φ1,0)
x9, iterations9, error_i9, x_k9, fx_k9 = fixed_point(func,Φ2,2)

f6, ((ax13,ax14,ax15)) = plt.subplots(1,3,figsize=(12.8,4.8))
f6.suptitle('Fixed Point Method Results for First Test Function\nΦ1 Initial Guess: 0\nΦ2 Initial Guess: 2', fontsize=16)

ax13.set_title('Error at Each Iteration')
ax13.plot(error_i8, label='Φ = 0.06064e^x')
ax13.plot(error_i9, label='Φ = log(x) - log(.06064)')
ax13.set_ylabel('Error',labelpad=10)
ax13.set_xlabel('Iteration')
ax13.legend()

ax14.set_title('x_k at Each Iteration')
ax14.plot(x_k8, label='Φ = 0.06064e^x')
ax14.plot(x_k9, label='Φ = log(x) - log(.06064)')
ax14.set_ylabel('x_k')
ax14.set_xlabel('Iteration')
ax14.legend()

ax15.set_title('f(x_k) at Each Iteration')
ax15.plot(fx_k8, label='Φ = 0.06064e^x')
ax15.plot(fx_k9, label='Φ = log(x) - log(.06064)')
ax15.set_ylabel('f(x_k)')
ax15.set_xlabel('Iteration')
ax15.legend()

plt.tight_layout()










#%%
def test2(x):
    return x**3 - x - 6

line = np.linspace(-2.5,2.5,1000)

f7, ax16 = plt.subplots()
f7.suptitle('Plot of Second Test Function', fontsize=16)

ax16.plot(line,test2(line))
ax16.axhline(y=0, color='black',linestyle='--')
ax16.set_ylabel('y',labelpad=10)
ax16.set_xlabel('x')

plt.show()

#%%
# BISECTION METHOD FOR SECOND TEST FUNCTION

x10, interval_size10, error_i10, x_k10, fx_k10 = bisection(test2,-10,10, sol=2)

f8, ((ax17,ax18,ax19)) = plt.subplots(1,3,figsize=(12.8,4.8))
f8.suptitle('Bisection Method Results for Second Test Function', fontsize=16)

ax17.set_title('Log Error at Each Iteration')
ax17.plot(np.log(error_i10))
ax17.set_ylabel('Log Error')
ax17.set_xlabel('Iteration')

ax18.set_title('x_k at Each Iteration')
ax18.plot(x_k10)
ax18.set_ylabel('x_k')
ax18.set_xlabel('Iteration')

ax19.set_title('f(x_k) at Each Iteration')
ax19.plot(fx_k10)
ax19.set_ylabel('f(x_k)')
ax19.set_xlabel('Iteration')

plt.tight_layout()

#%%
# TODO: DEMONSTRATE CONVERGENCE ORDER FOR [-5,10]. OUTPUT NUMBER OF ITERATIONS

theoretical_iter = np.log2((10--5)/(10**-6))-1
print('theoretical number of iterations:',theoretical_iter)
x101, interval_size101, error_i101, x_k101, fx_k101 = bisection(test2,-5,10,tol=1e-6, sol=2)
print('iterations:', len(x_k101))

f_bonus1, ax_bonus1 = plt.subplots()
# f_bonus1.suptitle('Bisection Method Results for First Test Function\nInitial Search Intervals: Blue = [0,1]  Green = [1,5]', fontsize=14)

ax_bonus1.set_title('Log of Size of Interval at Each Iteration')
ax_bonus1.plot(np.log(interval_size101), label='First Root')
ax_bonus1.set_ylabel('log(interval size)')
ax_bonus1.legend()

#%%
# NEWTON'S METHOD FOR SECOND TEST FUNCTION

x = sp.symbols('x')
_, _, _, _, _ = newton(x**3-x-6, 1/sp.sqrt(3)) # will result in error so not storing variables
x11, iterations11, error_i11, x_k11, fx_k11 = newton(x**3-x-6, .57735)
x12, iterations12, error_i12, x_k12, fx_k12 = newton(x**3-x-6, 5) # better guess

f9, ((ax20,ax21,ax22)) = plt.subplots(1,3,figsize=(12.8,4.8))
f9.suptitle('Newton Method Results for Second Test Function\nInitial guess: .57735', fontsize=16)

ax20.set_title('Log Error at Each Iteration')
ax20.plot(np.log(error_i11), label='Initial guess: .57735')
ax20.set_ylabel('Log Error',labelpad=10)
ax20.set_xlabel('Iteration')

ax21.set_title('x_k at Each Iteration')
ax21.plot(x_k11, label='Initial guess: .57735')
ax21.set_ylabel('x_k')
ax21.set_xlabel('Iteration')

ax22.set_title('f(x_k) at Each Iteration')
ax22.plot(fx_k11, label='Initial guess: .57735')
ax22.set_ylabel('f(x_k)')
ax22.set_xlabel('Iteration')

plt.tight_layout()


f10, ((ax23,ax24,ax25)) = plt.subplots(1,3,figsize=(12.8,4.8))
f10.suptitle('Newton Method Results for Second Test Function\nInitial guess: 5', fontsize=16)

ax23.set_title('Log Error at Each Iteration')
ax23.plot(np.log(error_i12), label='Initial guess: 5')
ax23.set_ylabel('Log Error',labelpad=10)
ax23.set_xlabel('Iteration')

ax24.set_title('x_k at Each Iteration')
ax24.plot(x_k12, label='Initial guess: 5')
ax24.set_ylabel('x_k')
ax24.set_xlabel('Iteration')

ax25.set_title('f(x_k) at Each Iteration')
ax25.plot(fx_k12, label='Initial guess: 5')
ax25.set_ylabel('f(x_k)')
ax25.set_xlabel('Iteration')

plt.tight_layout()

#%%
# TODO: DEMONSTRATE CONVERGENCE ORDER FOR [-5,10]. OUTPUT NUMBER OF ITERATIONS

def newton_quad_conv(func,x_0,sol,tol=1e-6,nmax=1e12):
    f = sp.lambdify(x, func, "numpy")
    df = sp.lambdify(x, sp.diff(func), "numpy")
    if round(sp.diff(func).evalf(subs={x: x_0}),120) == 0: # need to write this mess so 1/sqrt 3 evals to 0
        print("ERROR: Derivative of f(x) at x_0 is 0, please pick a different initial guess.")
        return 'error', 'error', 'error', 'error', 'error'
    x_k_track = np.array(x_0)
    fx_k_track = np.array(f(x_0))
    error_i = np.array(abs(sol-x_0))
    x_k = x_0
    i = 0
    error = tol + 1
    while i < nmax and abs(error) > tol:
        if round(sp.diff(func).evalf(subs={x: x_k}),120) == 0:
            print("ERROR: Derivative of f(x) at x_k is 0, Newton's method fails for this choice of initial guess")
            return 'error', 'error', 'error', 'error', 'error'
        x_k = x_k - f(x_k)/df(x_k)
        x_k_track = np.append(x_k_track,x_k)
        error = abs(sol-x_k)
        fx_k_track = np.append(fx_k_track,f(x_k))
        error_i = np.append(error_i,abs(f(x_k)))
        i += 1
    if i == nmax:
        print('ERROR: tolerance was not reached in maximum number of iterations met')
        return 'error', 'error', 'error', 'error', 'error'
    return x_k, i, error_i, x_k_track, fx_k_track

x = sp.symbols('x')
x_bonus2, iterations_bonus2, error_i_bonus2, x_k_bonus2, fx_k_bonus2 = newton_quad_conv(x**3-x-6, .57735, sol=2, tol=1e-9)
error_i_bonus2_prev = np.roll(error_i_bonus2,1)
error_i_bonus2_prev_sq = error_i_bonus2_prev**2

f_bonus2, ax_bonus2 = plt.subplots()

ax_bonus2.set_title('Log of Error and Squared Previous Error Scaled by M=5')
ax_bonus2.plot(np.log(error_i_bonus2[2:]), label='e_k')
ax_bonus2.plot(np.log(5*error_i_bonus2_prev_sq[2:]), label='5 * e_(k-1)^2')
ax_bonus2.set_ylabel('log(error)')
ax_bonus2.set_xlabel('Iteration')
ax_bonus2.legend()

plt.tight_layout()

#%%
# FIXED POINT METHOD FOR SECOND TEST FUNCTION

def test_phi_prime3(x):
    return 1/(3*(x+6)**(2/3))

line = np.linspace(-5.99,10,1000)

f11, ax26 = plt.subplots()
f11.suptitle("Checking Where |Φ'(x)|<1", fontsize=16)

ax26.plot(line,test_phi_prime3(line), label="Φ'(x) = 1/(3*(x+6)^(2/3))")
ax26.axhline(y=1, color='black',linestyle='--')
ax26.axhline(y=-1, color='black',linestyle='--')
ax26.set_ylabel("Φ'(x)",labelpad=10)
ax26.set_xlabel('x')
ax26.legend()

plt.show()

#%%

x = sp.symbols('x')
func = x**3-x-6

Φ = (x+6)**(1/3)
dΦ = sp.diff(Φ)

x13, iterations13, error_i13, x_k13, fx_k13 = fixed_point(func,Φ,-5)

f12, ((ax27,ax28,ax29)) = plt.subplots(1,3,figsize=(12.8,4.8))
f12.suptitle('Fixed Point Method Results for Second Test Function\nΦ(x) = (x+6)^(1/3), Initial guess: -5', fontsize=16)

ax27.set_title('Log Error at Each Iteration')
ax27.plot(np.log(error_i13), label='Φ = (x+6)^(1/3)')
ax27.set_ylabel('Log Error',labelpad=10)
ax27.set_xlabel('Iteration')
ax27.legend()

ax28.set_title('x_k at Each Iteration')
ax28.plot(x_k13, label='Φ = (x+6)^(1/3)')
ax28.set_ylabel('x_k')
ax28.set_xlabel('Iteration')
ax28.legend()

ax29.set_title('f(x_k) at Each Iteration')
ax29.plot(fx_k13, label='Φ = (x+6)^(1/3)')
ax29.set_ylabel('f(x_k)')
ax29.set_xlabel('Iteration')
ax29.legend()

plt.tight_layout()

#%%
# CORRECTNESS TESTS

x14, interval_size14, error_i14, x_k14, fx_k14 = bisection(test1,1,10,tol=1e-6)
iterations14 = len(interval_size14)-1
print('Number of iterations for Bisection Correctness Test:', iterations14)

f13, (ax30,ax31) = plt.subplots(1,2,figsize=(12.8,9.6))
f13.suptitle('Bisection Method Results for Correctness Test', fontsize=14)

ax30.set_title('x_k at Each Iteration')
ax30.plot(x_k14, label='First Root')
ax30.set_ylabel('x_k')
ax30.set_xlabel('Iteration')

ax31.set_title('f(x_k) at Each Iteration')
ax31.plot(fx_k14, label='First Root')
ax31.set_ylabel('f(x_k)')
ax31.set_xlabel('Iteration')

plt.tight_layout()

#%%
x = sp.symbols('x')
x15, iterations15, error_i15, x_k15, fx_k15 = newton(x*sp.exp(-x)-.06064,2)
print('Number of iterations for Newton Correctness Test:',iterations15)

f14, (ax32,ax33) = plt.subplots(1,2,figsize=(12.8,4.8))
f14.suptitle('Newton Method Results for Correctness Test', fontsize=14)

x_ticks = np.arange(iterations15+1)
ax32.set_title('x_k at Each Iteration')
ax32.plot(x_ticks,x_k15)
ax32.set_ylabel('x_k')
ax32.set_xlabel('Iteration')

ax33.set_title('f(x_k) at Each Iteration')
ax33.plot(x_ticks,fx_k15)
ax33.set_ylabel('f(x_k)')
ax33.set_xlabel('Iteration')

plt.sca(ax32)
plt.xticks(x_ticks)
plt.sca(ax33)
plt.xticks(x_ticks)

plt.tight_layout()

#%%
x = sp.symbols('x')
func = x*sp.exp(-x)-.06064

Φ = x*sp.exp(-x)+x-.06064 
dΦ = sp.diff(Φ)

x16, iterations16, error_i16, x_k16, fx_k16 = fixed_point(func,Φ,2)

f12, (ax34,ax35) = plt.subplots(1,2,figsize=(12.8,4.8))
f12.suptitle('Fixed Point Method Results for Correctness Test', fontsize=16)

ax34.set_title('x_k at Each Iteration')
ax34.plot(x_k16, label='Φ = xe^(-x) + x - .06064')
ax34.set_ylabel('x_k')
ax34.set_xlabel('Iteration')
ax34.legend()

ax35.set_title('f(x_k) at Each Iteration')
ax35.plot(fx_k16, label='Φ = xe^(-x) + x - .06064')
ax35.set_ylabel('f(x_k)')
ax35.set_xlabel('Iteration')
ax35.legend()

plt.tight_layout()