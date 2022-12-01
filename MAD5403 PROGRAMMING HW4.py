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
def bisection(func,a,b,tol=1e-6,nmax=1e12):
    if a > b:
        print("Please ensure 'a' is less than 'b' when entering search interval")
        print('a:',a)
        print('b:',b)
        return 'error', 'error', 'error'
    if func(a)*func(b) >= 0:
        print('Please pick different search interval, function evaluated at each endpoint is the same sign:')
        print('f(a):',func(a))
        print('f(b):',func(b))
        return 'error', 'error', 'error'
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
            print('bisection method has failed for given search interval.')
            return 'error', 'error', 'error'
        interval_size = np.append(interval_size,abs(b-a))
        error = abs(fc)
        if i == 0:
            error_i = np.array(error)
        else:
            error_i = np.append(error_i,error)
        i += 1
    if i == nmax:
        print('maximum number of iterations met')
        return 'error', 'error', 'error'
    return c, interval_size, error_i

# function has a maximum at x = 1, so try intervals from left of max and right of max
x1, interval_size1, error_i1 = bisection(test1,0,1)
iterations1 = len(interval_size1)
x2, interval_size2, error_i2 = bisection(test1,1,5)
iterations2 = len(interval_size2)

f2, ax2 = plt.subplots()
f2.suptitle('Log of Size of Bisection Interval at each Iteration', fontsize=16)

# ax2.set_title('First Root')
x_ticks = np.arange(max(iterations1,iterations2))
ax2.plot(np.log(interval_size1), label='First Root')
ax2.plot(np.log(interval_size2), label='Second Root')
ax2.set_ylabel('log(interval size)',labelpad=10)
ax2.set_xlabel('Iteration')
ax2.legend()

plt.show()

f3, ax3 = plt.subplots()
f3.suptitle('Error at each Iteration', fontsize=16)

# ax3.set_title('First Root')
x_ticks = np.arange(max(iterations1,iterations2))
ax3.plot(np.log(error_i1), label='First Root')
ax3.plot(np.log(error_i2), label='Second Root')
ax3.set_ylabel('Error',labelpad=10)
ax3.set_xlabel('Iteration')
ax3.legend()

plt.show()

#%%

def newton(func,x_0,tol=1e-6,nmax=1e12):
    f = sp.lambdify(x, func, "numpy")
    df = sp.lambdify(x, sp.diff(func), "numpy")
    if round(sp.diff(func).evalf(subs={x: x_0}),120) == 0: # need to write this mess so 1/sqrt 3 evals to 0
        print("ERROR: Derivative of f(x) at x_0 is 0, please pick a different initial guess.")
        return 'error', 'error', 'error'
    x_k = x_0
    i = 0
    error = tol + 1
    while i < nmax and abs(error) > tol:
        if round(sp.diff(func).evalf(subs={x: x_k}),120) == 0:
            print("ERROR: Derivative of f(x) at x_k is 0, Newton's method fails for this choice of initial guess")
            return 'error', 'error', 'error'
        x_k2 = x_k - f(x_k)/df(x_k)
        error = abs(x_k2 - x_k)/2
        x_k = x_k2
        if i == 0:
            error_i = np.array(error)
        else:
            error_i = np.append(error_i,error)
        i += 1
    if i == nmax:
        print('ERROR: tolerance was not reached in maximum number of iterations met')
    return x_k, i, error_i

x = sp.symbols('x')
_, _, _ = newton(x*sp.exp(-x)-.06064,1)
x3, iterations3, error_i3 = newton(x*sp.exp(-x)-.06064,.99)
x4, iterations4, error_i4 = newton(x*sp.exp(-x)-.06064,-1) # better guess of first root due to slope
x5, iterations5, error_i5 = newton(x*sp.exp(-x)-.06064,-.5) # better guess of first root due to slope
x6, iterations6, error_i6 = newton(x*sp.exp(-x)-.06064,5)
x7, iterations7, error_i7 = newton(x*sp.exp(-x)-.06064,3)

f4, ax4 = plt.subplots()
f4.suptitle('Error at each Iteration', fontsize=16)

# ax3.set_title('First Root')
x_ticks = np.arange(max(iterations1,iterations2))
ax4.plot(error_i4, label='Initial guess -1')
ax4.plot(error_i5, label='Initial guess -.5')
ax4.plot(error_i6, label='Initial guess 5')
ax4.plot(error_i7, label='Initial guess 3')
ax4.set_ylabel('Error',labelpad=10)
ax4.set_xlabel('Iteration')
ax4.legend()

plt.show()

#%%
def fixed_point(func,Φ,x_0,tol=1e-6,nmax=1e12):
    f = sp.lambdify(x, func, "numpy")
    dΦ = sp.lambdify(x, sp.diff(Φ), "numpy")
    Φ = sp.lambdify(x, Φ, "numpy")
    x_k = x_0
    i = 0
    error = tol + 1
    while i < nmax and abs(error) > tol:
        if abs(dΦ(x_k)) >= 1:
            print('selected Φ or initial guess is not appropriate for this problem')
            print('x_k:', x_k)
            print("|Φ'(x_k)|=",dΦ(x_k),">= 1")
            return 'error', 'error', 'error'
        x_k2 = Φ(x_k)
        error = abs(f(x_k2))
        x_k = x_k2
        if i == 0:
            error_i = np.array(error)
        else:
            error_i = np.append(error_i,error)
        i += 1
    if i == nmax:
        print('maximum number of iterations met')
        return 'error', 'error', 'error'
    return x_k, i , error_i

#%%
def test_phi_prime1(x):
    return .06064*np.exp(x)

def test_phi_prime2(x):
    return 1/x

line = np.linspace(.05,5,1000)

f5, ax5 = plt.subplots()
f5.suptitle("Checking Where |Φ'|<1", fontsize=16)

ax5.plot(line,test_phi_prime1(line), label="Φ'(x) = 0.06064e^x")
ax5.plot(line,test_phi_prime2(line), label="Φ'(x) = 1/x")
ax5.axhline(y=1, color='black',linestyle='--')
ax5.axhline(y=-1, color='black',linestyle='--')
ax5.set_ylabel("Φ'(x)",labelpad=10)
ax5.set_xlabel('x')
ax5.legend()

plt.show()

x = sp.symbols('x')
func = x*sp.exp(-x)-.06064

Φ1 = .06064*sp.exp(x) # for first root
# dΦ1 = sp.diff(Φ1)

Φ2 = sp.log(x) - sp.log(.06064) # for second root
# dΦ2 = sp.diff(Φ2)

x8, iterations8, error_i8 = fixed_point(func,Φ1,0)
x9, iterations9, error_i9 = fixed_point(func,Φ2,2)

f6, ax6 = plt.subplots()
f6.suptitle('Error at each Iteration', fontsize=16)

ax6.plot(error_i8, label='Φ = 0.06064e^x')
ax6.plot(error_i9, label='Φ = log(x) - log(.06064)')
ax6.set_ylabel('Error',labelpad=10)
ax6.set_xlabel('Iteration')
ax6.legend()

plt.show()




#%%
def test2(x):
    return x**3 - x - 6

line = np.linspace(-2.5,2.5,1000)

f7, ax7 = plt.subplots()
f7.suptitle('Plot of Second Test Function', fontsize=16)

ax7.plot(line,test2(line))
ax7.axhline(y=0, color='black',linestyle='--')
ax7.set_ylabel('y',labelpad=10)
ax7.set_xlabel('x')

plt.show()

#%%

x10, interval_size10, error_i10 = bisection(test2,-10,10)

f3, ax3 = plt.subplots()
f3.suptitle('Log Error at Each Iteration', fontsize=16)

# ax3.set_title('First Root')
x_ticks = np.arange(max(iterations1,iterations2))
ax3.plot(np.log(error_i10))
ax3.set_ylabel('Log Error',labelpad=10)
ax3.set_xlabel('Iteration')

plt.show()

# TODO: DEMONSTRATE CONVERGENCE ORDER FOR [-5,10]. OUTPUT NUMBER OF ITERATIONS


#%%

x = sp.symbols('x')
_, _, _ = newton(x**3-x-6, 1/sp.sqrt(3))
x11, iterations11, error_i11 = newton(x**3-x-6, .57735)
x12, iterations12, error_i12 = newton(x**3-x-6, 5) # better guess

f8, ax8 = plt.subplots()
f8.suptitle('Log Error at each Iteration', fontsize=16)

# x_ticks = np.arange(max(iterations11,iterations12))
ax8.plot(np.log(error_i11), label='Initial guess .57735')
ax8.plot(np.log(error_i12), label='Initial guess 5')
ax8.set_ylabel('Log Error',labelpad=10)
ax8.set_xlabel('Iteration')
ax8.legend()

plt.show()

# TODO: DEMONSTRATE CONVERGENCE ORDER FOR [-5,10]. OUTPUT NUMBER OF ITERATIONS

#%%
def test_phi_prime3(x):
    return 1/(3*(x+6)**(2/3))

line = np.linspace(-5.99,10,1000)

f9, ax9 = plt.subplots()
f9.suptitle("Checking Where |Φ'|<1", fontsize=16)

ax9.plot(line,test_phi_prime3(line), label="Φ'(x) = 1/(3*(x+6)^(2/3))")
ax9.axhline(y=1, color='black',linestyle='--')
ax9.axhline(y=-1, color='black',linestyle='--')
ax9.set_ylabel("Φ'(x)",labelpad=10)
ax9.set_xlabel('x')
ax9.legend()

plt.show()

#%%

x = sp.symbols('x')
func = x**3-x-6

Φ = (x+6)**(1/3) # for first root
dΦ = sp.diff(Φ)

x13, iterations13, error_i13 = fixed_point(func,Φ,-5)

f10, ax10 = plt.subplots()
f10.suptitle('Log Error at Each Iteration', fontsize=16)

ax10.plot(np.log(error_i13), label='Φ = (x+6)^(1/3)')
ax10.set_ylabel('Log Error',labelpad=10)
ax10.set_xlabel('Iteration')
ax10.legend()

plt.show()