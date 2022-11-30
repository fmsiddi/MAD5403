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

# ax1.set_title('First Root')
ax1.plot(line,test1(line))
ax1.set_ylabel('y',labelpad=10)
ax1.set_xlabel('x')

plt.show()

#%%
def bisection(func,a,b,tol=1e-6,nmax=1e12):
    if a > b:
        print("Please ensure 'a' is less than 'b' when entering search interval")
        print('a:',a)
        print('b:',b)
        return ''
    if func(a)*func(b) >= 0:
        print('Please pick different search interval, function evaluated at each endpoint is the same sign:')
        print('f(a):',func(a))
        print('f(b):',func(b))
        return ''
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
            break
        interval_size = np.append(interval_size,abs(b-a))
        error = abs(fc)
        if i == 0:
            error_i = np.array(error)
        else:
            error_i = np.append(error_i,error)
        i += 1
    if i == nmax:
        print('maximum number of iterations met')
        return ''
    return c, interval_size, error_i

# function has a maximum at x = 1, so try intervals from left of max and right of max
x1, interval_size1, error_i1 = bisection(test1,-10,1)
iterations1 = len(interval_size1)
x2, interval_size2, error_i2 = bisection(test1,1,10)
iterations2 = len(interval_size2)

#%%
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
#%%

def newton(func,x_0,tol=1e-6,nmax=1e12):
    f = sp.lambdify(x, func, "numpy")
    df = sp.lambdify(x, sp.diff(func), "numpy")
    if df(x_0) == 0:
        print("ERROR: Derivative of f(x) at x_0 is 0, please pick a different initial guess.")
        return 'error', 'error'
    x_k = x_0
    i = 0
    error = tol + 1
    while i < nmax and abs(error) > tol:
        if df(x_k) == 0:
            print("ERROR: Derivative of f(x) at x_k is 0, Newton's method fails for this choice of initial guess")
            return 'error', 'error'
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
newton_fail, newton_fail_i_fail = newton(x*sp.exp(-x)-.06064,1)
x3, iterations3, error_i3 = newton(x*sp.exp(-x)-.06064,.99)
x4, iterations4, error_i4 = newton(x*sp.exp(-x)-.06064,-1) # better guess of first root due to slope
x5, iterations5, error_i5 = newton(x*sp.exp(-x)-.06064,-.5) # better guess of first root due to slope
x6, iterations6, error_i6 = newton(x*sp.exp(-x)-.06064,5)
x7, iterations7, error_i7 = newton(x*sp.exp(-x)-.06064,3)

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
            return ''
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
        return ''
    return x_k, i , error_i

x = sp.symbols('x')
func = x*sp.exp(-x)-.06064

Φ1 = .06064*sp.exp(x) # for first root
# dΦ1 = sp.diff(Φ1)

Φ2 = sp.log(x) - sp.log(.06064) # for second root
# dΦ2 = sp.diff(Φ2)

x8, iterations8, error_i8 = fixed_point(func,Φ1,0)
x9, iterations9, error_i9 = fixed_point(func,Φ2,2)

#%%
def test_phi_prime1(x):
    return .06064*np.exp(x)

def test_phi_prime2(x):
    return 1/x

line = np.linspace(.05,5,1000)

plt.axhline(y=1, color='black',linestyle='--')
plt.axhline(y=-1, color='black',linestyle='--')
plt.plot(line,test_phi_prime1(line))
plt.plot(line,test_phi_prime2(line))
plt.show()

#%%
def test2(x):
    return x**3 - x -6

line = np.linspace(-2.5,2.5,1000)

plt.axhline(y=0, color='black',linestyle='--')
plt.plot(line,test2(line))
plt.show()

#%%

x = bisection(test2,-10,10)
print(x)

#%%
x = sp.symbols('x')
test = newton(x**3-x-6,0)
print(test)

#%%
def test_phi_prime(x):
    return 1/(3*(x+6)**(2/3))

line = np.linspace(-10,10,1000)

plt.axhline(y=1, color='black',linestyle='--')
plt.axhline(y=-1, color='black',linestyle='--')
plt.plot(line,test_phi_prime(line))
plt.show()

#%%

x = sp.symbols('x')
func = x**3-x-6

Φ = (x+6)**(1/3) # for first root
dΦ = sp.diff(Φ)

test = fixed_point(func,Φ,-5)
print(test)