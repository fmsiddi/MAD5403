import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
plt.style.use('seaborn')

def test1(x):
    return x*np.exp(-x)-.06064

line = np.linspace(0,6,1000)

plt.axhline(y=0, color='black',linestyle='--')
plt.plot(line,test1(line))
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
        error = abs(fc)
        i += 1
    if i == nmax:
        print('maximum number of iterations met')
        return ''
    return c

x1 = bisection(test1,-1,3)
print(x1)

x2 = bisection(test1,1,6)
print(x2)
#%%

def newton(func,x_0,tol=1e-6,nmax=1e12):
    f = sp.lambdify(x, func, "numpy")
    df = sp.lambdify(x, sp.diff(func), "numpy")
    if df(x_0) == 0:
        print("Derivative of f(x) at x_0 is 0, please pick a different initial guess.")
        return ''
    x_k = x_0
    i = 0
    error = tol + 1
    while i < nmax and abs(error) > tol:
        if df(x_k) == 0:
            print("Derivative of f(x) at x_k is 0, Newton's method fails for this choice of initial guess")
            break
        x_k2 = x_k - f(x_k)/df(x_k)
        error = abs(x_k2 - x_k)/2
        x_k = x_k2
        i += 1
    if i == nmax:
        print('maximum number of iterations met')
        return ''
    return x_k

x = sp.symbols('x')
test = newton(x*sp.exp(-x)-.06064,5)
print(test)

#%%
def fixed_point(func,Φ,x_0,tol=1e-6,nmax=1e12):
    f = sp.lambdify(x, func, "numpy")
    dΦ = sp.lambdify(x, sp.diff(Φ), "numpy")
    Φ = sp.lambdify(x, Φ, "numpy")
    x_k = x_0
    i = 0
    error = tol + 1
    while i < nmax and abs(error) > tol:
        x_k2 = Φ(x_k)
        error = abs(f(x_k2))
        x_k = x_k2
        i += 1
    if i == nmax:
        print('maximum number of iterations met')
        return ''
    return x_k

x = sp.symbols('x')
func = x*sp.exp(-x)-.06064

Φ1 = .06064*sp.exp(x) # for first root
dΦ1 = sp.diff(Φ1)

Φ2 = sp.log(x) - sp.log(.06064) # for second root
dΦ2 = sp.diff(Φ2)


test = fixed_point(func,Φ2,1)
print(test)

#%%
def test_phi(x):
    return 1/x

line = np.linspace(.05,6,1000)

plt.axhline(y=1, color='black',linestyle='--')
plt.plot(line,test_phi(line))
plt.show()