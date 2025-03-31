import numpy as np

N=2
G=6.67e-11
#1.5倍太阳质量
M=2.9835e30
c=3e8

time=0
def runge_kutta(y, x, dx, f):
    """ y is the initial value for y
        x is the initial value for x
        dx is the time step in x
        f is derivative of function y(t)
    """
    k1 = dx * f(y, x)
    k2 = dx * f(y + 0.5 * k1, x + 0.5 * dx)
    k3 = dx * f(y + 0.5 * k2, x + 0.5 * dx)
    k4 = dx * f(y + k3, x + dx)
    return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6.

def af(a,t):
    return -32*G**3*M**3*(1+(73*e**2)/24+(37*e**4)/96)/(5*a**3*c**5*(1-e**2)**3.5)
def ef(e,t):
    return -152*G**3*M**3*e*(1+(121*e**2)/304)/(15*c**5*a**4*(1-e**2)*2.5)
e=0.8
a=1e7
E=[]
A=[]
dt=0.2
time=0
def af(a,t):
    return -32*G**3*M**3*(1+(73*e**2)/24+(37*e**4)/96)/(5*a**3*c**5*(1-e**2)**3.5)
def ef(e,t):
    return -152*G**3*M**3*e*(1+(121*e**2)/304)/(15*c**5*a**4*(1-e**2)*2.5)
while (a>1000):
    time+=dt
    a=runge_kutta(a,time,dt,af)
    e=runge_kutta(e,time,dt,ef)
    A.append(a)
    E.append(e)
    
