
import numpy
import math

# define the function blocks

def prod(it):
    p = 1
    for n in it:
        p *= n
    return p


def Ufun(x, a, k, m):
    y = k*((x-a)**m)*(x > a)+k*((-x-a)**m)*(x < (-a))
    return y


# Sphere Function
def F1(x):
    s = numpy.sum(x**2)    
    return s


# Ackley Function
def F2(x):
    dim = len(x)
    o = -20*numpy.exp(-.2*numpy.sqrt(numpy.sum(x**2)/dim)) - \
        numpy.exp(numpy.sum(numpy.cos(2*math.pi*x))/dim)+20+numpy.exp(1)
    return o


# Penalized function
def F3(x):
    dim = len(x)
    o = .1*((numpy.sin(3*math.pi*x[1]))**2+sum((x[0:dim-2]-1)**2*(1+(numpy.sin(3*math.pi*x[1:dim-1]))**2)) +
            ((x[dim-1]-1)**2)*(1+(numpy.sin(2*math.pi*x[dim-1]))**2))+numpy.sum(Ufun(x, 5, 100, 4))
    return o


# Schwefel function
def F4(x):
    dim = len(x)
    o = 418.9829 * dim - numpy.sum(x * numpy.sin(numpy.sqrt(abs(x))))
    return o


 # Michalewicz Function
def F5(x):
    m = 10   # Steepness of the function; increase to make it harder
    dim = len(x)
    o = -numpy.sum(numpy.sin(x) * numpy.sin((numpy.arange(1, dim + 1) * x*2) / math.pi)*(2 * m))
    return o


# Rosenbrock Function
def F6(x):
    dim = len(x)
    o = sum(100 * (x[i + 1] - x[i]**2)**2 + (x[i] - 1)**2 for i in range(dim - 1))
    return o


# Levy Function
def F7(x):
    w = 1 + (x - 1) / 4
    term1 = numpy.sin(numpy.pi * w[0])**2
    term2 = sum((w[:-1] - 1)**2 * (1 + 10 * numpy.sin(numpy.pi * w[1:])**2))
    term3 = (w[-1] - 1)**2 * (1 + numpy.sin(2 * numpy.pi * w[-1])**2)
    return term1 + term2 + term3

