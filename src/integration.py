import numpy as np 
import scipy
from scipy import integrate
from scipy import misc
from scipy.optimize import minimize_scalar

import time
def pcorrgcselcpt(cpt, csel, mu1, sigma1, mu2, sigma2):
	return lambda cp : np.exp(-(cpt-cp-mu1)/2 - (csel-cp-mu2)/2*sigma2)/(1+scipy.special.erf((cp-mu1)/np.sqrt(2)*sigma1))

t0 = time.time()
r = integrate.quad(pcorrgcselcpt(0.1,0.2,0.3,0.4,0.5,0.6),0,1)
print(time.time()-t0)
print(r)

def f(x):
	return -integrate.quad(pcorrgcselcpt(0.1,x,0.3,0.4,0.5,0.6),0,1)[0]


# print misc.derivative(f, 1.0, dx=1e-6)
print(minimize_scalar(f, bounds=(0, 1), method='bounded'))