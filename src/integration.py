import numpy as np 
import scipy
from scipy import integrate
from scipy import misc
from scipy.optimize import minimize_scalar
from scipy.stats import truncnorm
from scipy.stats import norm

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
# print(minimize_scalar(f, bounds=(0, 1), method='bounded'))


# def pQHcpt(cpt, pcp, sigma, upper, lower):
# 	return pcp * integrate.quad(pcptgivencp(cpt, sigma, upper, lower), lower, upper)[0]

def pcptgivencp(cpt, sigma, upper, lower):
	return lambda cp: (1 / np.sqrt(2 * np.pi) * (np.exp(- ((cpt - cp)**2) / (2 * (sigma**2))))) / (0.5 * sigma * (scipy.special.erf((upper - cp)/(sigma * np.sqrt(2))) - scipy.special.erf((lower - cp)/(sigma * np.sqrt(2)))))

def easypQHcpt(pcp, cpt, sigma, lower, upper):
	return pcp * integrate.quad(lambda cp: truncnorm.pdf(cpt, lower, upper, cp, sigma), lower, upper)[0]

def doubleIntegralExample():
	return integrate.quad(lambda y: integrate.quad(lambda x: x + 2 + y, 0, 0.5)[0], 0, 1)


# def truncated_normal(x, a, b, loc=0, scale=1):
# 	return norm.pdf(x, loc=loc, scale=scale)/ ((norm.cdf(b, loc=loc, scale=scale)) - (norm.cdf(a, loc=loc, scale=scale)))

def truncated_normal(x, a, b, loc=0, scale=1):
	return truncnorm.pdf(x, (a - loc)/scale, (b - loc)/scale, loc=loc, scale=scale)


# print(pcptgivencp(2, 1, 5, 1)(4))
# def _probAgivenB()


# print(pQHcpt(2, 1, 1, 2, 0))
# print(easypQHcpt(2, 1, 1, 2, 0))
# print(easypQHcpt(1, 3, 1, 0, 2))

# print("Answer:", doubleIntegralExample())

# print(truncnorm.pdf(x=5, a=1, b=5, loc=4, scale=1))

print(truncated_normal(2, 1, 5, 4))