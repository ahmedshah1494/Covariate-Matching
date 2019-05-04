import numpy as np 
import scipy
from scipy import integrate
import time
def pcorrgcselcpt(cpt, csel, mu1, sigma1, mu2, sigma2):
	return lambda cp : np.exp(-(cpt-cp-mu1)/2 - (csel-cp-mu2)/2*sigma2)/(1+scipy.special.erf((cp-mu1)/np.sqrt(2)*sigma1))

t0 = time.time()
r = integrate.quad(pcorrgcselcpt(0.1,0.2,0.3,0.4,0.5,0.6),0,1)
print(time.time()-t0)
print(r)