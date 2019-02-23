import numpy as np
from scipy.special import binom, logsumexp
import time

t0 = time.time()
N = 10
factorials = np.zeros(N+1).astype('float32')
factorials[0] = 0
factorials[1] = 0
for i in range(2,len(factorials)):
	factorials[i] = np.log(i) + factorials[i-1]

nfactorials = factorials.reshape(-1,1)
print 1		
denom = np.zeros((len(factorials), len(factorials)))
print 2
for i in range(len(factorials)):
	if i == 0:
		continue
	else:
		f = factorials[:i+1]
		denom[i,:i+1] = f + f[::-1]
print 3
combs = nfactorials - denom
combs[np.triu_indices(combs.shape[0], k=1)] = -np.inf
print t0 - time.time()

# print np.exp(combs).round()

P = 0.08366733118701218
l = 2
coeffs = combs[N-l,:N+1-l]
ks = np.arange(N+1-l)
# print ks[::-1]
# print np.exp(logsumexp(coeffs + ks*np.log(P) + ks[::-1]*np.log(1-P)))

probs = []
ks = []
kmNs = []
for k in range(N+1-l):
	ks.append(k)
	kmNs.append(N-l-k)
	probs.append(binom(N-l, k)* P**k * (1-P)**(N-l-k))
probs = np.array(probs)
# print probs
# print ks
# print kmNs

def getAllBinomialProbs(combs, P, invert_ks=False):
	if len(P.shape) == 0:
		P = np.expand_dims(P, 0)
	
	prob_terms = np.zeros(combs.shape+(P.shape[0],))
	prob_terms -= np.inf
	prob_terms[0,0] = 0
	print prob_terms.shape

	ks = np.arange(combs.shape[1])
	mP = np.log(1-np.exp(P))
	combs_expanded = np.expand_dims(combs,-1)
				
	for i in range(combs.shape[0]):
		if i == 0:
			continue
		else:
			for j in range(prob_terms.shape[-1]):					
				ks_ = ks[:i+1]
				if invert_ks:
					prob_terms[i,:i+1,j] = ks_[::-1]*P[j] + ks_*mP[j]
				else:
					prob_terms[i,:i+1,j] = ks_*P[j] + ks_[::-1]*mP[j]
	probs = combs_expanded + prob_terms
	return probs

P = np.log(np.array([0.2,0.8]))
binomials_ = getAllBinomialProbs(combs, P)[:,:,0]
print np.exp(binomials_)

binomials = np.zeros(combs.shape)-np.inf
for n in range(combs.shape[0]):
	for k in range(n+1):
		binomials[n,k] = combs[n,k] + k*P[0] + (n-k)*np.log(1-np.exp(P[0]))
print np.exp(binomials)

print np.allclose(binomials, binomials_)