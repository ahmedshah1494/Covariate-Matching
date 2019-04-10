import numpy as np
from scipy.special import logsumexp

def getAllBinomialProbs_(combs, P, invert_ks=False):
		if len(P.shape) == 0:
			P = np.expand_dims(P, 0)
		
		# print 1
		
		prob_terms = np.empty(combs.shape+(P.shape[0],), dtype='float32')
		prob_terms[:] = -np.inf		
		prob_terms[0,0] = 0
		# print 2, time.time()-t0
		
		ks = np.arange(combs.shape[1])
		mP = np.log(1-np.exp(P))
		combs_expanded = np.expand_dims(combs,-1)		
		# print 3,time.time()-t0

		for i in range(combs.shape[0]):
			ncs = combs[i]
			if i == 0:
				continue
			else:
				for j in range(prob_terms.shape[-1]):					
					ks_ = ks[:i+1]
					ncks = ncs[ks_]
					if invert_ks:
						prob_terms[i,:i+1,j] = ncks + ks_[::-1]*P[j] + ks_*mP[j]
					else:
						prob_terms[i,:i+1,j] = ncks + ks_*P[j] + ks_[::-1]*mP[j]
		# print 4, time.time()-t0
		# t0 = time.time()
		# probs = combs_expanded + prob_terms
		# print 5, time.time()-t0
		return prob_terms

def PCorrgCtqCsel_((P_L, N, max_L, max_K, P_J, P_GJ, P_QHcgct, combs)):
		sum_terms = []

		bins_P_GJ = getAllBinomialProbs_(combs, P_GJ).squeeze()		
		bins_P_J = getAllBinomialProbs_(combs, P_J)		
		rand_i = np.random.randint(bins_P_J.shape[0])		

		inner_sums = []
		for L in range(1, max_L+1):
			k_range = max_K+1
			l_range = lambda k: min(L, k)+1						
			ps = []
			
			for k in range(1,k_range):
				# bin_L = self.getBinomialProbInRange(L, l_range(k), P_J)				
				log_k = np.log(k)
				for l in range(1,l_range(k)):
					# p = self.binomial(self.N-l, k-l, P_GJ) + bin_L[l] - log_k
					p = bins_P_GJ[N-l, k-l] + bins_P_J[L,l] - log_k
					ps.append(p)
					# print bins_P_GJ[self.N-l, k-l], bins_P_J[L,l][:3], log_k, p[:3]				
			inner_sums.append(logsumexp(ps, axis=0))		
		inner_sums = np.array(inner_sums)		
		sum_terms = P_L[1:].reshape(-1,1) + inner_sums		
		return logsumexp(logsumexp(sum_terms,axis=0) + P_QHcgct)