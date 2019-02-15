import sys
import torch
import itertools
from numbers import Number
import numpy as np
from torch import nn
import cvxpy as cp
import torch.distributions as D
import torch.nn.functional as F
import cvxpy.atoms.elementwise as E
import cvxpy.atoms.affine as A
from scipy.special import comb, logsumexp
from scipy.stats import expon
from data_processing import *


def log_sum_exp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation

    value.exp().sum(dim, keepdim).log()
    """
    # TODO: torch.max(value, dim=None) threw an error at time of writing
    if 'tensor' not in str(type(value)).lower():
	    value = torch.tensor(value, requires_grad=True)

    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0),
                                       dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        if torch.isinf(m):
        	sum_exp = torch.tensor(0.0)
        else:
	        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, Number):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)

class VectorMultinomial(object):
	"""docstring for VectorMultinomial"""
	def __init__(self, data=None, probs=None):
		super(VectorMultinomial, self).__init__()
		if data is not None:
			self.data = data		
			self.probs = [np.histogram(data[:,[i]], bins=data[:,[i]].max()+1)[0].astype('float32') for i in range(data.shape[1])]			
		elif probs is not None:
			self.probs = probs
		self.probs = [(torch.from_numpy(x/np.sum(x))) for x in self.probs]
		print self.probs
	
	def sample(self):
		return torch.tensor([torch.argmax(D.Multinomial(1,p).sample()) for p in self.probs])
	
	def sample_n(self, n):
		return torch.tensor([self.sample() for i in range(n)])

	def pdf(self,x):	
		if len(x.shape) < 2:
			x = x.unsqueeze(0)
		return torch.tensor([torch.sum(torch.tensor([torch.log(self.probs[i][y[i]] if y[i] < len(self.probs[i]) else torch.tensor(0.0)) for i in range(len(y))])) for y in x])

class VectorUniform(object):
	"""docstring for VectorMultinomial"""
	def __init__(self, data=None, probs=None):
		super(VectorUniform, self).__init__()
		if data is not None:
			self.data = data.numpy()				
			self.probs = [torch.tensor([1.0/(self.data[:,[i]].max()+1)]*(self.data[:,[i]].max()+1)) for i in range(data.shape[1])]			
			print self.probs
		elif probs is not None:
			self.probs = probs
	
	def sample(self):
		return torch.tensor([torch.argmax(D.Multinomial(1,p).sample()) for p in self.probs])
	
	def sample_n(self, n):
		return torch.tensor([self.sample() for i in range(n)])

	def pdf(self,x):	
		if len(x.shape) < 2:
			x = x.unsqueeze(0)
		return torch.tensor([torch.sum(torch.tensor([torch.log(self.probs[i][y[i]] if y[i] < len(self.probs[i]) else torch.tensor(0.0)) for i in range(len(y))])) for y in x])
		
class GaussianNoisyChannel(object):
	"""docstring for NoisyChannel"""
	def __init__(self, mean, cov, clip_range=None):
		super(NoisyChannel, self).__init__()
		self.noiseDist = D.MultivariateNormal(mean, cov)
		self.clip_range = clip_range

	def addNoise(self, x):
		noisy = x + self.noiseDist.sample_n(x.size()[0] if len(x.size()) >= 2 else 1).long()
		if self.clip_range is not None:
			noisy = noisy.transpose(0,1)
			for i in range(noisy.shape[0]):
				noisy[i] = torch.clamp(noisy[i], min=self.clip_range[i][0], max=self.clip_range[i][1])
			noisy = noisy.transpose(0,1)
		return noisy

	def __call__(self, x):
		return self.addNoise(x)

	def pdf(self, ct, c):
		diff = ct - c
		return self.noiseDist.log_prob(diff.float())

class IdentityNoisyChannel(object):
	"""docstring for IdentityNoisyChannel"""
	def __init__(self):
		super(IdentityNoisyChannel, self).__init__()
	
	def addNoise(self, x):
		return x

	def __call__(self, x):
		return self.addNoise(x)

	def pdf(self, ct, c):
		nz = torch.nonzero(ct - c)[:,[0]].squeeze()		
		p = torch.ones(max(ct.shape[0], c.shape[0]))
		p[nz] = 0
		# print ct - c, nz, p
		return torch.log(p)
		
class RandomNoisyChannel(object):
	"""docstring for RandomNoisyChannel"""
	def __init__(self, true_prob, VC):
		super(RandomNoisyChannel, self).__init__()
		self.true_prob = true_prob
		self.noisy_prob = (1 - self.true_prob)/(len(VC)-1)
		self.VC = VC

	def addNoise(self,x):
		noisy = torch.zeros(x.shape).long()
		flips = [np.random.binomial(1, self.true_prob) for i in range(x.shape[0])]
		for i in range(len(flips)):
			if flips[i]:
				noisy[i] = x[i]
			else:				
				vc = self.VC[torch.min(self.VC != x[i], dim=1)[0]]
				noisy[i] = vc[np.random.randint(vc.shape[0])]
		return noisy

	def __call__(self, x):
		return self.addNoise(x)

	def pdf(self, ct, c):
		if len(ct.shape) == 1 and len(c.shape) == 1:
			if torch.min(ct == c):
				return np.log(self.true_prob)
			else:
				return np.log(self.noisy_prob)
		else:
			nz = torch.nonzero(ct - c)[:,[0]].squeeze()		
			p = torch.zeros(max(ct.shape[0], c.shape[0])) + self.true_prob
			p[nz] = self.noisy_prob
			# print ct - c, nz, p
			return torch.log(p)

class Experiment(object):
	"""docstring for Experiment"""	

	def __init__(self, VC, G=None, Q=None):
		super(Experiment, self).__init__()

		self.VC = VC
		rangeVC = [(VC[:,[i]].min(), VC[:,[i]].max()) for i in range(self.VC.shape[1])]
		if G is None:
			G = self.VC#.repeat(10,1)
		else:
			G = torch.from_numpy(G).long()

		if Q is None:
			Q = self.VC.repeat(100,1)
		else:
			Q = torch.from_numpy(Q).long()
		print G.shape, Q.shape

		self.PQ = VectorMultinomial(Q)
		self.PG = VectorMultinomial(G)

		# self.H = GaussianNoisyChannel(torch.zeros(Q.shape[1]), torch.tensor([[0.5,0],[0,5]]), rangeVC)
		# self.H = IdentityNoisyChannel()
		self.H = RandomNoisyChannel(.90, self.VC)
		self.P_H = self.H.pdf
		# self.J = IdentityNoisyChannel()
		# self.J = GaussianNoisyChannel(torch.zeros(G.shape[1]), torch.tensor([[0.5,0],[0,7]]), rangeVC)
		self.J = RandomNoisyChannel(.90, self.VC)
		self.P_J = self.J.pdf


		self.Qt = self.H(Q)
		self.Gt = self.J(G)		
		
		PQ_Q = self.PQ.pdf(self.VC)
		self.P_QHct = lambda ct: log_sum_exp(PQ_Q + self. P_H(ct,self.VC))
		self.P_QHcgct = lambda c,ct: (self.PQ.pdf(c)+self.P_H(ct,c)-self.P_QHct(ct))

		PG_G = self.PG.pdf(self.VC)		
		self.P_GJct = lambda ct: log_sum_exp(PG_G + self.P_J(ct,self.VC))
		self.P_GJcgct = lambda c,ct: self.PG.pdf([c])+self.P_J(ct,c)-self.P_GJct(ct)		

		self.Q = Q
		self.G = G

class UniqueMatchExperiment(Experiment):
	"""docstring for UniqueMatchExperiment"""
	def __init__(self, G, Q=None):
		rangeVC = [np.arange(G[:,[i]].min(), G[:,[i]].max()+1) for i in range(G.shape[1])]		
		VC = torch.tensor([[x,y] for x,y in itertools.product(*rangeVC)])

		super(UniqueMatchExperiment, self).__init__(VC, G, Q)

		N = self.G.shape[0]
		self.PCorrgCqCsel = lambda cq, csel: torch.log(torch.exp(self.P_J(csel,cq)) - torch.exp(self.P_J(csel,cq) + N*(torch.log(1-torch.exp(self.P_GJct(csel))))))
		self.PCorrgCtqCsel = lambda ctq, csel: log_sum_exp(self.P_QHcgct(self.VC, ctq)+self.PCorrgCqCsel(self.VC, csel))
		self.hatC = lambda ctq: self.VC[torch.argmax(torch.tensor([self.PCorrgCtqCsel(ctq,csel) for csel in self.VC]))]

	def test(self, ids):
		hatC_cache = {}
		total_correct = 0
		total_cov_correct = 0
		total = 0
		confmat = np.zeros((self.G.shape[0], self.G.shape[0]))
		for i in range(len(self.Qt)):			
			cq = self.Q[i]
			ctq = self.Qt[i]
			csel = hatC_cache.get(ctq, self.hatC(ctq))	
			# print torch.min(self.Gt == csel, dim=1)

			shortlist = torch.arange(0,self.Gt.shape[0])[torch.min(self.Gt == csel, dim=1)[0]]
			choice = -1
			correct = 0
			cov_correct = 0
			if len(shortlist) > 0:
				choice = shortlist[np.random.randint(0,shortlist.shape[0])]
				# choice = self.G[choice]
				# correct = int(torch.equal(choice,cq))
				correct = int(choice == ids[i])
				cov_correct = int(torch.equal(self.G[choice], cq))
			total_correct += correct
			total_cov_correct += cov_correct
			total += 1
			print i, cq, ctq, csel, ids[i], int(choice), self.G[choice], float(total_correct)/total, float(total_cov_correct)/total
			
			# confentry = confmat.setdefault(str(list(cq.numpy())), {})
			# confentry[str(list(choice.numpy()))] = confentry.get(str(list(choice.numpy())),0)+1		
			if choice >= 0:
				confmat[ids[i], choice] += 1		
		print float(total_correct)/total
		np.save(confmat, 'confmat.npy')
		# for k in sorted(confmat.keys(), key=lambda x: eval(x)):
		# 	print k, float(confmat[k].get(k,0))/sum(confmat[k].values()), sorted([(kk, confmat[k][kk]) for kk in confmat[k]], key=lambda x:x[1])[-1]
		# print {k: sorted([(kk, confmat[k][kk]) for kk in confmat[k]], key=lambda x:x[1])[-1] for k in confmat}

class VerificationExperiment(Experiment):
	"""docstring for VerificationExperiment"""
	def __init__(self, G, Q=None):		
		rangeVC = [np.arange(G[:,[i]].min(), G[:,[i]].max()+1) for i in range(G.shape[1])]		
		VC = torch.tensor([[x,y] for x,y in itertools.product(*rangeVC)])
		print rangeVC
		super(VerificationExperiment, self).__init__(VC, G, Q=Q)

		# self.P_match = lambda ctq, ctg: log_sum_exp([self.P_H(ctq,cq)+self.P_J(ctg,cq)+self.PQ.pdf(cq)  for cq in self.VC])
		# self.P_mismatch = lambda ctq, ctg: log_sum_exp([self.P_H(ctq,cq)+self.P_J(ctg,cg)+self.PQ.pdf(cq)+self.PG.pdf(cg)  for (cq,cg) in itertools.product(self.VC,self.VC)])				
		self.P_mismatch = np.zeros((self.VC.shape[0], self.VC.shape[0]))
		self.P_match = np.zeros((self.VC.shape[0], self.VC.shape[0]))
		
		cov_pairs = [x for x in itertools.product(range(self.VC.shape[0]),range(self.VC.shape[0]))]
		P_H = np.array([self.P_H(self.VC[i],self.VC[j]) for (i, j) in cov_pairs]).reshape(self.VC.shape[0], self.VC.shape[0])
		P_J = np.array([self.P_J(self.VC[i],self.VC[j]) for (i, j) in cov_pairs]).reshape(self.VC.shape[0], self.VC.shape[0])
		PG = np.array([self.PG.pdf(self.VC[i]) for i in range(self.VC.shape[0])])
		PQ = np.array([self.PQ.pdf(self.VC[i]) for i in range(self.VC.shape[0])])
			
		for (i_ctq, i_ctg) in cov_pairs:
			p_match = []
			p_mismatch = []
			for i_cq in range(self.VC.shape[0]):
				p_h = P_H[i_ctq,i_cq]
				p_q = PQ[i_cq]
				p_match.append(p_h + P_J[i_ctg,i_cq] + p_q)
				for i_cg in range(self.VC.shape[0]):
					p_mismatch.append(p_h+P_J[i_ctg,i_cg]+p_q+PG[i_cg])
			
			self.P_mismatch[i_ctq, i_ctg] = logsumexp(p_mismatch)
			self.P_match[i_ctq, i_ctg] = logsumexp(p_match)		

		self.P_mismatch = self.P_mismatch.reshape(-1)
		self.P_match = self.P_match.reshape(-1)
		print np.sum(np.exp(self.P_mismatch)), np.sum(np.exp(self.P_match))

		self.r = cp.Variable(self.VC.shape[0] * self.VC.shape[0])
		E.log.log(self.r)
		fas = self.r * np.exp(self.P_mismatch)
		frs = (1-self.r) * np.exp(self.P_match)
		self.FA =  A.sum.sum(fas)
		self.FR = A.sum.sum(frs)		
		objective = cp.Minimize(self.FA)
		constraints = [self.FA == self.FR, self.r >= 0.000, self.r <= 1.000]
		prob = cp.Problem(objective, constraints)
		print("Optimal value", prob.solve())
		print("Optimal var", self.r.value)

		self.r = self.r.value
		# print self.r[self.r >= 1]
		# print self.r[self.r <= 0]
		self.r[self.r > 1] = 1
		self.r[self.r < 0] = 0
		print np.sum(self.r * np.exp(self.P_mismatch))
		print np.sum((1-self.r) * np.exp(self.P_match))		
		self.r = self.r.reshape(self.VC.shape[0], self.VC.shape[0])

	def test(self, qset=None, ids=None, labels=None):
		if qset is None:
			assert(self.Q.shape[0] == self.G.shape[0])
			qset = [x for x in itertools.product(range(self.Q.shape[0]),range(self.G.shape[0]))]			
		if labels is None:
			labels = [int(ids[i]==ids[j]) for (i,j) in itertools.product(range(ids.shape[0]),range(ids.shape[0]))]

		assert(len(qset) == len(labels))

		correct_pos = 0.0
		correct_neg = 0.0
		total_pos = 0.0
		total_neg = 0.0
		i = 0
		for (i_ctq, i_ctg) in qset:
			[i_ctq_vc] = [j for j in range(len(self.VC)) if np.allclose(self.VC[j], self.Qt[i_ctq])]
			[i_ctg_vc] = [j for j in range(len(self.VC)) if np.allclose(self.VC[j], self.Gt[i_ctg])]
			
			flip = np.random.binomial(1, self.r[i_ctq_vc, i_ctg_vc])
			
			if labels[i]:
				correct_pos += int(flip == labels[i])
				total_pos += 1	
			else:
				correct_neg += int(flip == labels[i])
				total_neg += 1
			if i > 0 and min(total_pos, total_neg) > 0 and i % 1000 == 0:				
				print i, self.Q[i_ctq], self.Qt[i_ctq], self.G[i_ctg], self.Gt[i_ctg], self.r[i_ctq_vc, i_ctg_vc], flip, labels[i]
				print correct_pos/total_pos,correct_neg/total_neg, (correct_pos+correct_neg)/(total_pos+total_neg)
			i+=1
		print correct/total

def binomialCoeff(N, K, P):
	return np.log(comb(N,K))+K*np.log(P)+(N-K)*np.log(1-P)

class OneofLExperiment(Experiment):
	"""docstring for OneofLExperiment"""
	def __init__(self, G, Q):
		rangeVC = [np.arange(G[:,[i]].min(), G[:,[i]].max()+1) for i in range(G.shape[1])]		
		VC = torch.tensor([[x,y] for x,y in itertools.product(*rangeVC)])

		super(OneofLExperiment, self).__init__(VC, G, Q)
		self.N = self.G.shape[0]

		P_K_mt = {}
		P_l_mt = {}
		inner_sum_mt = {}
		P_Kglctq = lambda K,l,csel: P_K_mt.setdefault((K,l,csel), binomialCoeff(self.N-1, K-l, self.P_GJct(csel)))
		P_lgLctq = lambda l,L,csel,cq: P_l_mt.setdefault((l,L,csel,cq), binomialCoeff(L,l, self.P_J(csel,cq)))
		inner_sum = lambda L,K,l, csel, cq: inner_sum_mt.setdefault((L,K,l), 
											np.logaddexp(P_Kglctq(K,l,csel), P_lgLctq(l,L,csel,cq)))

		self.PCorrgCqCsel = lambda cq, csel: logsumexp([np.logaddexp(P_Kglctq(K,L,csel), P_lgLctq(K,L,csel,cq))+np.log(1.0/K) 
												for l in range(min(L,K)) for (L,K) in zip(range(N),range(N))])
		self.PCorrgCtqCsel = lambda ctq, csel: logsumexp(self.P_QHcgct(self.VC, ctq)+self.PCorrgCqCsel(self.VC, csel))

id_map,_ = loadVCMeta('vox1_meta.csv')
# ids, qdata = loadVCdata('iden_split.txt', gdata)
labels, qset = loadVCVeriData('veri_test.txt', id_map)
# G = genGallery([(0,1),(0,35)], 1)
# ids, Q = genQueries(G, 1)
Q = qset[:,[0]].squeeze()
G = qset[:,[1]].squeeze()
print Q.shape, G.shape
e = OneofLExperiment(G, Q)
# print e.P_J(torch.tensor([0,0]), e.VC)
# print e.P_H(torch.tensor([[0,0],[0,1],[0,33]]), torch.tensor([[0,33]]))
print e.PCorrgCtqCsel(np.array([[0,5]]),np.array([[0,5]]))
e.test(ids=ids)
# e.test(qset=zip(range(Q.shape[0]), range(G.shape[0])), labels=labels)