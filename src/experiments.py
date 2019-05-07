import sys
import itertools
import time
from numbers import Number
import numpy as np
# import torch
# from torch import nn
# import torch.distributions as D
# import torch.nn.functional as F

import cvxpy as cp
import cvxpy.atoms.elementwise as E
import cvxpy.atoms.affine as A

from scipy.special import logsumexp
from scipy.stats import expon, multinomial, multivariate_normal, norm
from concurrent.futures import ThreadPoolExecutor, wait
from multiprocessing.pool import Pool

from data_processing import *
from utils import *
from distributions import *
from channels import *

class Experiment(object):
    """docstring for Experiment"""  

    def __init__(self, VC, G=None, Q=None, true_prob=0.9):
        super(Experiment, self).__init__()
        self.reset(VC, G, Q, true_prob=true_prob)

    def reset(self, VC, G=None, Q=None, true_prob=0.9):
        self.VC = VC        
        rangeVC = [(VC[:,[i]].min(), VC[:,[i]].max()) for i in range(self.VC.shape[1])]

        if G is None:
            self.G = self.VC#.repeat(10,1)
        else:
            self.G = G

        if Q is None:
            self.Q = self.VC.repeat(100,1)
        else:
            self.Q = Q

        self.filter_VC = lambda keep_values: self.VC[[i for i in range(len(self.VC)) 
                                                    if len(keep_values[np.all(keep_values == self.VC[i], axis=1)]) > 0]]

        self.VC_G = self.filter_VC(self.G)      

        
        self.PQ = VectorMultinomial(Q)
        self.PG = VectorMultinomial(G)

        # self.H = GaussianNoisyChannel(torch.zeros(Q.shape[1]), torch.tensor([[0.5,0],[0,5]]), rangeVC)
        # self.H = IdentityNoisyChannel()
        self.H = RandomNoisyChannel(true_prob, self.VC)
        # self.H = IndependentRandomNoisyChannel(np.array([true_prob]*self.VC.shape[1]), self.VC)
        self.P_H = self.H.pdf
        # self.J = IdentityNoisyChannel()
        # self.J = GaussianNoisyChannel(torch.zeros(G.shape[1]), torch.tensor([[0.5,0],[0,7]]), rangeVC)
        self.J = RandomNoisyChannel(true_prob, self.VC)
        # self.J = IndependentRandomNoisyChannel(np.array([true_prob]*self.VC.shape[1]), self.VC)
        self.P_J = self.J.pdf


        self.Qt = self.H(Q)
        self.Gt = self.J(G)     
        
        self.VC_Gt = self.filter_VC(self.Gt)

        PQ_Q = self.PQ.pdf(self.VC)
        self.P_QHct = lambda ct: logsumexp(PQ_Q + self. P_H(ct,self.VC))
        self.P_QHcgct = lambda c,ct: (self.PQ.pdf(c)+self.P_H(ct,c)-self.P_QHct(ct))

        self.PG_G = self.PG.pdf(self.VC_G)      
        self.P_GJct = lambda ct: logsumexp(self.PG_G + self.P_J(ct,self.VC_G))
        self.P_GJcgct = lambda c,ct: self.PG.pdf([c])+self.P_J(ct,c)-self.P_GJct(ct)        

        self.Q = Q
        self.G = G

    def updateGallery(self, remove_idices):
        self.G = np.delete(self.G, remove_idices, axis=0)
        self.Gt = np.delete(self.Gt, remove_idices, axis=0)

        self.PG = VectorMultinomial(self.G)

        self.VC_G = self.filter_VC(self.G)
        self.PG_G = self.PG.pdf(self.VC_G)

        self.VC_Gt = self.filter_VC(self.Gt)

class UniqueMatchExperiment(Experiment):
    """docstring for UniqueMatchExperiment"""
    def __init__(self, G, Q=None, true_prob=0.9):
        rangeVC = [np.arange(G[:,[i]].min(), G[:,[i]].max()+1) for i in range(G.shape[1])]      
        VC = np.array([[x,y] for x,y in itertools.product(*rangeVC)])

        super(UniqueMatchExperiment, self).__init__(VC, G, Q, true_prob=true_prob)

        N = self.G.shape[0]
        self.PCorrgCqCsel = lambda cq, csel: np.log(np.exp(self.P_J(csel,cq)) - np.exp(self.P_J(csel,cq) + N*(np.log(1-np.exp(self.P_GJct(csel))))))
        self.PCorrgCtqCsel = lambda ctq, csel: logsumexp(self.P_QHcgct(self.VC, ctq)+self.PCorrgCqCsel(self.VC, csel))
        self.hatC = lambda ctq: self.VC_Gt[np.argmax([self.PCorrgCtqCsel(ctq,csel) for csel in self.VC_Gt])]

    def test(self, ids, verbose=True, naive=False):
        hatC_cache = {}
        total_correct = 0
        total_cov_correct = 0
        total = 0
        confmat = np.zeros((self.G.shape[0], self.G.shape[0]))
        for i in range(len(self.Qt)):           
            cq = self.Q[i]
            ctq = self.Qt[i]
            if naive:
                csel = ctq
            else:
                csel = hatC_cache.setdefault(tuple(ctq), self.hatC(ctq))
            # print torch.min(self.Gt == csel, dim=1)

            shortlist = np.arange(0,self.Gt.shape[0])[np.all(self.Gt == csel, axis=1)]
            choice = -1
            correct = 0
            cov_correct = 0
            if len(shortlist) > 0:
                choice = shortlist[np.random.randint(0,shortlist.shape[0])]
                # choice = self.G[choice]
                # correct = int(torch.equal(choice,cq))
                correct = int(choice == ids[i])
                cov_correct = int(np.allclose(self.G[choice], cq))
            total_correct += correct
            total_cov_correct += cov_correct
            total += 1
            if verbose:
                print (i, cq, ctq, csel, ids[i], int(choice), self.G[choice], float(total_correct)/total, float(total_cov_correct)/total)
            
            # confentry = confmat.setdefault(str(list(cq.numpy())), {})
            # confentry[str(list(choice.numpy()))] = confentry.get(str(list(choice.numpy())),0)+1       
            if choice >= 0:
                confmat[ids[i], choice] += 1        
        return float(total_correct)/total
        np.save('confmat.npy', confmat)
        # for k in sorted(confmat.keys(), key=lambda x: eval(x)):
        #   print k, float(confmat[k].get(k,0))/sum(confmat[k].values()), sorted([(kk, confmat[k][kk]) for kk in confmat[k]], key=lambda x:x[1])[-1]
        # print {k: sorted([(kk, confmat[k][kk]) for kk in confmat[k]], key=lambda x:x[1])[-1] for k in confmat}

class VerificationExperiment(Experiment):
    """docstring for VerificationExperiment"""
    def __init__(self, VC, G, Q=None, true_prob=0.9):       
        # rangeVC = [np.arange(G[:,[i]].min(), G[:,[i]].max()+1) for i in range(G.shape[1])]        
        # VC = np.array([[x,y] for x,y in itertools.product(*rangeVC)])
        # print rangeVC
        super(VerificationExperiment, self).__init__(VC, G, Q=Q, true_prob=true_prob)

        # self.P_match = lambda ctq, ctg: log_sum_exp([self.P_H(ctq,cq)+self.P_J(ctg,cq)+self.PQ.pdf(cq)  for cq in self.VC])
        # self.P_mismatch = lambda ctq, ctg: log_sum_exp([self.P_H(ctq,cq)+self.P_J(ctg,cg)+self.PQ.pdf(cq)+self.PG.pdf(cg)  for (cq,cg) in itertools.product(self.VC,self.VC)])                
        self.P_mismatch = np.zeros((self.VC.shape[0], self.VC.shape[0]))
        self.P_match = np.zeros((self.VC.shape[0], self.VC.shape[0]))

        # self.PQ = UniformVectorMultinomial(self.VC)
        # self.PG = UniformVectorMultinomial(self.VC)
        
        cov_pairs = [x for x in itertools.product(range(self.VC.shape[0]),range(self.VC.shape[0]))]
        P_H = np.array([self.P_H(self.VC[i],self.VC[j]) for (i, j) in cov_pairs]).reshape(self.VC.shape[0], self.VC.shape[0])
        P_J = np.array([self.P_J(self.VC[i],self.VC[j]) for (i, j) in cov_pairs]).reshape(self.VC.shape[0], self.VC.shape[0])
        PG = self.PG.pdf(self.VC)
        PQ = self.PQ.pdf(self.VC)
        print(np.exp(P_J))
        print (np.sum(np.exp(P_J), axis=1), np.sum(np.exp(P_H), axis=1))

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
        print (np.sum(np.exp(self.P_mismatch)), np.sum(np.exp(self.P_match)))

        self.r = cp.Variable(self.VC.shape[0] * self.VC.shape[0])
        # E.log.log(self.r)
        fas = self.r * np.exp(self.P_mismatch)
        frs = (1-self.r) * np.exp(self.P_match)
        self.FA =  A.sum.sum(fas)
        self.FR = A.sum.sum(frs)        
        objective = cp.Minimize(self.FR)
        constraints = [self.FA == self.FR, self.r >= 0.000, self.r <= 1.000]
        # constraints = [1-self.FR == 0.654, self.r >= 0.000, self.r <= 1.000]
        prob = cp.Problem(objective, constraints)
        prob.solve()
        # print("Optimal value", prob.solve())
        # print("Optimal var", self.r.value)

        self.r = self.r.value
        # print self.r[self.r >= 1]
        # print self.r[self.r <= 0]
        self.r[self.r > 1] = 1
        self.r[self.r < 0] = 0      
        print ("FA =",np.sum(self.r * np.exp(self.P_mismatch)))
        print ("FR =",np.sum((1-self.r) * np.exp(self.P_match))     )
        self.r = self.r.reshape(self.VC.shape[0], self.VC.shape[0])

    def test(self, qset=None, ids=None, labels=None, verbose=True, naive=False):
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
            # flip = int(np.all(self.Qt[i_ctq] == self.Gt[i_ctg]))
            if labels[i]:
                if naive:
                    correct_pos += (i_ctq_vc == i_ctg_vc)
                else:
                    correct_pos += int(flip == labels[i])
                total_pos += 1  
            else:
                if naive:
                    correct_neg += (i_ctq_vc != i_ctg_vc)
                else:
                    correct_neg += int(flip == labels[i])
                total_neg += 1
            if verbose and i > 0 and min(total_pos, total_neg) > 0 and i % 1000 == 0:               
                print (i, self.Q[i_ctq], self.Qt[i_ctq], self.G[i_ctg], self.Gt[i_ctg], self.r[i_ctq_vc, i_ctg_vc], flip, labels[i])
                print (correct_pos/total_pos,correct_neg/total_neg, (correct_pos+correct_neg)/(total_pos+total_neg))
            i+=1
        return (correct_pos+correct_neg)/(total_pos+total_neg)

class OneofLExperiment(Experiment):
    """docstring for OneofLExperiment"""
    def __init__(self, G, Q, max_L=None, true_prob=0.9):
        rangeVC = [np.arange(G[:,[i]].min(), G[:,[i]].max()+1) for i in range(G.shape[1])]      
        VC = np.array([[x,y] for x,y in itertools.product(*rangeVC)])

        super(OneofLExperiment, self).__init__(VC, G, Q, true_prob=true_prob)
        
        self.N = self.G.shape[0]
        if max_L is None:
            self.max_L = self.N
        else:
            self.max_L = min(max_L, self.N)

        self.max_K = np.max([self.Gt[np.all(self.Gt == c, axis=1)].shape[0] for c in self.VC])

        # self.max_K = self.N

        P_K_mt = {}
        P_l_mt = {}
        inner_sum_mt = {}
    
        factorials = np.zeros(self.N+1).astype('float32')
        factorials[0] = 0
        factorials[1] = 0
        for i in range(2,len(factorials)):
            factorials[i] = np.log(i) + factorials[i-1]

        nfactorials = factorials.reshape(-1,1)
        denom = np.zeros((len(factorials), len(factorials)))
        for i in range(len(factorials)):
            if i == 0:
                continue
            else:
                f = factorials[:i+1]
                denom[i,:i+1] = f + f[::-1]
        self.combs = nfactorials - denom
        self.combs[np.triu_indices(self.combs.shape[0], k=1)] = -np.inf 

        avg_rel = float(G.shape[0])/Q.shape[0]      
        self.P_L = norm.logpdf(np.arange(self.max_L+1), loc=avg_rel)

        self.binomial = lambda n,k,P: self.combs[n,k]+k*P+(n-k)*np.log(1-np.exp(P))
        P_Kglctq = lambda K,l,csel: binomial(self.N-l, K-l, self.P_GJct(csel))
        P_lgLctq = lambda L,l,csel,cq: binomial(L,l, self.P_J(csel,cq))
        inner_sum = lambda L,K,l, csel, cq: inner_sum_mt.setdefault((L,K,l), 
                                            np.logaddexp(P_Kglctq(K,l,csel), P_lgLctq(L,l,csel,cq)))

        # self.PCorrgCqCsel = lambda cq, csel: logsumexp([[inner_sum(L,K,l,csel,cq)-np.log(K) 
        #                                               for l in range(min(L,K))] for (L,K) in zip(range(self.N),range(self.N))])
        self.PCorrgCtqCsel = lambda (ctq, csel) : logsumexp(self.P_QHcgct(self.VC, ctq)+
                                                                self.PCorrgCqCsel(self.VC, csel))

        # self.hatC = lambda ctq: self.VC[np.argmax([self.PCorrgCtqCsel(ctq,csel) for csel in self.VC])]
        self.P_C = lambda ctq: executor.map(self.PCorrgCtqCsel, [(ctq,csel) for csel in self.VC])       

    def reset_super(self, VC, G, Q):
        super(OneofLExperiment, self).reset(VC, G, Q)

    def getBinomialProbInRange(self, N, k_range, P):
        coeffs = self.combs[N,:k_range]
        ks = np.arange(k_range)
        return coeffs + ks*P + ks[::-1]*np.log(1-np.exp(P))

    def getAllBinomialProbs(self, P, invert_ks=False):
        if len(P.shape) == 0:
            P = np.expand_dims(P, 0)
        
        # print 1
        t0 = time.time()
        prob_terms = np.empty(self.combs.shape+(P.shape[0],), dtype='float32')
        prob_terms -= np.inf        
        prob_terms[0,0] = 0
        # print 2, time.time()-t0
        t0 = time.time()
        ks = np.arange(self.combs.shape[1])
        mP = np.log(1-np.exp(P))
        combs_expanded = np.expand_dims(self.combs,-1)      
        # print 3,time.time()-t0

        t0 = time.time()
        for i in range(self.combs.shape[0]):
            ncs = self.combs[i]
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


    def PCorrgCqCsel(self, cq, csel):
        print (csel)
        sum_terms = []
        P_GJ = self.P_GJct(csel)
        P_J = self.P_J(csel,cq)
        
        bins_P_GJ = self.getAllBinomialProbs(P_GJ).squeeze()        
        bins_P_J = self.getAllBinomialProbs(P_J)        
        rand_i = np.random.randint(bins_P_J.shape[0])       

        inner_sums = []
        for L in range(1, self.max_L+1):
            k_range = self.max_K+1
            l_range = lambda k: min(L, k)+1                     
            ps = []
            
            for k in range(1,k_range):
                # bin_L = self.getBinomialProbInRange(L, l_range(k), P_J)               
                log_k = np.log(k)
                for l in range(1,l_range(k)):
                    # p = self.binomial(self.N-l, k-l, P_GJ) + bin_L[l] - log_k
                    p = bins_P_GJ[self.N-l, k-l] + bins_P_J[L,l] - log_k
                    ps.append(p)
                    # print bins_P_GJ[self.N-l, k-l], bins_P_J[L,l][:3], log_k, p[:3]               
            inner_sums.append(logsumexp(ps, axis=0))        
        inner_sums = np.array(inner_sums)       
        sum_terms = self.P_L[1:].reshape(-1,1) + inner_sums     
        return logsumexp(sum_terms,axis=0)
    
    def hatC(self, ctq, P=None):
        # print [self.PCorrgCtqCsel(ctq,csel) for csel in self.VC]
        # return self.VC[np.argmax([self.PCorrgCtqCsel((ctq,csel)) for csel in self.VC])]
        if P is not None:
            Ps = P.map(PCorrgCtqCsel_, [(self.P_L, self.N, self.max_L, self.max_K, 
                    self.P_J(csel,self.VC), self.P_GJct(csel), 
                    self.P_QHcgct(self.VC, ctq), self.combs) for csel in self.VC_Gt])
        else:
            Ps = map(PCorrgCtqCsel_, [(self.P_L, self.N, self.max_L, self.max_K, 
                    self.P_J(csel,self.VC), self.P_GJct(csel), 
                    self.P_QHcgct(self.VC, ctq), self.combs) for csel in self.VC_Gt])
        Ps = np.array(Ps)
        return self.VC_Gt[np.argmax(Ps)]

    def test(self, g_ids):
        hatC_cache = {}
        total_correct = 0
        total_cov_correct = 0
        total = 0
        confmat = np.zeros((self.Q.shape[0], self.Q.shape[0]))
        for i in range(len(self.Qt)):
            t0 = time.time()            
            cq = self.Q[i]
            ctq = self.Qt[i]
            csel = hatC_cache.setdefault(tuple(ctq), self.hatC(ctq))    
            # print torch.min(self.Gt == csel, dim=1)

            shortlist = np.arange(0,self.Gt.shape[0])[np.all(self.Gt == csel, axis=1)]
            choice = -1
            correct = 0
            cov_correct = 0
            if len(shortlist) > 0:
                choice = shortlist[np.random.randint(0,shortlist.shape[0])]
                # choice = self.G[choice]
                # correct = int(torch.equal(choice,cq))
                choice_id = g_ids[choice]
                correct = int(i == choice_id)
                cov_correct = int(np.allclose(self.G[choice], cq))
            total_correct += correct
            total_cov_correct += cov_correct
            total += 1
            print (i, cq, ctq, csel, choice, ids[i], int(choice_id), float(total_correct)/total, float(total_cov_correct)/total, time.time()-t0)
            print (self.VC_G)
            # confentry = confmat.setdefault(str(list(cq.numpy())), {})
            # confentry[str(list(choice.numpy()))] = confentry.get(str(list(choice.numpy())),0)+1       
            if choice >= 0:
                confmat[i, choice_id] += 1      
        print(float(total_correct)/total)
        np.save(confmat, 'confmat.npy')

class RankingExperiment(OneofLExperiment):
    """docstring for RankingExperiment"""
    def __init__(self, G, Q, max_L=None, true_prob=0.9):
        super(RankingExperiment, self).__init__(G, Q, max_L, true_prob=true_prob)
    
    @staticmethod
    def average_precision(ranking, total_hits):     
        ranks = np.arange(1,len(ranking)+1).astype('float32')
        return np.sum((np.cumsum(ranking)/ranks)[ranking == 1])/float(total_hits)

    def test(self, g_ids, verbose=False, naive=False, pool=None):
        hatC_cache = {}
        total_correct = 0
        total_cov_correct = 0
        total = 0
        confmat = np.zeros((self.Q.shape[0], self.Q.shape[0]))

        zero_hits = 0
        full_G = self.G.copy()
        full_g_ids = g_ids.copy()

        APs = []
        RRs = []
        for i in range(len(self.Qt)):                   
            cq = self.Q[i]
            ctq = self.Qt[i]

            if verbose:
                print ("running query for ", ctq, cq)

            total_hits = len(g_ids[g_ids == i])         
            if total_hits == 0:
                zero_hits += 1
                continue

            if verbose:
                print ('total_hits =', total_hits)

            ranking_labels = []
            ranking_ids = []
            total_correct = 0.0
            ap = 0
            rr = -1
            for r in range(self.max_L):             
                if naive:
                    csel = ctq
                else:
                    csel = self.hatC(ctq, pool)
                np.random.seed(0)
                shortlist = np.arange(0,self.Gt.shape[0])[np.all(self.Gt == csel, axis=1)]
                choice = -1
                correct = 0
                cov_correct = 0
                if len(shortlist) > 0:
                    choice = shortlist[np.random.randint(0,shortlist.shape[0])]
                    choice_id = g_ids[choice]
                    correct = int(i == choice_id)
                    total_correct += correct
                    if correct:
                        ap +=( total_correct/(r+1))/total_hits
                        if rr == -1:
                            rr = 1.0/(r+1)
                    ranking_labels.append(correct)
                    ranking_ids.append(choice_id)

                    g_ids = np.delete(g_ids, choice, 0)
                    n_left = g_ids[g_ids == i].shape[0]
                    
                if verbose:                             
                    print (r, csel, self.G[choice], choice_id, i, ap, n_left, g_ids.shape) #self.P_GJct(csel), self.Gt[np.all(self.Gt == ctq, axis=1)].shape[0]
                if n_left == 0:
                        break
                if choice != -1:
                    self.updateGallery([choice])
            APs.append(ap)
            RRs.append(rr)
            
            self.reset_super(self.VC, full_G, self.Q)
            g_ids = full_g_ids      
            np.savetxt('APs.txt', APs)
            np.savetxt('RRs.txt', RRs)
        print ("MAP =",np.mean(APs), 'MRR =', np.mean(RRs))

if __name__ == '__main__':
    # for NOISY_PROB in [0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.0]:
    #   print NOISY_PROB, runRankingTest(True, true_prob=NOISY_PROB)

    # P = Pool(11)
    # print P.map(runIDTest, [(True,p) for p in [1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.0]])
    id_map, _ = loadVCMeta('../data/vox1_meta.csv')
    ids, gdata = loadVCIDdata('../data/iden_split.txt', id_map)

    # VC, id_map, gdata, ids = buildToyDataset([(0,1),(0,5)], 3, 5)
    # # gdata = gdata[:1000]
    # # ids = ids[:1000]
    Q, G = id_map, gdata

    # labels, qset = loadVCVeriData('veri_test.txt', id_map)
    # # Q = qset[:,[0]].squeeze()
    # # G = qset[:,[1]].squeeze()
    # # print Q.shape, G.shape
    # # max_L = np.max(np.bincount(ids[:1000]))
    # # print max_L

    # e = RankingExperiment(G, Q)
    # # e = OneofLExperiment(G, Q, max_L=20)
    # e = UniqueMatchExperiment(Q,G)
    # id_map = np.array([[1],[0]])
    # e = VerificationExperiment(id_map, id_map, id_map, true_prob=1.0)
    # # t0 = time.time()
    # print e.PCorrgCtqCsel((np.array([[1,5]]),np.array([[1,5]])))
    # print e.hatC(np.array([[1,2]]))
    # # print time.time() - t0

    # e.test(ids)
    # e.test(qset, labels=labels)
    # # e.test(qset=zip(range(Q.shape[0]), range(G.shape[0])), labels=labels)