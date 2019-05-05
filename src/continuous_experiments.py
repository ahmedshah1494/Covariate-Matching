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
from scipy.optimize import minimize

from concurrent.futures import ThreadPoolExecutor, wait
from multiprocessing.pool import Pool

from data_processing import *
from utils import *
from distributions import *
from channels import *

def getpct(pc, ct, sigma, lower, upper):
    return pc * integrate.quad(lambda c: mTruncnorm.pdf(ct, lower, upper, c, sigma), lower, upper)[0]

def getpcgct(pc, c, ct, sigma, lower, upper):
    num = pc * mTruncnorm.pdf(ct, lower, upper, c, sigma)
    den = getpct(pc, ct, sigma, lower, upper)
    return num/float(den)

class ContinuousExperiment(object):
    """docstring for Experiment"""  

    def __init__(self, rangeVC, G=None, Q=None):
        super(Experiment, self).__init__()
        self.reset(rangeVC, G, Q)

    def reset(self, rangeVC, G, Q):
        self.G = G       
        self.Q = Q      

        
        self.PQ = ContinuousUniformVectorMultinomial(0, 1)
        self.PG = ContinuousUniformVectorMultinomial(0, 1)

        self.std_H = 1.0
        self.std_J = 1.0

        self.H = TruncatedGaussianNoisyChannel(np.zeros(Q.shape[1]), 
                                                np.zeros(Q.shape[1])+self.std_H,
                                                [l for l,_ in rangeVC],
                                                [u for _,u in rangeVC])
        # self.H = IdentityNoisyChannel()
        # self.H = RandomNoisyChannel(true_prob, self.VC)
        # self.H = IndependentRandomNoisyChannel(np.array([true_prob]*self.VC.shape[1]), self.VC)
        self.P_H = self.H.pdf
        # self.J = IdentityNoisyChannel()
        self.J = TruncatedGaussianNoisyChannel(np.zeros(G.shape[1]), 
                                                np.zeros(G.shape[1])+self.std_J,
                                                [l for l,_ in rangeVC],
                                                [u for _,u in rangeVC])
        # self.J = RandomNoisyChannel(true_prob, self.VC)
        # self.J = IndependentRandomNoisyChannel(np.array([true_prob]*self.VC.shape[1]), self.VC)
        self.P_J = self.J.pdf


        self.Qt = self.H(Q)
        self.Gt = self.J(G)

        

        PQ_Q = [1.0/(u-l) for l,u in rangeVC]
        self.P_QHct = lambda cti, i: getpct(PQ_Q[i], cti, self.std_H, rangeVC[i][0], rangeVC[i][1])
        self.P_QHcgct = lambda ci, cti, i: getpcgct(PQ_Q[i], ci, cti, self.std_H, rangeVC[i][0], rangeVC[i][1])


        PG_G = [1.0/(u-l) for l,u in rangeVC]
        self.P_GJct = lambda cti, i: getpct(PG_G[i], cti, self.std_J, rangeVC[i][0], rangeVC[i][1])
        self.P_GJcgct = lambda ci, cti, i: getpcgct(PG_G[i], ci, cti, self.std_J, rangeVC[i][0], rangeVC[i][1])        

        self.Q = Q
        self.G = G

class ClassificationExperiment(ContinuousExperiment):
    """docstring for ClassificationExperiment"""
    def __init__(self, rangeVC, G, Q):
        super(ClassificationExperiment, self).__init__(rangeVC, G, Q)
        self.N = float(len(G))

    def PCorrgCqCsel(self, ci, cseli, i):
        pj = self.P_J.pdf_i(cseli, ci, i)
        pgj = self.P_GJct(cseli, i)

        return pj*(1-(1-pgj)**self.N)/(self.N*pgj)

    def PCorrgCtqCsel(self, ct, csel):        
        g = lambda cseli, cti, i: integrate.quad(lambda ci: self.P_QHcgct(ci, 
                                                        cti, i) * self.PCorrgCqCsel(ci, 
                                                        cseli, i), 
                                                        self.rangeVC[i][0], 
                                                        self.rangeVC[i][1])[0]

        return np.prod([g(cseli, cti, i) for i,(cseli, cti) in enumerate(zip(csel, ct))])
    
    def genInitialVec(self):
        L, U = zip(*self.rangeVC)
        return np.random.uniform(L, U)

    def getAnswer(self, chat, Gt):
        D = np.sum(np.square(chat - Gt), axis=1)
        return np.argmin(D)

    def run(self, labels):
        Qt = np.array(self.H.addNoise(self.Q))
        Gt = np.array(self.J.addNoise(self.G))

        correct = 0        
        for i,cpt in enumerate(self.Qt):
            obj = lambda csel: -self.PCorrgCtqCsel(cpt, csel)
            init_csel = self.genInitialVec()
            res = minimize(obj, init_csel, bounds=self.rangeVC, method='L-BFGS-B')
            print(res)
            chat = res.x

            csel = self.getAnswer(chat, Gt)
            correct += int(labels[i] == match_idx)

            print('accuracy: %0.4f, cp: %d, cpt:%d, csel:%d, cg:%d' % (correct/float(i+1),
                                                                        Q[i],
                                                                        Qt[i],
                                                                        csel,
                                                                        G[i]))