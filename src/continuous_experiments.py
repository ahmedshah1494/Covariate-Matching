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

class ContinuousExperiment(object):
    """docstring for Experiment"""  

    def __init__(self, rangeVC, G=None, Q=None):
        super(Experiment, self).__init__()
        self.reset(rangeVC, G, Q)

    def reset(self, rangeVC, G=None, Q=None):
        self.G = G       
        self.Q = Q      

        
        self.PQ = ContinuousUniformVectorMultinomial(0, 1)
        self.PG = ContinuousUniformVectorMultinomial(0, 1)

        self.H = TruncatedGaussianNoisyChannel(np.zeros(Q.shape[1]), 
                                                np.identity(Q.shape[1]),
                                                [l for l,_ in rangeVC],
                                                [u for _,u in rangeVC])
        # self.H = IdentityNoisyChannel()
        # self.H = RandomNoisyChannel(true_prob, self.VC)
        # self.H = IndependentRandomNoisyChannel(np.array([true_prob]*self.VC.shape[1]), self.VC)
        self.P_H = self.H.pdf
        # self.J = IdentityNoisyChannel()
        self.J = TruncatedGaussianNoisyChannel(np.zeros(G.shape[1]), 
                                                np.identity(G.shape[1]),
                                                [l for l,_ in rangeVC],
                                                [u for _,u in rangeVC])
        # self.J = RandomNoisyChannel(true_prob, self.VC)
        # self.J = IndependentRandomNoisyChannel(np.array([true_prob]*self.VC.shape[1]), self.VC)
        self.P_J = self.J.pdf


        self.Qt = self.H(Q)
        self.Gt = self.J(G)

        PQ_Q = self.PQ.pdf(self.VC)
        self.P_QHct = lambda ct: logsumexp(PQ_Q + self. P_H(ct,self.VC))
        self.P_QHcgct = lambda c,ct: (self.PQ.pdf(c)+self.P_H(ct,c)-self.P_QHct(ct))

        self.PG_G = self.PG.pdf(self.VC)      
        self.P_GJct = lambda ct: logsumexp(self.PG_G + self.P_J(ct,self.VC))
        self.P_GJcgct = lambda c,ct: self.PG.pdf([c])+self.P_J(ct,c)-self.P_GJct(ct)        

        self.Q = Q
        self.G = G