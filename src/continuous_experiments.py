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

from scipy import integrate
from scipy.special import logsumexp
from scipy.stats import expon, multinomial, multivariate_normal, norm
from scipy.optimize import minimize

from concurrent.futures import ThreadPoolExecutor, wait
from multiprocessing.pool import Pool

from data_processing import *
from utils import *
from distributions import *
from channels import *
import logging

def getpct(pc, ct, sigma, lower, upper):
    return pc * integrate.quad(lambda c: mTruncnorm.pdf(ct, lower, upper, c, sigma), lower, upper)[0]

def getpcgct(pc, c, ct, sigma, lower, upper):
    num = pc * mTruncnorm.pdf(ct, lower, upper, c, sigma)
    den = getpct(pc, ct, sigma, lower, upper)
    return num/float(den)

class ContinuousExperiment(object):
    """docstring for Experiment"""  

    def __init__(self, rangeVC, G=None, Q=None):
        super(ContinuousExperiment, self).__init__()
        self.reset(rangeVC, G, Q)

    def reset(self, rangeVC, G, Q):
        self.G = G       
        self.Q = Q      

        self.rangeVC = rangeVC

        
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
        self.P_H = self.H
        # self.J = IdentityNoisyChannel()
        self.J = TruncatedGaussianNoisyChannel(np.zeros(G.shape[1]), 
                                                np.zeros(G.shape[1])+self.std_J,
                                                [l for l,_ in rangeVC],
                                                [u for _,u in rangeVC])
        # self.J = RandomNoisyChannel(true_prob, self.VC)
        # self.J = IndependentRandomNoisyChannel(np.array([true_prob]*self.VC.shape[1]), self.VC)
        self.P_J = self.J


        self.Qt = self.H(Q)
        self.Gt = self.J(G)

        

        PQ_Q = [1.0/(u-l) for l,u in rangeVC]
        self.PQ_Q = [1.0/(u-l) for l,u in rangeVC]
        self.P_QHct = lambda cti, i: getpct(PQ_Q[i], cti, self.std_H, rangeVC[i][0], rangeVC[i][1])
        self.P_QHcgct = lambda ci, cti, i: getpcgct(PQ_Q[i], ci, cti, self.std_H, rangeVC[i][0], rangeVC[i][1])


        PG_G = [1.0/(u-l) for l,u in rangeVC]
        self.PG_G = [1.0/(u-l) for l,u in rangeVC]
        self.P_GJct = lambda cti, i: getpct(PG_G[i], cti, self.std_J, rangeVC[i][0], rangeVC[i][1])
        self.P_GJcgct = lambda ci, cti, i: getpcgct(PG_G[i], ci, cti, self.std_J, rangeVC[i][0], rangeVC[i][1])        

        self.Q = Q
        self.G = G

class ContinuousClassificationExperiment(ContinuousExperiment):
    """docstring for ClassificationExperiment"""
    def __init__(self, rangeVC, G, Q):
        super(ContinuousClassificationExperiment, self).__init__(rangeVC, G, Q)
        self.N = float(len(G))
        self.logger = logging.getLogger('ContinuousClassificationExperiment')

    def PCorrgCqCsel(self, ci, cseli, i):
        pj = self.P_J.pdf_i(cseli, ci, i)
        pgj = self.P_GJct(cseli, i)

        return pj*(1-(1-pgj)**self.N)/(self.N*pgj)

    def PCorrgCqCsel_logged(self, ci, cseli, i):
        pj = self.P_J.pdf_i(cseli, ci, i)
        pgj = min(self.P_GJct(cseli, i), 1)

        return (pj - np.exp(np.log(pj) + (self.N * np.log(1 - pgj)))) / (self.N * pgj)

        # return pj*(1-(1-pgj)**self.N)/(self.N*pgj)

    def PCorrgCtqCsel(self, ct, csel):        
        
        g = lambda cseli, cti, i: (
            integrate.quad(
            (lambda ci: self.P_QHcgct(ci, cti, i) * self.PCorrgCqCsel(ci, 
                                                        cseli, i)), 
            self.rangeVC[i][0], 
            self.rangeVC[i][1])[0]
        )

        # g = lambda cseli, cti, i: (
        #     integrate.quad(
        #     (lambda ci: self.P_QHcgct(ci, cti, i) * self.PCorrgCqCsel_logged(ci, 
        #                                                 cseli, i)), 
        #     self.rangeVC[i][0], 
        #     self.rangeVC[i][1])[0]
        # )

        return np.prod([g(cseli, cti, i) for i,(cseli, cti) in enumerate(zip(csel, ct))])
    
    def genInitialVec(self):
        L, U = zip(*self.rangeVC)
        return np.random.uniform(L, U)

    def getAnswer(self, chat, Gt):
        D = np.sum(np.square(chat - Gt), axis=1)
        return np.argmin(D)

    def test(self, labels, verbose=False, naive=True):
        Qt = np.array(self.H.addNoise(self.Q))
        Gt = np.array(self.J.addNoise(self.G))

        correct = 0        
        for i,cpt in enumerate(self.Qt):
            if naive:
                inferred = cpt
            else:
                obj = lambda csel: -self.PCorrgCtqCsel(cpt, csel)
                init_csel = self.genInitialVec()
                solverLBFGSB = 'L-BFGS-B'
                solverTNC = 'TNC'
                solverSLSQP = 'SLSQP'
                res = minimize(obj, init_csel, bounds=self.rangeVC, method=solverSLSQP)
                print(res)
                inferred = res.x

            csel = self.getAnswer(inferred, Gt)
            correct += int(labels[i] == csel)

            # print('accuracy: %0.4f, cp: %d, cpt:%d, csel:%d, cg:%d' % (correct/float(i+1),
            #                                                             self.Q[i],
            #                                                             Qt[i],
            #                                                             csel,
            #                                                             self.G[i]))

            self.logger.debug('accuracy: %s, cp: %s, cpt:%s, inferred:%s, cg:%s, cgt:%s', correct/float(i+1),
                                                                        self.Q[i],
                                                                        Qt[i],
                                                                        inferred,
                                                                        Gt[csel],
                                                                        self.G[csel])

def r(cpti, cgti, lambda_i):
    return lambda_i * np.exp(-lambda_i * abs(cpti - cgti))

class ContinuousVerificationExperiment(ContinuousExperiment):
    """docstring for ClassificationExperiment"""
    def __init__(self, rangeVC, G, Q):
        super(ContinuousVerificationExperiment, self).__init__(rangeVC, G, Q)
        self.N = float(len(G))
        self.logger = logging.getLogger('ContinuousClassificationExperiment')
    
    def genInitialVec(self):
        return np.random.random(size=3)

    def Pcptcgtgmatch(self, cpti, cgti, low, high, i):
        return integrate.quad(lambda cpi: self.P_H.pdf_i(cpti, cpi, i) * self.P_J.pdf_i(cgti, cpi, i) * self.PQ_Q[i] , low, high)[0]


    def Pcptcgtgmismatch(self, cpti, cgti, low, high, i):
        integral_cp = integrate.quad(lambda cpi: self.P_H.pdf_i(cpti, cpi, i) * self.PQ_Q[i], low, high)[0]

        integral_cg = integrate.quad(lambda cgi: self.P_J.pdf_i(cgti, cgi, i) * self.PG_G[i], low, high)[0]
        # print(integral_cp * integral_cg)
        return integral_cp * integral_cg

    def FAi(self, lambda_i, i, low, high):
        return integrate.quad(lambda cpti: integrate.quad(lambda cgti: r(cpti, cgti, lambda_i) * self.Pcptcgtgmismatch(cpti, cgti, low, high, i), low, high)[0], low, high)[0]

    def FA(self, lambda_vector):
        res = 1
        for i in range(len(lambda_vector)):
            res *= self.FAi(lambda_vector[i], i, self.rangeVC[i][0], self.rangeVC[i][1])
        
        return res

    def FRi(self, lambda_i, i, low, high):
        return integrate.quad(lambda cpti: integrate.quad(lambda cgti: (1 - r(cpti, cgti, lambda_i)) * self.Pcptcgtgmatch(cpti, cgti, low, high, i), low, high)[0], low, high)[0]

    def FR(self, lambda_vector):
        res = 1
        for i in range(len(lambda_vector)):
            res *= self.FRi(lambda_vector[i], i, self.rangeVC[i][0], self.rangeVC[i][1])
        
        return res

    #size of lambda vector is no. of covariates + 1
    def objective_function(self, lambda_vector):
        return (self.FA(lambda_vector[:-1]) * (1 - lambda_vector[-1])) + (lambda_vector[-1] * self.FR(lambda_vector[:-1]))

    def learn_lambdas(self):
        solverSLSQP = 'SLSQP'
        res = minimize(self.objective_function, self.genInitialVec(), method=solverSLSQP, tol=1e-2)
        print(res)
        lambdas = res.x[:-1]
        return lambdas

    def test(self, qset=None, ids=None, labels=None, verbose=True, naive=False):
        assert(len(qset) == len(labels))

        lambdas = self.learn_lambdas()
        print(lambdas)

        # lambdas = np.array([0.37236356, 0.29762754])

        correct_pos = 0.0
        correct_neg = 0.0
        total_pos = 0.0
        total_neg = 0.0
        i = 0
        for i, (i_ctq, i_ctg) in enumerate(qset):       
            p = [r(self.Qt[i_ctq, j], self.Gt[i_ctg, j], lambdas[j]) for j in range(self.Gt.shape[1])]
            p = np.prod(p)
            flip = np.random.binomial(1, p)
            # flip = int(np.all(self.Qt[i_ctq] == self.Gt[i_ctg]))
            if labels[i]:
                if naive:
                    correct_pos += int(np.allclose(self.Qt[i_ctq], self.Gt[i_ctg]))
                else:
                    correct_pos += int(flip == labels[i])
                total_pos += 1  
            else:
                if naive:
                    correct_neg += int(not np.allclose(self.Qt[i_ctq], self.Gt[i_ctg]))
                else:
                    correct_neg += int(flip == labels[i])
                total_neg += 1
            if verbose and i > 0 and min(total_pos, total_neg) > 0 and i % 1000 == 0:               
                # print (i, self.Q[i_ctq], self.Qt[i_ctq], self.G[i_ctg], self.Gt[i_ctg], self.r[i_ctq_vc, i_ctg_vc], flip, labels[i])
                print (correct_pos/total_pos,correct_neg/total_neg, (correct_pos+correct_neg)/(total_pos+total_neg)) 
        print ((correct_pos/total_pos),correct_neg/total_neg, (correct_pos+correct_neg)/(total_pos+total_neg))
        return (correct_pos+correct_neg)/(total_pos+total_neg)


def check_probability_pct(low, high, sigma):
    prior = 1.0/(high - low)
    print(integrate.quad(lambda ct: getpct(prior, ct, sigma, low, high), low, high))

def check_probability_pcgct(low, high, sigma):
    pc= 1.0/(high - low)
    ct = 2
    print(integrate.quad(lambda c: getpcgct(pc, c, ct, sigma, low, high), low, high))
    print(integrate.quad(lambda ct: mTruncnorm.pdf(ct, low, high, 2, sigma), low, high))

def main():
    check_probability_pct(1, 5, 1)
    check_probability_pcgct(1, 5, 1)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG,
    format="%(name)s: %(message)s")
    main()