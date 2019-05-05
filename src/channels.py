import numpy as np
from scipy.stats import multivariate_normal, trucnorm

class TruncatedGaussianNoisyChannel(object):
    """docstring for GaussianNoisyChannel"""
    def __init__(self, mean, std, low, high):
        super(GaussianNoisyChannel, self).__init__()
        self.low = low
        self.high = high
        self.mean = mean
        self.std = std

    def addNoise(self, x):
        noisy_x = x.copy()
        if len(noisy_x.shape) == 1:
            noisy_x = [noisy_x]
        for i in range(len(noisy_x)):
            for j in range(len(noisy_x[i])):
                c = noisy_x[i][j]
                eps = trucnorm.rvs(self.low[j]-c, self.high[j]-c, loc=self.mean, scale=1)
                noisy_x[i][j] += eps
        if len(x.shape) == 1:
            return noisy_x[0]
        else:
            return noisy_x  
            
    def pdf(self, ct, c):
        probs = []
        for i, (cti, ci) in enumerate(zip(ct, c)):
            probs.append(trucnorm.pdf(cti, self.low[i], self.high[i], loc=ci))
        return probs
        

class IdentityNoisyChannel(object):
    """docstring for IdentityNoisyChannel"""
    def __init__(self):
        super(IdentityNoisyChannel, self).__init__()
    
    def addNoise(self, x):
        return x

    def __call__(self, x):
        return self.addNoise(x)

    def pdf(self, ct, c):
        nz = np.nonzero(ct - c)[0]
        p = np.ones(max(ct.shape[0], c.shape[0]))
        p[nz] = 0
        # print ct - c, nz, p
        return np.log(p)
        
class RandomNoisyChannel(object):
    """docstring for RandomNoisyChannel"""
    def __init__(self, true_prob, VC):
        super(RandomNoisyChannel, self).__init__()
        self.true_prob = true_prob
        self.noisy_prob = (1 - self.true_prob)/(len(VC)-1)
        self.VC = VC

    def addNoise(self,x):
        noisy = np.zeros(x.shape)
        flips = [np.random.binomial(1, self.true_prob) for i in range(x.shape[0])]
        for i in range(len(flips)):
            if flips[i]:
                noisy[i] = x[i]
            else:               
                j = np.random.randint(self.VC.shape[0])
                while np.allclose(self.VC[j], x[i]):
                    j = np.random.randint(self.VC.shape[0])
                noisy[i] = self.VC[j]
        return noisy

    def __call__(self, x):
        return self.addNoise(x)

    def pdf(self, ct, c):
        if len(ct.shape) == 1 and len(c.shape) == 1:
            if np.min(ct == c):
                return np.log(self.true_prob)
            else:
                return np.log(self.noisy_prob)
        else:
            nz = np.nonzero(ct - c)[0]
            p = np.zeros(max(ct.shape[0], c.shape[0])) + self.true_prob
            p[nz] = self.noisy_prob
            # print ct - c, nz, p
            return np.log(p)

class IndependentRandomNoisyChannel(object):
    """docstring for RandomNoisyChannel"""
    def __init__(self, true_prob, VC):
        super(IndependentRandomNoisyChannel, self).__init__()
        self.true_prob = true_prob
        self.noisy_prob = np.array([(1 - self.true_prob[i])/(len(set(VC[:,i]))-1) for i in range(VC.shape[1])])
        self.VC = VC

    def addNoise(self,x):
        noisy = np.zeros(x.shape)       
        for i in range(len(x)):         
            for j in range(x.shape[1]):
                flip = np.random.binomial(1, self.true_prob[j])
                if flip:
                    noisy[i,j] = x[i,j]
                else:               
                    k = np.random.randint(self.VC.shape[0])
                    while np.allclose(self.VC[k,j], x[i,j]):
                        k = np.random.randint(self.VC.shape[0])
                    noisy[i,j] = self.VC[k,j]
        return noisy

    def __call__(self, x):
        return self.addNoise(x)

    def pdf(self, ct, c):
        prob = 0
        if len(ct.shape) == 1 and len(c.shape) == 1:
            p = np.sum(np.log(self.true_prob[ct == c])) + np.sum(np.log(self.noisy_prob[ct != c]))
            return p
        else:
            nz = (ct == c).astype('int32')
            p = (nz * self.true_prob) + ((1-nz) * (self.noisy_prob))
            p = np.sum(np.log(p), axis=1)
            return p