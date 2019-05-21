import numpy as np

class VectorMultinomial(object):
    """docstring for VectorMultinomial"""
    def __init__(self, data=None, probs=None):
        super(VectorMultinomial, self).__init__()
        if data is not None:
            self.data = data        
            self.probs = [np.histogram(data[:,[i]], bins=data[:,[i]].max()+1)[0].astype('float32')+1e-20 for i in range(data.shape[1])]         
        elif probs is not None:
            self.probs = probs
        self.probs = [x/np.sum(x) for x in self.probs]
    
    def sample(self):
        return [np.argmax(np.random.multinomial(1,p)) for p in self.probs]
    
    def sample_n(self, n):
        return [self.sample() for i in range(n)]

    def pdf(self,x):    
        if len(x.shape) < 2:
            x = np.expand_dims(x, 0)
        p = [np.sum([np.log(self.probs[i][y[i]] if y[i] < len(self.probs[i]) else 1e-20) for i in range(len(y))]) for y in x]
        return np.array(p)

class UniformVectorMultinomial(object):
    """docstring for UniformVectorMultinomial"""
    def __init__(self, data):       
        super(UniformVectorMultinomial, self).__init__()
        self.N = reduce(lambda x,y: x*y, [len(set(data[:,i].tolist())) for i in range(data.shape[1])])
        self.lower = np.min(data, axis=0)
        self.upper = np.max(data, axis=0)

    def pdf(self, x):
        p = np.prod((x >= self.lower) * (x <= self.upper), 1) * np.log(1.0/self.N)         
        return p

class ContinuousUniformVectorMultinomial(object):
    """docstring for ContinuousUniformVectorMultinomial"""
    def __init__(self, low, high):
        super(ContinuousUniformVectorMultinomial, self).__init__()
        self.low = low
        self.high = high
        self.diff = high - low
        self.p = 1.0/(high - low)
    def pdf(self, x):
        probs = np.zeros(x.shape)
        probs[(x < self.low) & (x > self.high)] = 0
        probs[(x >= self.low) & (x <= self.high)] = self.p
        p = np.prod(probs, axis=-1)
        return p
        

class VectorUniform(object):
    """docstring for VectorMultinomial"""
    def __init__(self, data=None, probs=None):
        super(VectorUniform, self).__init__()
        if data is not None:
            self.data = data                
            self.probs = [[1.0/(self.data[:,[i]].max()+1)]*(self.data[:,[i]].max()+1) for i in range(data.shape[1])]            
        elif probs is not None:
            self.probs = probs
    
    def sample(self):
        return [np.argmax(np.random.multinomial(1,p)) for p in self.probs]
    
    def sample_n(self, n):
        return [self.sample() for i in range(n)]

    def pdf(self,x):    
        if len(x.shape) < 2:
            x = x.unsqueeze(0)
        return [np.sum([np.log(self.probs[i][y[i]] if y[i] < len(self.probs[i]) else 1e-20) for i in range(len(y))]) for y in x]