"""

Natural resource model
John Stachurski, Jan 2013

"""

from __future__ import division
import numpy as np

class BM:

    def __init__(self):
        self.theta, self.alpha, self.rho = 0.5, 0.5, 0.98      
        self.a, self.b, self.c, self.d = 1, 2, 20, 1
        self.s = 0.2
        self.m = - self.s/2 
        self.shock_mean = np.exp(self.m + self.s/2)

    def U(self, x): 
        # return c**0.5  
        return 1 - np.exp(- self.theta * x**0.9)  

    def f(self, k, z): 
        " Cobb-Douglas multiplied by generalised logistic "
        return (k**self.alpha) * \
               (self.a + (self.b-self.a) / (1 + np.exp(-self.c * (k-self.d)))) * z

    def shocks(self, n):
        return np.random.lognormal(mean=self.m, sigma=self.s, size=n)

