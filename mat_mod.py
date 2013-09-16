"""

## Author: John Stachurski, Jan 2013

Takashi's stochastic OLG model.  This file describes the model.

Parameter restrictions:  R < theta, gamma * R < 1, rho < 1

"""

import numpy as np

class Model:

    def __init__(self):
        self.gamma = 0.2
        self.R = 1.05
        self.theta = 1.1
        self.rho = 0.9
        self.lmb = 0.9  # lambda
        self.eps_params = -3, 0.1  # mu and sigma in lognormal
        self.eta_params = 2, 18  # Shape and scale in beta

    def kappa(self, b, e):
        return self.R * (1 - b) <= self.lmb * (self.theta + self.rho * e)

    def ts(self, b_init=0.1, e_init=0.5, n=1000):
        "Inputs are initial conditions and length of time series."
        e = np.empty(n)
        e[0] = e_init
        b = np.empty(n)
        b[0] = b_init
        np.random.seed(1)
        eps = np.random.lognormal(self.eps_params[0], self.eps_params[1], n) 
        eta = np.random.beta(self.eta_params[0], self.eta_params[1], n)
        for t in range(n-1):
            kappa_t = self.kappa(b[t], e[t])
            e[t+1] = self.rho * e[t] + eps[t]
            b[t+1] = self.gamma * ((self.theta - self.R + eta[t]) * kappa_t + \
                    self.R * b[t] + e[t+1])
        w = b/self.gamma
        return e, w


