
"""

## Author: John Stachurski, Jan 2013

Simulation of Takashi's OLG model to generate stationary distribution.   This
simulation is for the marginal distribution of wealth under different
parameters.  Generates the second plot in the paper.

Parameter restrictions:  R < theta, gamma * R < 1, rho < 1


"""

import numpy as np
from matplotlib import pyplot as plt
import npkde
import mat_mod

#from matplotlib import rc
#rc('font',**{'family':'serif','serif':['Utopia']})
#rc('text', usetex=True)

aa, bb = 0.5, 1.4
ts_length = 100000
xgrid = np.linspace(aa, bb, 200)
m = mat_mod.Model()
m.eta_params = (3,10)
lmb_vals = 0.57, 0.58
plot_types = ['k-', 'k--']
for lmb in lmb_vals:
    m.lmb = lmb
    e, b = m.ts(n=ts_length)
    print np.mean(m.kappa(b, e) == 0)
    #plt.hist(b, normed=1, range=(aa, bb), alpha=0.2, bins=100)
    kde = npkde.NPKDE(X=b)
    plt.plot(xgrid, [kde(x) for x in xgrid], plot_types.pop(), 
            label=r"$\lambda=%.2f$" % m.lmb)
plt.legend()

plt.show()
