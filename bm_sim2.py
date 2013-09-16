"""

## Author: John Stachurski, Jan 2013

Plots stationary distribution of natural resource model.  Reproduces first
figure of the paper.


"""

from __future__ import division
import numpy as np
from scipy.optimize import fminbound, brentq
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection
from lininterp import LinInterp        
import bm_model 
import npkde

#from matplotlib import rc
#rc('font',**{'family':'serif','serif':['Utopia']})
#rc('text', usetex=True)

gridmax, gridsize = 8, 120
grid = np.linspace(0, gridmax**1e-1, gridsize)**10

def maximizer(h, a, b):
    g = lambda x: - h(x)               # Negate 
    return fminbound(g, a, b)          # and minimize


def maximum(h, a, b):
    return float(h(fminbound(lambda x: -h(x), a, b)))


def bellman(w, bm):
    """
    Parameters: 
        w is a vectorized function 
        bm is an instance of BM
    Returns: 
        An instance of LinInterp.
    """
    W = bm.shocks(1000)
    vals = []
    for y in grid:
        h = lambda k: bm.U(y - k) + bm.rho * np.mean(w(bm.f(k,W)))
        vals.append(maximum(h, 0, y))
    return LinInterp(grid, vals)


def get_policy(bm, tol = 0.05):
    " Returns an instance of LinInterp representing optimal policy."
    # Compute the value function
    v = bm.U
    while 1:
        new_v = bellman(v, bm)
        err = max(np.absolute(new_v(grid) - v(grid)))
        print err
        v = new_v
        if err < tol:
            break
    # And then the optimal policy
    policy = []
    W = bm.shocks(1000)
    for y in grid:
        h = lambda k: bm.U(y-k) + bm.rho * np.mean(v(bm.f(k,W)))
        policy.append(maximizer(h, 0, y))
    return LinInterp(grid, policy)


def policy_deterministic():
    top = 5
    bm = bm_model.BM()
    bm.rho = 0.98
    sigma = get_policy(bm, tol=0.1)
    xgrid = np.linspace(0, top, 200)
    plt.plot(xgrid, xgrid, 'k--')
    plt.plot(xgrid, bm.f(xgrid, bm.shock_mean), 'g-')#, label=r'$f(y,E \xi)$')
    plt.plot(xgrid, bm.f(sigma(xgrid), bm.shock_mean), 'b-')#, label=r'$f(k(y),E \xi)$')
    plt.plot(xgrid, sigma(xgrid), 'r-')#, label=r'$k(y)$')
    plt.xlim((0, top))
    plt.ylim((0, top))
    plt.show()


def density_plot():
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    a, b = (np.log(0.2), np.log(4))
    #a, b = (0.1, 4)
    gs = 41 # grid size
    xs = np.linspace(a, b, gs)
    bm = bm_model.BM()
    num_rhos = 12
    rhos = np.linspace(0.945, 0.99, num_rhos)
    greys = np.linspace(0.3, 0.7, num_rhos)
    init, n = 5, 50000
    Y = np.empty(n)
    verts = []
    for rho in rhos:
        bm.rho = rho
        sigma = get_policy(bm, tol=0.05)
        Y[0] = init
        W = bm.shocks(n)
        for t in range(n-1):
            Y[t+1] = bm.f(sigma(Y[t]), W[t])
        print rho, np.mean(Y)
        Y = np.log(Y)
        ys = np.empty(gs - 1)
        for i in range(gs - 1):
            ys[i] = np.sum((xs[i] <= Y) * (Y < xs[i+1])) / n
        ys[0], ys[-1] = 0, 0
        verts.append(zip(xs, ys))

    poly = PolyCollection(verts, facecolors = [str(g) for g in greys])
    poly.set_alpha(0.85)
    ax.add_collection3d(poly, zs=rhos, zdir='x')
    ax.text(np.mean(rhos), a-1.4, -0.02, r'$\beta$', fontsize=16)
    ax.text(np.max(rhos)+0.016, (a+b)/2, -0.02, r'$\log(y)$', fontsize=16)
    ax.set_ylim3d(-1.5, 1.5)
    ax.set_xlim3d(rhos[0], rhos[num_rhos-1])
    ax.set_zlim3d(0, 0.15)
    ax.set_zticks((0.15,))
    plt.show()


#        plt.plot(xgrid, [kde(x) for x in xgrid], color=str(grey),
#                label=r"$\rho=%.3f$" % rho)
