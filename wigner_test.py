import qutip as qt
import matplotlib.pyplot as pl
import matplotlib as mpl
import numpy as np
import datetime
import os
import itertools

def wigner4d(rho, vec):
    n = len(vec)
    N = rho.dims[0][0]
    W = np.empty((n, n, n, n))
    for i,j,k,l in itertools.product(range(n), range(n), range(n), range(n)):
        reA, imA, reB, imB = vec[i], vec[j] * 1j, vec[k], vec[l] * 1j
        A = 
        W[i,j,k,l] = 4./(np.pi**2) * 
                    (qt.displace(N, reA + imA)).tr()
                    
    
    
wigner4d(0, np.array([.1,2,.7]))
    

NA = 20
NB = 20
alpha = 2

a = qt.tensor(qt.destroy(NA), qt.qeye(NB))
b = qt.tensor(qt.qeye(NB), qt.destroy(NA))

PJ = (1j * np.pi * a.dag() * a).expm() * (1j * np.pi * b.dag() * b).expm()

psi0 = (qt.tensor(qt.coherent(NA, alpha), qt.coherent(NB, alpha)) - 
        qt.tensor(qt.coherent(NA, -alpha), qt.coherent(NB, -alpha))).unit()

xvec = np.linspace(-5, 5, 100)
W = qt.wigner(psi0.ptrace(0), xvec, xvec) * np.pi
pl.contourf(xvec, xvec, W, np.linspace(-.5, 1.0, 100, endpoint=True), cmap=mpl.cm.RdBu_r)
pl.colorbar(ticks = np.linspace(-.5, 1.0, 16, endpoint=True))

pl.show()