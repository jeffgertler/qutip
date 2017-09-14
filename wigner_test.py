import qutip as qt
import matplotlib.pyplot as pl
import matplotlib as mpl
import numpy as np
import datetime
import os
import itertools

''' Call must set two of the arrays to the vec desired and leave other two '''
def wigner4d(rho, points):
    n = max(len(points[0]), len(points[1]), len(points[2]), len(points[3])) 
    N = rho.dims[0][0]
    W = np.empty((n, n), dtype = complex)
    a = qt.tensor(qt.destroy(N), qt.qeye(N))
    b = qt.tensor(qt.qeye(N), qt.destroy(N))
    PJ = (1j * np.pi * a.dag() * a).expm() * (1j * np.pi * b.dag() * b).expm()
    for i,j in itertools.product(range(n), range(n)):
        #if (i*n+j)%n == 0: print(str(i) + '/' + str(n))
        A = points[0][i,j] + points[1][i,j] * 1j
        B = points[2][i,j] + points[3][i,j] * 1j

        DA = qt.tensor(qt.displace(N, A), qt.qeye(N))
        DB = qt.tensor(qt.qeye(N), qt.displace(N, B))
        W[i,j] = (rho * DA * DB * PJ * DB.dag() * DA.dag()).tr()
    return W
                    
    

NA = 20
NB = 20
alpha = 1.9
drive = 5

num_steps = 4
max_time = 4


date = list(str(datetime.datetime.now())[:19])
date[13] = '-'
date[16] = '-'

''' SUPER IMPORTANT: change the filepath to wherever you want the plots saved '''
filepath = 'C:/Users/Wang Lab/Documents/qutip/out/wigner_test/' + ''.join(date) + '/'

if not os.path.exists(filepath):
    os.makedirs(filepath)

a = qt.tensor(qt.destroy(NA), qt.qeye(NB))
b = qt.tensor(qt.qeye(NB), qt.destroy(NA))

PJ = (1j * np.pi * a.dag() * a).expm() * (1j * np.pi * b.dag() * b).expm()

H = drive * a * b + drive.conjugate() * a.dag() * b.dag()
'''
psi0 = (qt.tensor(qt.coherent(NA, alpha), qt.coherent(NB, alpha)) - 
        qt.tensor(qt.coherent(NA, -alpha), qt.coherent(NB, -alpha))).unit()
'''

psi0 = qt.tensor(qt.fock(NA, 0), qt.fock(NB, 0))

''' Solve the system '''
times = np.linspace(0.0, max_time, num_steps)
opts = qt.Options(store_states=True, nsteps=10000)
print(opts)
print('solving...')
result = qt.mesolve(H, psi0, times, [], 
                    [a.dag() * a, b.dag() * b], 
                    options=opts, progress_bar = True)
print('solved!')



num_points = 100
xvec = np.linspace(-5, 5, num_points)

print('plotting reA vs imA')
''' reA vs imA '''
fig = pl.figure(figsize=(10,10))
for i in range(num_steps):
    pl.subplot(2,2,i+1)
    W = qt.wigner(result.states[i].ptrace(0), xvec, xvec) * np.pi
    pl.contourf(xvec, xvec, W, np.linspace(-1.0, 1.0, 100, endpoint=True), cmap=mpl.cm.RdBu_r)
    pl.title('t=' + str(i*max_time/num_steps))
    pl.colorbar(ticks = np.linspace(-1.0, 1.0, 11, endpoint=True))

pl.savefig(filepath + 'reA-imA.png')
pl.clf()

print('plotting reB vs imB')
''' reB vs imB '''
fig = pl.figure(figsize=(10,10))
for i in range(num_steps):
    pl.subplot(2,2,i+1)
    W = qt.wigner(result.states[i].ptrace(1), xvec, xvec) * np.pi
    pl.contourf(xvec, xvec, W, np.linspace(-1.0, 1.0, 100, endpoint=True), cmap=mpl.cm.RdBu_r)
    pl.title('t=' + str(i*max_time/num_steps))
    pl.colorbar(ticks = np.linspace(-1.0, 1.0, 11, endpoint=True))

pl.savefig(filepath + 'reB-imB.png')
pl.clf()

num_points = 40
xvec = np.linspace(-2.5, 2.5, num_points)
"""
print('plotting reA vs reB')
''' reA vs reB '''
fig = pl.figure(figsize=(10,10))
for i in range(num_steps):
    pl.subplot(2,2,i+1)
    points = [np.meshgrid(xvec, xvec)[0], np.zeros((num_points, num_points)), 
              np.meshgrid(xvec, xvec)[1], np.zeros((num_points, num_points))]
    W = wigner4d(qt.ket2dm(result.states[i]), points)
    pl.contourf(xvec, xvec, W, np.linspace(-1.0, 1.0, 100, endpoint=True), cmap=mpl.cm.RdBu_r)
    pl.title('t=' + str(i*max_time/num_steps))
    pl.colorbar(ticks = np.linspace(-1.0, 1.0, 11, endpoint=True))

pl.savefig(filepath + 'reA-reB.png')
pl.clf()

print('plotting imA vs imB')
''' imA vs imB '''
fig = pl.figure(figsize=(10,10))
for i in range(num_steps):
    pl.subplot(2,2,i+1)
    points = [np.zeros((num_points, num_points)), np.meshgrid(xvec, xvec)[0],  
              np.zeros((num_points, num_points)), np.meshgrid(xvec, xvec)[1]]
    W = wigner4d(qt.ket2dm(result.states[i]), points)
    pl.contourf(xvec, xvec, W, np.linspace(-1.0, 1.0, 100, endpoint=True), cmap=mpl.cm.RdBu_r)
    pl.title('t=' + str(i*max_time/num_steps))
    pl.colorbar(ticks = np.linspace(-1.0, 1.0, 11, endpoint=True))

pl.savefig(filepath + 'imA-imB.png')
pl.clf()

"""
