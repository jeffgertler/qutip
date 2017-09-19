import qutip as qt
import matplotlib.pyplot as pl
import matplotlib as mpl
import numpy as np
import datetime
import os
import itertools
import profile


''' Call must set two of the arrays to the vec desired and leave other two '''
def wigner4d(rho, xvec, cut_dim):
    if(rho.type == 'ket'):
        rho = qt.ket2dm(rho)
    n = len(xvec)
    N = rho.dims[0][0]
    W = np.empty((n, n), dtype = float)
    a = qt.tensor(qt.destroy(N), qt.qeye(N))
    b = qt.tensor(qt.qeye(N), qt.destroy(N))
    PJ = (1j * np.pi * a.dag() * a).expm() * (1j * np.pi * b.dag() * b).expm()
    if(cut_dim == 'im'):
        xvec = xvec.astype(complex) * 1j
        
    for i in range(n):
        DA = qt.tensor(qt.displace(N, xvec[i]), qt.qeye(N))
        for j in range(n):
            DB = qt.tensor(qt.qeye(N), qt.displace(N, xvec[j]))
            W[i,j] = np.real((rho * DA * DB * PJ * DB.dag() * DA.dag()).tr())

    return W


def winger2d(rho, xvec, cavity):
    if(rho.type == 'ket'):
        rho = qt.ket2dm(rho)
    n = len(xvec)    
    N = rho.dims[0][0]
    W = np.empty((n, n), dtype = float)
    a = qt.tensor(qt.destroy(N), qt.qeye(N))
    b = qt.tensor(qt.qeye(N), qt.destroy(N))
    P = (1j * np.pi * a.dag() * a).expm() * (1j * np.pi * b.dag() * b).expm()
    if(cavity == 'a'):
        for i,j in itertools.product(range(n), range(n)):
            D = qt.tensor(qt.displace(N, xvec[i] + xvec[j]*1j), qt.qeye(N))
            W[i,j] = np.real((rho * D * P * D.dag()).tr())
    elif(cavity == 'b'):  
        for i,j in itertools.product(range(n), range(n)):
            D = qt.tensor(qt.qeye(N), qt.displace(N, xvec[i] + xvec[j]*1j))
            W[i,j] = np.real((rho * D * P * D.dag()).tr())
    
    return W

NA = 20
NB = 20
alpha = 1.8
drive = 1.5
loss = 1.2


num_steps = 1
max_time = 10


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


psi0 = (qt.tensor(qt.coherent(NA, alpha), qt.coherent(NB, alpha)) - 
        qt.tensor(qt.coherent(NA, -alpha), qt.coherent(NB, -alpha))).unit()


#psi0 = qt.tensor(qt.fock(NA, 0), qt.fock(NB, 0))

''' Solve the system '''
times = np.linspace(0.0, max_time, num_steps)
opts = qt.Options(store_states=True, nsteps=10000)
print(opts)
print('solving...')
result = qt.mesolve(H, psi0, times, [loss * (a+b)**2], 
                    [a.dag() * a, b.dag() * b], 
                    options=opts, progress_bar = True)
print('solved!')




"""
num_points = 25
xvec = np.linspace(-2, 2, num_points)

print('plotting single cavity traces')
fig = pl.figure(figsize=(5*num_steps, 10))
for i in range(num_steps):
    pl.subplot(2, num_steps, i+1)
    W = qt.wigner(result.states[i].ptrace(0), xvec, xvec) * np.pi
    pl.contourf(xvec, xvec, W, np.linspace(-1.0, 1.0, 41, endpoint=True), cmap=mpl.cm.RdBu_r)
    pl.colorbar(ticks = np.linspace(-1.0, 1.0, 11, endpoint=True))
    pl.title('Cavity A t=' + str(i*max_time/num_steps))
    
    pl.subplot(2, num_steps, i+1+num_steps)
    W = qt.wigner(result.states[i].ptrace(1), xvec, xvec) * np.pi
    pl.contourf(xvec, xvec, W, np.linspace(-1.0, 1.0, 41, endpoint=True), cmap=mpl.cm.RdBu_r)
    pl.colorbar(ticks = np.linspace(-1.0, 1.0, 11, endpoint=True))
    pl.title('Cavity B t=' + str(i*max_time/num_steps))

pl.savefig(filepath + 'single_traces.png')
pl.clf()
"""



num_points = 25
xvec = np.linspace(-2, 2, num_points)

if 0:
    points = [np.zeros((num_points, num_points)), np.meshgrid(xvec, xvec)[0],  
                  np.zeros((num_points, num_points)), np.meshgrid(xvec, xvec)[1]]
    profile.run('wigner4d(result.states[0], points); print()')

print('after dis_lib, before single' + str(datetime.datetime.now().time()))



fig = pl.figure(figsize=(5*num_steps, 20))
for i in range(num_steps):
    print(str(i) + '/' + str(num_steps))
    ''' reA vs imA '''
    pl.subplot(4, num_steps, i+1)
    W = winger2d(result.states[i], xvec, 'a')
    pl.contourf(xvec, xvec, W, np.linspace(-1.0, 1.0, 41, endpoint=True), cmap=mpl.cm.RdBu_r)
    pl.title(' reA vs imA t=' + str(i*max_time/num_steps))
    pl.colorbar(ticks = np.linspace(-1.0, 1.0, 11, endpoint=True))

    ''' reB vs imB '''
    pl.subplot(4, num_steps, i+1 + num_steps)
    W = winger2d(result.states[i], xvec, 'b')
    pl.contourf(xvec, xvec, W, np.linspace(-1.0, 1.0, 41, endpoint=True), cmap=mpl.cm.RdBu_r)
    pl.title('reB vs imB t=' + str(i*max_time/num_steps))
    pl.colorbar(ticks = np.linspace(-1.0, 1.0, 11, endpoint=True))

    ''' reA vs reB '''
    pl.subplot(4, num_steps, i+1 + 2*num_steps)
    W = wigner4d(result.states[i], xvec, 're')
    pl.contourf(xvec, xvec, W, np.linspace(-1.0, 1.0, 41, endpoint=True), cmap=mpl.cm.RdBu_r)
    pl.title('reA vs reB t=' + str(i*max_time/num_steps))
    pl.colorbar(ticks = np.linspace(-1.0, 1.0, 11, endpoint=True))

    ''' imA vs imB '''
    pl.subplot(4, num_steps, i+1 + 3*num_steps)
    W = wigner4d(result.states[i], xvec, 'im')
    pl.contourf(xvec, xvec, W, np.linspace(-1.0, 1.0, 41, endpoint=True), cmap=mpl.cm.RdBu_r)
    pl.title('imA vs imB t=' + str(i*max_time/num_steps))
    pl.colorbar(ticks = np.linspace(-1.0, 1.0, 11, endpoint=True))

print('after mixed' + str(datetime.datetime.now().time()))

pl.savefig(filepath + '4d_wigner_cuts.png')
pl.clf()


