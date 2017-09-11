import qutip as qt
import matplotlib.pyplot as pl
import matplotlib as mpl
import numpy as np
import datetime
import os

''' Declare constants'''
NA = 25
NB = 25
NR = 8

g2 = -.111 * 2 * np.pi
eD = 4
kR = np.sqrt(40)
kA = np.sqrt(.05)

num_steps = 20
max_time = 20

date = list(str(datetime.datetime.now())[:19])
date[13] = '-'
date[16] = '-'

''' SUPER IMPORTANT: change the filepath to wherever you want the plots saved '''
filepath = 'C:/Users/Wang Lab/Documents/qutip/out/test/' + ''.join(date) + '/'

''' Declare opperators and Hamiltonian '''
aS = qt.tensor(qt.destroy(NA), qt.qeye(NR))
aR = qt.tensor(qt.qeye(NA), qt.destroy(NR))

xS = (aS + aS.dag())/2
pS = 1j*(aS - aS.dag())/2

H_2 = (g2 * aS**2 * aR.dag() 
		+ g2.conjugate() * aS.dag()**2 * aR
		+ eD * aR.dag() 
		+ eD.conjugate() * aR)

H = H_2

''' initial state '''
psi0 = qt.tensor(qt.fock(NA, 0), qt.fock(NR, 0))

''' Solve the system '''
times = np.linspace(0.0, max_time, num_steps)
opts = qt.Options(store_states=True, nsteps=10000)
print(opts)
print('solving...')
result = qt.mesolve(H, psi0, times, [kR * aR, kA * aS], 
                    [aS.dag() * aS, xS, pS, xS**2, pS**2, xS**4, pS**4], 
                    options=opts, progress_bar = True)
print('solved!')

''' Wigner and mode occupation plotting'''

if not os.path.exists(filepath):
    os.makedirs(filepath)

xvec = np.linspace(-7, 7, 100)
xvec_norm = np.linspace(-np.sqrt(7), np.sqrt(7), 100)
''' Storage Wigner plots '''
fig = pl.figure(figsize=(40,10))
for i in range(num_steps):
    pl.subplot(2,10,i+1)
    W = qt.wigner(result.states[i].ptrace(0), xvec, xvec) * np.pi
    pl.contourf(xvec, xvec, W, np.linspace(-.5, 1.0, 100, endpoint=True), cmap=mpl.cm.RdBu_r)
    pl.title('t=' + str(i*max_time/num_steps))
    pl.colorbar(ticks = np.linspace(-.5, 1.0, 16, endpoint=True))

pl.savefig(filepath + 'storage_wigner.png')
pl.clf()

''' Readout Wigner plots '''
fig = pl.figure(figsize=(40,10))
for i in range(num_steps):
    pl.subplot(2,10,i+1)
    W = qt.wigner(result.states[i].ptrace(1), xvec, xvec) * np.pi
    pl.contourf(xvec, xvec, W, np.linspace(-.5, 1.0, 100, endpoint=True), cmap=mpl.cm.RdBu_r)
    pl.title('t=' + str(i*max_time/num_steps))
    pl.colorbar(ticks = np.linspace(-.5, 1.0, 16, endpoint=True))


pl.savefig(filepath + 'readout_wigner.png')
pl.clf()

''' Storage Coefficent plots '''
fig = pl.figure(figsize=(40,10))
for i in range(num_steps):
    pl.subplot(2,10,i+1)
    pl.plot(range(NA), result.states[i].ptrace(0).diag())
    pl.title('t=' + str(i*max_time/num_steps))

pl.savefig(filepath + 'storage_coef.png')
pl.clf()

''' Readout Coefficient plots '''
fig = pl.figure(figsize=(40,10))
for i in range(num_steps):
    pl.subplot(2,10,i+1)
    pl.plot(range(NR), result.states[i].ptrace(1).diag())
    pl.title('t=' + str(i*max_time/num_steps))

pl.savefig(filepath + 'readout_coef.png')
pl.clf()




''' Expectation value plotting '''
pl.figure(figsize=(10, 8))
pl.subplot(221)
pl.plot(times, result.expect[0])
pl.xlabel('time (us)')
pl.ylabel('<n>')

pl.subplot(222)
pl.plot(times, result.expect[1], 'r')
pl.plot(times, result.expect[2], 'b')
pl.xlabel('time (us)')
pl.ylabel('<x> and <p>')

pl.subplot(223)
pl.plot(times, result.expect[3] - result.expect[1]**2, 'r')
pl.plot(times, result.expect[4] - result.expect[2]**2, 'b')
pl.xlabel('time (us)')
pl.ylabel('var(x) and var(p)')

pl.subplot(224)
pl.plot(times, result.expect[5] - 3 * result.expect[3]**2, 'r')
pl.plot(times, result.expect[6] - 3 * result.expect[4]**2, 'b')
pl.xlabel('time (us)')
pl.ylabel('4th order cumulents')

pl.savefig(filepath + 'expectation.png')
pl.clf()



''' Saving textfile with information about the run '''
np.savetxt(filepath + 'header.txt', [0], header = 
         'NA = ' + str(NA) + 
         ', NR = ' + str(NR) +  
         ', g2 = ' + str(g2) + 
         ', eD = ' + str(eD) + 
         ', kR = ' + str(kR) +  
         ', kA = ' + str(kA) +  
         ', num_steps = ' + str(num_steps) + 
         ', max_time = ' + str(max_time))
    

