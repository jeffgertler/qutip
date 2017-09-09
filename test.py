import qutip as qt
import matplotlib.pyplot as pl
import matplotlib as mpl
import numpy as np
import datetime
import os

NS = 35
NR = 8

aS = qt.tensor(qt.destroy(NS), qt.qeye(NR))
aR = qt.tensor(qt.qeye(NS), qt.destroy(NR))

xS = (aS + aS.dag())/2
pS = 1j*(aS - aS.dag())/2


xSS = .004
xRR = 2.14
xSR = .206

'''

xSS = 0
xRR = 0
xSR = 0
'''

g2 = -.111 * 2 * np.pi
eD = 3
kR = np.sqrt(40)
kS = np.sqrt(.05)


H_kerr = (- xSS/2 * aS.dag()**2 * aS**2 
		- xRR/2 * aR.dag()**2 * aR**2
		- xSR * aS.dag() * aS * aR.dag() * aR)


H_2 = (g2 * aS**2 * aR.dag() 
		+ g2.conjugate() * aS.dag()**2 * aR
		+ eD * aR.dag() 
		+ eD.conjugate() * aR)

H = H_2 + H_kerr

psi0 = qt.tensor(qt.fock(NS, 0), qt.fock(NR, 0))

num_steps = 20
max_time = 20
times = np.linspace(0.0, max_time, num_steps)
opts = qt.Options(store_states=True, nsteps=10000)
print(opts)
print('solving...')
result = qt.mesolve(H, psi0, times, [kR * aR, kS * aS], 
                    [aS.dag() * aS, xS, pS, xS**2, pS**2, xS**4, pS**4], 
                    options=opts, progress_bar = True)
print('solved!')

date = list(str(datetime.datetime.now())[:19])
date[13] = '-'
date[16] = '-'

filepath = 'C:/Users/Wang Lab/Documents/qutip/out/test/' + ''.join(date) + '/'
#filepath = 'C:/Users/Wang Lab/Documents/qutip/out/test/laksdfjlaksdf/'
if not os.path.exists(filepath):
    os.makedirs(filepath)

for i in range(num_steps):
    pl.figure(figsize=(10,8))
    xvec = np.linspace(-7, 7, 100)

    pl.subplot(221)
    W = qt.wigner(result.states[i].ptrace(0), xvec, xvec) * np.pi
    pl.contourf(xvec, xvec, W, 100, cmap=mpl.cm.RdBu_r)
    pl.title('t=' + str(i*max_time/num_steps) + ' storage')
    pl.colorbar()
    
    pl.subplot(222)
    W = qt.wigner(result.states[i].ptrace(1), xvec, xvec) * np.pi
    pl.contourf(xvec, xvec, W, 100, cmap=mpl.cm.RdBu_r)
    pl.title('t=' + str(i*max_time/num_steps) + ' readout')
    pl.colorbar()
    
    pl.subplot(223)
    pl.title('t=' + str(i*max_time/num_steps) + ' storage occupation')
    pl.plot(range(NS), result.states[i].ptrace(0).diag())
    
    pl.subplot(224)
    pl.title('t=' + str(i*max_time/num_steps) + ' readout occupation')
    pl.plot(range(NR), result.states[i].ptrace(1).diag())

    pl.savefig(filepath + str(i) + '.png')

'''
varx = np.empty(num_steps)
varp = np.empty_like(varx)
for i in range(num_steps):
    varx[i] = 
    varp[i] = qt.variance(result.states[i], 1j*(aS - aS.dag())/2)
'''

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
pl.ylabel('var(x) and var(p)')

pl.savefig(filepath + 'expectation.png')



'''
pl.figure(figsize=(10, 8))
pl.subplot(211)
pl.plot(times, result.expect[0])
pl.xlabel('time (us)')
pl.ylabel('<n>')

pl.subplot(212)
pl.plot(times, result.expect[1], 'r')
pl.plot(times, result.expect[2], 'b')
pl.xlabel('time (us)')
pl.ylabel('<p> and <x>')

pl.savefig(filepath + 'expectation.png')
'''



np.savetxt(filepath + 'header.txt', result.states, 
         header = 
         'NS = ' + str(NS) + 
         ', NR = ' + str(NR) + 
         ', xSS = ' + str(xSS) +  
         ', xRR = ' + str(xRR) +  
         ', xSR = ' + str(xSR) +  
         ', g2 = ' + str(g2) + 
         ', eD = ' + str(eD) + 
         ', kR = ' + str(kR) +  
         ', kS = ' + str(kS) +  
         ', num_steps = ' + str(num_steps) + 
         ', max_time = ' + str(max_time))

'''
pl.plot(times, result.expect[0])
pl.xlabel('Time')
pl.ylabel('Expectation values')
pl.title('Storage')
pl.show()


W = qt.wigner(psi2.ptrace([0,1]), xvec, xvec)
pl.contourf(xvec, xvec, W, 100, cmap=mpl.cm.RdBu)
pl.show()
'''