import qutip as qt
import matplotlib.pyplot as pl
import matplotlib as mpl
import numpy as np



N = 30 #max cavity excitation
'''
aQ = qt.tensor(qt.destroy(2), qt.qeye(N), qt.qeye(N)) # qubit
aS = qt.tensor(qt.qeye(2), qt.destroy(N), qt.qeye(N)) # storage
aR = qt.tensor(qt.qeye(2), qt.qeye(N), qt.destroy(N)) # readout
'''
aS = qt.tensor(qt.destroy(N), qt.qeye(N))
aR = qt.tensor(qt.qeye(N), qt.destroy(N))

wQ = 4900.7
wS = 7578.6
wR = 7152.0

xQQ = 130.
xSS = .004
xRR = 2.14
xQS = 1.585
xSR = .206
xRQ = 35.

g2 = .111
eD = 1
kR = .8
'''
H_shift = (wQ * aQ.dag() * aQ 
		+ wS * aS.dag() * aS 
		+ wR * aR.dag() * aR)
'''
H_kerr = (
#		- xQQ/2 * aQ.dag()**2 * aQ**2 
		- xSS/2 * aS.dag()**2 * aS**2 
		- xRR/2 * aR.dag()**2 * aR**2
#		- xQS * aQ.dag() * aQ * aS.dag() * aS
		- xSR * aS.dag() * aS * aR.dag() * aR
#		- xRQ * aR.dag() * aR * aQ.dag() * aQ
		)
H_2 = (g2 * aS**2 * aR.dag() 
		+ g2.conjugate() * aS.dag()**2 * aR
		+ eD * aR.dag() 
		+ eD.conjugate() * aR)

H = H_kerr + H_2

'''
rho0 = qt.tensor(qt.fock(2, 0), 
				 qt.fock(N, 0), 
				 qt.fock(N, 0))
'''
rho0 = qt.tensor(qt.fock(N, 0), qt.fock(N, 0))

num_steps = 20
max_time = 20
times = np.linspace(0.0, max_time, num_steps)

if 1: #run or load the result
	opts = qt.Options(store_states=True, nsteps=10000)
	print(opts)
	print('solving...')
	result = qt.mesolve(H, rho0, times, [kR * aR], [aS.dag()*aS, aR.dag()*aR], options=opts)
	print('Solved! Saving and plotting...')
	qt.qsave(result, 'leghtas_test')
else:
	result = qt.qload('leghtas_test')


# plotting wigner topography
if 1:
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
		value = result.states[i].ptrace(0).diag()
		pl.plot(range(N), value)

		pl.subplot(224)
		pl.title('t=' + str(i*max_time/num_steps) + ' readout occupation')
		value = result.states[i].ptrace(1).diag()
		pl.plot(range(N), value)

		pl.savefig('out/leghtas/' + str(i) + '.png')
