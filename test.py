import qutip as qt
import matplotlib.pyplot as pl
import matplotlib as mpl
import numpy as np
import datetime
import os

import profile


N = 50
alpha = 1.
drive = 9.
loss = 9.



num_steps = 5
max_time = 50




a = qt.destroy(N)


H = drive * a + drive.conjugate() * a.dag()


psi0 = qt.fock(N, 0)

''' Solve the system '''
times = np.linspace(0.0, max_time, num_steps)
opts = qt.Options(store_states=True, nsteps=100000)
print(opts)
print('solving...')
#result = qt.mesolve(H, psi0, times, [loss * a], 
#                    [a.dag() * a, a], 
#                    options=opts, progress_bar = True)

result = qt.mesolve(H*.0000001, psi0, times, [loss * (a + 2.0j * drive/loss**2 * qt.qeye(N))], 
                    [a.dag() * a, a], 
                    options=opts, progress_bar = True)
print('solved!')


num_points = 25
xvec = np.linspace(-5, 5, num_points)


fig = pl.figure(figsize=(5*num_steps, 10))
for i in range(num_steps):
    print(str(i) + '/' + str(num_steps))
    pl.subplot(2, num_steps, i+1)
    W = qt.wigner(result.states[i], xvec, xvec)
    pl.contourf(xvec, xvec, W, np.linspace(-1.0, 1.0, 41, endpoint=True), cmap=mpl.cm.RdBu_r)
    pl.title(' reA vs imA t=' + str(i*max_time/num_steps))
    pl.colorbar(ticks = np.linspace(-1.0, 1.0, 11, endpoint=True))
    
    pl.subplot(2, num_steps, i+1 + num_steps)
    pl.plot(range(N), result.states[i].diag())


#pl.show()

print(result.expect[0])

print(result.expect[1])

