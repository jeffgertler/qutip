

import qutip as qt
import matplotlib.pyplot as pl
import matplotlib as mpl
import numpy as np
import datetime
import os
import itertools
import time
from scipy.optimize import curve_fit
import sys


date = list(str(datetime.datetime.now())[:19])
date[13] = '-'
date[16] = '-'

''' SUPER IMPORTANT: change the filepath to wherever you want the plots saved '''
qutip_filepath = os.getcwd() + '/'
plot_filepath = qutip_filepath + 'out/decay_test/' + ''.join(date) + '/'
data_filepath = qutip_filepath + 'data/'

if not os.path.exists(plot_filepath):
    os.makedirs(plot_filepath)
if not os.path.exists(data_filepath):
    os.makedirs(data_filepath)



alpha = float(sys.argv[1])
kappa = float(sys.argv[1])
N = int(round(np.absolute(alpha)**2))
while qt.coherent(N, alpha).full()[-1]/np.max(qt.coherent(N, alpha).full()) > .01:
    N+=1
print('N=', N)


num_steps = 100
max_time = 10
times = np.linspace(0, max_time, num_steps)

psi0 = qt.coherent(N, alpha)

a = qt.destroy(N)


H = qt.qeye(N)

opts = qt.Options(store_states=True, nsteps=100000)

result1 = qt.mesolve(H, psi0, times, [np.sqrt(kappa) * a], [a.dag() * a], options=opts)
result2 = qt.mesolve(H, psi0, times, [np.sqrt(kappa) * a*a], [a.dag() * a], options=opts)

                                                

alpha_steps = 100
cat_state_arr = []
alpha_arr = np.linspace(0, alpha, alpha_steps)
for j in range(alpha_steps):
    cat_state_arr.append(qt.coherent(N, alpha_arr[j]))

print('calculating 2d fidelity')
fidelity_arr1 = np.zeros((num_steps, alpha_steps))
fidelity_arr2 = np.zeros((num_steps, alpha_steps))
for i in range(num_steps):
    #print(i)
    for j in range(alpha_steps):
        fidelity_arr1[i][j] = round(qt.fidelity(result1.states[i], cat_state_arr[j]), 4)
        fidelity_arr2[i][j] = round(qt.fidelity(result2.states[i], cat_state_arr[j]), 4)







pl.figure(figsize = (10, 10))        
pl.subplot(2,2,1)
pl.imshow(fidelity_arr1.transpose(), extent = (0, max_time, alpha_arr[-1], alpha_arr[0]), aspect='auto')
pl.colorbar()
pl.ylabel('alpha')
pl.grid(True)

pl.subplot(2,2,2)
pl.imshow(fidelity_arr2.transpose(), extent = (0, max_time, alpha_arr[-1], alpha_arr[0]), aspect='auto')
pl.colorbar()
pl.ylabel('alpha')
pl.grid(True)

pl.subplot(2,2,3)
alpha_max = alpha_arr[np.argmax(fidelity_arr1, axis=1)]
pl.plot(times, alpha_max, label='1 photon loss')
x_fit = times
y_fit = alpha_max

popt, pcov = curve_fit(lambda x_fit,a_fit,b_fit,c_fit : a_fit * np.exp(b_fit * x_fit) + c_fit, x_fit, y_fit, p0 = (1, -10, y_fit[-1]))
a_fit,b_fit,c_fit = popt
print(a_fit, -1/b_fit, c_fit)
pl.plot(x_fit, a_fit * np.exp(b_fit * x_fit) + c_fit, label='exp(-t)')

pl.ylabel('alpha')
pl.legend() 
pl.grid(True)

pl.subplot(2,2,4)
alpha_max = alpha_arr[np.argmax(fidelity_arr2, axis=1)]
pl.plot(times, alpha_max, label='2 photon loss')
x_fit = times
y_fit = alpha_max

popt, pcov = curve_fit(lambda x_fit,a_fit,b_fit,c_fit : a_fit * np.exp(b_fit * x_fit) + c_fit, x_fit, y_fit, p0 = (1, -10, y_fit[-1]))
a_fit,b_fit,c_fit = popt
print(a_fit, 'tau=',-1/b_fit,'c=',c_fit)
pl.plot(x_fit, a_fit * np.exp(b_fit * x_fit) + c_fit, label='exp(-t)')

popt, pcov = curve_fit(lambda x_fit,a_fit,b_fit,c_fit : a_fit/(x_fit+b_fit) + c_fit, x_fit, y_fit, p0 = (1, .1, .1))
a_fit,b_fit,c_fit = popt
print(a_fit, b_fit, c_fit)
pl.plot(x_fit,  a_fit/(x_fit+b_fit) + c_fit, label='1/t')

pl.ylabel('alpha')
pl.legend() 
pl.grid(True)

#pl.savefig(plot_filepath + '2d_fidelity.png')
#pl.clf()

#num_points = 40
#xvec_2d = np.linspace(-6, 6, num_points)
#
#fig = pl.figure(figsize=(2.5 * num_steps, 10))
#for i in range(num_steps):
#    ''' Cavity A wigner '''
#    pl.subplot(4, num_steps, i+1)
#    W = qt.wigner(result1.states[i], xvec_2d, xvec_2d) * np.pi
#    pl.contourf(xvec_2d, xvec_2d, W, np.linspace(-1.0, 1.0, 41, endpoint=True), cmap=mpl.cm.RdBu_r)
#    pl.title('Cavity A wigner')
#    pl.colorbar(ticks = np.linspace(-1.0, 1.0, 11, endpoint=True))
#
#
#pl.savefig(plot_filepath + '2d_wigner.png')
#pl.clf()

#pl.close('all')