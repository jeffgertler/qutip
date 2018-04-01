

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

''' Call must set two of the arrays to the vec desired and leave other two '''
def wigner4d(rho, xvec):
    if(rho.type == 'ket'):
        rho = qt.ket2dm(rho)
    elif(rho.type is not 'oper'):
        print('invalid type')
        
    num_points = len(xvec)
    N = rho.shape[0]
    I = qt.qeye(N)
    a = qt.tensor(qt.destroy(N), qt.qeye(N))
    b = qt.tensor(qt.qeye(N), qt.destroy(N))
    PJ = (1j * np.pi * a.dag() * a).expm() * (1j * np.pi * b.dag() * b).expm()
    
    W = np.zeros((num_points, num_points), dtype = complex)
    for i, j in itertools.product(range(num_points), range(num_points)):
        print(i)
        DA = qt.tensor(qt.displace(N, xvec[i]), I)
        DB = qt.tensor(I, qt.displace(N, xvec[j]))
        W[i][j] = (rho * DA * DB * PJ * DB.dag() * DA.dag()).tr()
    return np.real(W)


def c_linear(t, args):
    t += args['start_time']
    if isinstance(t, float):
        return (max(1.0-args['v'] * t, 0.0))**2
    else:
        return (np.maximum(1.0-args['v'] * t, np.zeros(len(t))))**2

''' Testing if middle state is steady state '''
#def c_linear(t, args):
#    if isinstance(t, float):
#        return (max(.15-args['v'] * t, 0.0))**2
#    else:
#        return np.power(np.maximum(.15-args['v'] * t, np.zeros(len(t))), 2)


def make_plots(plot_num, num_steps, times, states, psi0, psi_final, N, plot_filepath):
    start_fidelity = np.zeros(num_steps)
    final_fidelity = np.zeros_like(start_fidelity)
    for i in range(num_steps):
        start_fidelity[i] = round(qt.fidelity(states[i], psi0), 4)
        final_fidelity[i] = round(qt.fidelity(states[i], psi_final), 4)
        
    fit_start = 20
    y_fit = final_fidelity[fit_start:]
    x_fit = times[fit_start:]

    ''' Fidelity with starting cat state '''
    fig = pl.figure(figsize=(10,15))
    pl.subplot(3, 1, 1)
    pl.plot(times, start_fidelity, '.')
    pl.title('Fidelity with starting cat state')

    ''' Fidelity with final cat state '''
    pl.subplot(3, 1, 2)
    pl.plot(times, final_fidelity, '.')

    try:
        popt, pcov = curve_fit(lambda x_fit,a_fit,b_fit,c_fit : a_fit * np.exp(b_fit * x_fit) + c_fit, x_fit, y_fit, p0 = (-1, -.1, 0))
        a_fit,b_fit,c_fit = popt
        print(a_fit, b_fit, c_fit)
        pl.plot(x_fit, a_fit * np.exp(b_fit * x_fit) + c_fit)
        pl.title(str(a_fit) + ',' + str(b_fit) + ',' + str(c_fit))
    except:
        print('fit didnt work')


    p = (1j * np.pi * (a.dag() * a + b.dag() * b)).expm()
    parity = np.zeros(num_steps)
    for i in range(num_steps):
        parity[i] = round(qt.expect(p, states[i]), 4)
    pl.subplot(3, 1, 3)
    pl.plot(times, parity)
    pl.title('joint parity')

    pl.savefig(plot_filepath + 'Fidelity' + str(plot_num) + '.png')
    pl.clf()

    np.savetxt(plot_filepath + 'fidelity' + str(plot_num) + '.txt', [times, final_fidelity])


    print('building cat array')
    beta_steps = 10
    cat_state_arr = []
    beta_arr = np.linspace(0, alpha, beta_steps)
    for j in range(beta_steps):
        cat_state_arr.append((np.round(np.cos(theta/2), 5) * qt.tensor(qt.coherent(N, alpha-beta_arr[j]), 
                                                                       qt.coherent(N, beta_arr[j])) 
         + np.round(np.sin(theta/2), 5) * np.exp(1j * phi) * qt.tensor(qt.coherent(N, -alpha+beta_arr[j]), 
                                                                       qt.coherent(N, -beta_arr[j]))).unit())

    print('calculating 2d fidelity')
    fidelity_arr = np.zeros((num_steps, beta_steps))
    for i in range(num_steps):
        #print(i)
        for j in range(beta_steps):
            fidelity_arr[i][j] = round(qt.fidelity(states[i], cat_state_arr[j]), 4)
            
    pl.subplot(3,1,1)
    pl.imshow(fidelity_arr.transpose(), extent = (0, max_time, beta_arr[-1]/alpha, beta_arr[0]/alpha))
    pl.colorbar()
    pl.ylabel('beta/alpha')
    pl.subplot(3,1,2)
    pl.plot(times, beta_arr[np.argmax(fidelity_arr, axis=1)]/alpha, label='current state')
    pl.plot(times, np.sqrt(c_linear(times, args)), label='steady state')
    pl.ylabel('beta/alpha')
    pl.legend() 
    pl.subplot(3,1,3)
    pl.plot(times, np.max(fidelity_arr, axis=1))
    pl.ylabel('fidelity')
    pl.xlabel('time')
    pl.savefig(plot_filepath + '2d_fidelity' + str(plot_num) + '.png')
    pl.clf()



    fig = pl.figure(figsize=(5 * num_steps, 10))
    for i in range(num_steps):
        ''' Cavity A occupation '''
        pl.subplot(4, num_steps, i+1)
        pl.plot(range(N), states[i].ptrace(0).diag())
        pl.title('Cavity A occupation')

        ''' Cavity A occupation '''
        pl.subplot(4, num_steps, i+1 + num_steps)
        pl.plot(range(N), states[i].ptrace(1).diag())
        pl.title('Cavity B occupation')

    pl.savefig(plot_filepath + 'Occupation' + str(plot_num) + '.png')
    pl.clf()

    num_points = 40
    xvec_2d = np.linspace(-6, 6, num_points)

    fig = pl.figure(figsize=(2.5 * num_steps, 10))
    for i in range(num_steps):
        ''' Cavity A wigner '''
        pl.subplot(4, num_steps, i+1)
        W = qt.wigner(states[i].ptrace(0), xvec_2d, xvec_2d) * np.pi
        pl.contourf(xvec_2d, xvec_2d, W, np.linspace(-1.0, 1.0, 41, endpoint=True), cmap=mpl.cm.RdBu_r)
        pl.title('Cavity A wigner')
        pl.colorbar(ticks = np.linspace(-1.0, 1.0, 11, endpoint=True))


        ''' Cavity B wigner '''
        pl.subplot(4, num_steps, i+1 + num_steps)
        W = qt.wigner(states[i].ptrace(1), xvec_2d, xvec_2d) * np.pi
        pl.contourf(xvec_2d, xvec_2d, W, np.linspace(-1.0, 1.0, 41, endpoint=True), cmap=mpl.cm.RdBu_r)
        pl.title('Cavity B wigner')
        pl.colorbar(ticks = np.linspace(-1.0, 1.0, 11, endpoint=True))

    pl.savefig(plot_filepath + '2d_wigner' + str(plot_num) + '.png')
    pl.clf()



    #num_points = 5
    #xvec = np.linspace(-4, 4, num_points)
    #t_start = time.time()
    #
    #fig = pl.figure(figsize=(5 * num_steps, 5))
    #for n in range(num_steps):
    #    ''' reA vs reB '''
    #    pl.subplot(1, num_steps, n+1)
    #    W = wigner4d(states[n], xvec)
    #    pl.contourf(xvec, xvec, W, np.linspace(-1.0, 1.0, 41, endpoint=True), cmap=mpl.cm.RdBu_r)
    #    pl.title('reA vs reB t=' + str(times[n]))
    #    pl.colorbar(ticks = np.linspace(-1.0, 1.0, 11, endpoint=True))
    #    
    #print(str(num_steps) + ' plots created in ' + str(time.time()-t_start) + ' sec')
    #
    #pl.savefig(plot_filepath + 'ReRe_wigner_cuts' + str(plot_num) + '.png')
    #pl.clf()



    pl.close('all')


print(sys.argv)
#theta = float(sys.argv[1]) * np.pi
#phi = float(sys.argv[2]) * np.pi
#lam = complex(sys.argv[3])
#gamma = float(sys.argv[4])
#v = float(sys.argv[5])
theta = np.pi / 2
phi = 0
lam = -2j
gamma = 1
v = .2


''' Parameters '''
N = 15
joint_drive = lam
loss = 0.
joint_loss = 1
confinment_loss = gamma

''' Solver time steps '''
num_steps = 20
max_time = 5
times = np.linspace(0.0, max_time, num_steps, endpoint=True)
break_points = [0, 5, 10, 15, num_steps]


''' Initial condition '''
#alpha = 0
#beta = np.sqrt(-2j * lam.conjugate())
''' Testing middle state is steady state '''
alpha = np.sqrt(-2j * lam.conjugate())
beta = alpha
psi0 = (np.round(np.cos(theta/2), 5) * qt.tensor(qt.coherent(N, alpha-beta), 
                                                 qt.coherent(N, beta)) 
        + np.round(np.sin(theta/2), 5) * np.exp(1j * phi) * qt.tensor(qt.coherent(N, -alpha+beta), 
                                                                      qt.coherent(N, -beta))).unit()

''' guess at final condition '''
beta = 0
psi_final = (np.round(np.cos(theta/2), 5) * qt.tensor(qt.coherent(N, alpha-beta), qt.coherent(N, beta)) 
            + np.round(np.sin(theta/2), 5) * np.exp(1j * phi) * qt.tensor(qt.coherent(N, -alpha+beta), qt.coherent(N, -beta))).unit()

''' Setup operators '''
a = qt.tensor(qt.destroy(N), qt.qeye(N))
b = qt.tensor(qt.qeye(N), qt.destroy(N))

drive_term = (a + b) **2
confinment_term = b ** 2


loss_ops = [joint_loss * drive_term, confinment_term * confinment_loss]
#loss_ops = [joint_loss * drive_term, loss * a, loss * b, confinment_term * confinment_loss]
#loss_ops = [joint_loss * drive_term, loss * a, loss * b, [confinment_term * confinment_loss, c_tanh]]


H = [joint_drive * drive_term + joint_drive.conjugate() * drive_term.dag(),
     [joint_drive * confinment_term + joint_drive.conjugate() * confinment_term.dag(), c_linear]]


date = list(str(datetime.datetime.now())[:19])
date[13] = '-'
date[16] = '-'

''' SUPER IMPORTANT: change the filepath to wherever you want the plots saved '''
qutip_filepath = ''
plot_filepath = qutip_filepath + 'out/adiabat/' + ''.join(date) + '/'
data_filepath = qutip_filepath + 'data/'

if not os.path.exists(plot_filepath):
    os.makedirs(plot_filepath)
if not os.path.exists(data_filepath):
    os.makedirs(data_filepath)

''' Solve the system or load from save'''
args = {'v': v, 'start_time':0.0}
if 1:
    print('solving...')
    opts = qt.Options(store_states=True, nsteps=100000)
    psi_current = psi0
    states = []

    for i in range(1, len(break_points)):
        print('sub_simulation #' + str(i))

        
        args['start_time'] = times[break_points[i-1]]
        result = qt.mesolve(H, psi_current, times[break_points[i-1]:break_points[i]]- times[break_points[i-1]]
                            , loss_ops, [a.dag() * a, b.dag() * b], 
                            options=opts, progress_bar = True, args = args)
        states += result.states
        print('solved!')

        psi_current = states[-1]

        print('plotting')
        plot_num = i

        args['start_time'] = 0.0
        if i == len(break_points)-1:
            plot_filepath += 'full/'
            os.makedirs(plot_filepath)
        make_plots(plot_num, break_points[i], times[0:break_points[i]], states, psi0, psi_final, N, plot_filepath)

        
        qt.fileio.qsave(result, name = data_filepath + 'result' + str(plot_num))

else:
    print('loading result')
    try:
        states = []
        for i in range(1, len(break_points)):
            result = qt.fileio.qload(data_filepath + 'result' + str(plot_num))
            states += result.states
    except:
        print('result needs to be solved')

    make_plots(0, num_steps, times, states, psi0, psi_final, N, plot_filepath)



''' Saving textfile with information about the run '''
np.savetxt(plot_filepath + 'header.txt', [0], 
         header = 
         'N = ' + str(N) + 
         '\n joint_drive = ' + str(joint_drive) + 
         '\n confinment_loss = ' + str(confinment_loss) + 
         '\n loss = ' + str(loss) +  
         '\n joint_loss = ' + str(joint_loss) +  
         '\n alpha = ' + str(alpha) + 
         '\n beta = ' + str(beta) + 
         '\n theta = ' + str(theta) + 
         '\n phi = ' + str(phi) + 
         '\n num_steps = ' + str(num_steps) + 
         '\n max_time = ' + str(max_time))



