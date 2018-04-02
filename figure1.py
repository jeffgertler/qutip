

import qutip as qt
import matplotlib.pyplot as pl
import matplotlib as mpl
import numpy as np
import datetime
import os
import itertools
import time
import imageio

''' Call must set two of the arrays to the vec desired and leave other two '''
def wigner4d(rho, xvec, cut_dim, m_lib):
    if(rho.type == 'ket'):
        rho = qt.ket2dm(rho).full()
    elif(rho.type == 'oper'):
        rho = rho.full()
    else:
        print('invalid type')
    num_points = len(xvec)
    N = len(rho)
    W = np.zeros((num_points, num_points), dtype = complex)
    for i, j in itertools.product(range(num_points), range(num_points)):
        m = m_lib[cut_dim][i][j].full()
        for k in range(N):
            W[i,j] += np.sum(rho[k,:] * m[:,k])
    return np.real(W)


#def c_linear(t, args):
#    if isinstance(t, float):
#        return max(1.0-args['v'] * t, 0.0)
#    else:
#        return np.maximum(1.0-args['v'] * t, np.zeros(len(t)))

''' Testing if middle state is steady state '''
def c_linear(t, args):
    if isinstance(t, float):
        return max(.25-args['v'] * t, 0.0)
    else:
        return np.maximum(.25-args['v'] * t, np.zeros(len(t)))


def make_plots(plot_num, num_steps, result, psi0, psi_final, N, plot_filepath):
    start_fidelity = np.zeros(num_steps)
    final_fidelity = np.zeros_like(start_fidelity)
    for i in range(num_steps):
        start_fidelity[i] = round(qt.fidelity(result.states[i], psi0), 4)
        final_fidelity[i] = round(qt.fidelity(result.states[i], psi_final), 4)
        
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
        parity[i] = round(qt.expect(p, result.states[i]), 4)
    pl.subplot(3, 1, 3)
    pl.plot(times, parity)
    pl.title('joint parity')

    pl.savefig(plot_filepath + 'Fidelity' + str(plot_num) + '.png')
    pl.clf()

    np.savetxt(plot_filepath + 'fidelity' + str(plot_num) + '.txt', [times, final_fidelity])


    print('building cat array')
    beta_steps = 50
    cat_state_arr = []
    alpha_max = np.sqrt(-2j * lam.conjugate())
    beta_arr = np.linspace(0, alpha_max, beta_steps)
    for j in range(beta_steps):
        alpha = alpha_max - beta_arr[j]
        beta = beta_arr[j]
        cat_state_arr.append((np.round(np.cos(theta/2), 5) * qt.tensor(qt.coherent(N, alpha), qt.coherent(N, beta)) 
            + np.round(np.sin(theta/2), 5) * np.exp(1j * phi) * qt.tensor(qt.coherent(N, -alpha), qt.coherent(N, -beta))).unit())

    print('calculating 2d fidelity')
    fidelity_arr = np.zeros((num_steps, beta_steps))
    for i in range(num_steps):
        print(i)
        for j in range(beta_steps):
            fidelity_arr[i][j] = round(qt.fidelity(result.states[i], cat_state_arr[j]), 4)
            
    pl.subplot(3,1,1)
    pl.imshow(fidelity_arr.transpose(), extent = (0, max_time, beta_arr[-1], beta_arr[0]))
    pl.colorbar()
    pl.ylabel('beta')
    pl.subplot(3,1,2)
    pl.plot(times, beta_arr[np.argmax(fidelity_arr, axis=1)]/alpha_max, label='current state')
    pl.plot(times, c_linear(times, args), label='steady state')
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
        pl.plot(range(N), result.states[i].ptrace(0).diag())
        pl.title('Cavity A occupation')

        ''' Cavity A occupation '''
        pl.subplot(4, num_steps, i+1 + num_steps)
        pl.plot(range(N), result.states[i].ptrace(1).diag())
        pl.title('Cavity B occupation')

    pl.savefig(plot_filepath + 'Occupation' + str(plot_num) + '.png')
    pl.clf()

    num_points = 40
    xvec_2d = np.linspace(-6, 6, num_points)

    fig = pl.figure(figsize=(2.5 * num_steps, 10))
    for i in range(num_steps):
        ''' Cavity A wigner '''
        pl.subplot(4, num_steps, i+1)
        W = qt.wigner(result.states[i].ptrace(0), xvec_2d, xvec_2d) * np.pi
        pl.contourf(xvec_2d, xvec_2d, W, np.linspace(-1.0, 1.0, 41, endpoint=True), cmap=mpl.cm.RdBu_r)
        pl.title('Cavity A wigner')
        pl.colorbar(ticks = np.linspace(-1.0, 1.0, 11, endpoint=True))


        ''' Cavity B wigner '''
        pl.subplot(4, num_steps, i+1 + num_steps)
        W = qt.wigner(result.states[i].ptrace(1), xvec_2d, xvec_2d) * np.pi
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
    #    W = wigner4d(result.states[n], xvec)
    #    pl.contourf(xvec, xvec, W, np.linspace(-1.0, 1.0, 41, endpoint=True), cmap=mpl.cm.RdBu_r)
    #    pl.title('reA vs reB t=' + str(times[n]))
    #    pl.colorbar(ticks = np.linspace(-1.0, 1.0, 11, endpoint=True))
    #    
    #print(str(num_steps) + ' plots created in ' + str(time.time()-t_start) + ' sec')
    #
    #pl.savefig(plot_filepath + 'ReRe_wigner_cuts' + str(plot_num) + '.png')
    #pl.clf()



    pl.close('all')




''' Parameters '''
N = 25
theta = np.pi / 2
phi = 0
lam = -2j
gamma = 1
v = .2

''' Wigner parameters '''
num_points = 25
xvec = np.linspace(-2, 2, num_points)

qutip_filepath = ''
data_filepath = qutip_filepath + 'data/'

print('loading m_lib')
try:
    m_lib = qt.fileio.qload(data_filepath + 'm_lib_[' + str(N) + ',' + str(min(xvec)) + ',' + str(max(xvec)) + ']_' + str(num_points))
except:
    print('m_lib needs to be built')
    bla


''' Initial condition '''
#alpha = 0
#beta = np.sqrt(-2j * lam.conjugate())
''' Testing middle state is steady state '''
alpha = 2



num_points = 40
xvec_2d = np.linspace(-3, 3, num_points)
beta_arr = np.linspace(alpha, 0, 10)

images = []
for beta in beta_arr:
    psi = (np.round(np.cos(theta/2), 5) * qt.tensor(qt.coherent(N, alpha-beta), qt.coherent(N, beta)) 
            + np.round(np.sin(theta/2), 5) * np.exp(1j * phi) * qt.tensor(qt.coherent(N, -alpha+beta), qt.coherent(N, -beta))).unit()
    #W = wigner4d(psi, xvec, 0, m_lib)
    W = qt.wigner(psi.ptrace(0), xvec_2d, xvec_2d) * np.pi

    fig, ax = pl.subplots(figsize=(5, 3))
    pl.contourf(xvec, xvec, W, np.linspace(-1.0, 1.0, 41, endpoint=True), cmap=mpl.cm.RdBu_r)
    pl.title('reA vs imA beta=' + str(beta))
    pl.colorbar(ticks = np.linspace(-1.0, 1.0, 11, endpoint=True))
    pl.savefig('out/temp.png')
    pl.clf()
    images.append(imageio.imread('out/temp.png'))
imageio.mimsave('out/fig1.gif', images, fps=1) 



