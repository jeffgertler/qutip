import qutip as qt
import matplotlib.pyplot as pl
import matplotlib as mpl
import numpy as np
import datetime
import os
import itertools
import profile
import time


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
            #W[i,j] = np.real((rho * m_lib[cut_dim][i][j]).tr())
    return np.real(W)

def build_m_lib(N, num_points, xvec, data_filepath):
    print('building m_lib')
    I = qt.qeye(N)
    a = qt.tensor(qt.destroy(N), qt.qeye(N))
    b = qt.tensor(qt.qeye(N), qt.destroy(N))
    PJ = (1j * np.pi * a.dag() * a).expm() * (1j * np.pi * b.dag() * b).expm()
    t_start = time.time()
    ''' reA vs imA '''
    m_lib = [[[0 for k in range(num_points)] for j in range(num_points)] for i in range(4)]
    DB = qt.tensor(I, I)
    for i, j in itertools.product(range(num_points), range(num_points)):
        DA = qt.tensor(qt.displace(N, xvec[i] + xvec[j]*1j), I)
        m_lib[0][i][j] = DA * DB * PJ * DB.dag() * DA.dag()
    
    ''' reB vs imB '''
    DA = qt.tensor(I, I)
    for i, j in itertools.product(range(num_points), range(num_points)):
        DB = qt.tensor(I, qt.displace(N, xvec[i] + xvec[j]*1j))
        m_lib[1][i][j] = DA * DB * PJ * DB.dag() * DA.dag()
    
    ''' reA vs reB '''
    for i, j in itertools.product(range(num_points), range(num_points)):
        DA = qt.tensor(qt.displace(N, xvec[i]), I)
        DB = qt.tensor(I, qt.displace(N, xvec[j]))
        m_lib[2][i][j] = DA * DB * PJ * DB.dag() * DA.dag()
    
    ''' imA vs imB '''
    for i, j in itertools.product(range(num_points), range(num_points)):
        DA = qt.tensor(qt.displace(N, xvec[i] * 1j), I)
        DB = qt.tensor(I, qt.displace(N, xvec[j]) * 1j)
        m_lib[3][i][j] = DA * DB * PJ * DB.dag() * DA.dag()
    
    filepath = data_filepath + 'm_lib_[' + str(min(xvec)) + ',' + str(max(xvec)) + ']_' + str(num_points)
    qt.fileio.qsave(m_lib, name = filepath)
    print('saved to ' + data_filepath + 'm_lib_[' + str(min(xvec)) + ',' + str(max(xvec)) + ']_' + str(num_points))
    print('build time: ' + str(time.time()-t_start) + ' sec')
    return m_lib







''' Parameters '''
N = 20
drive = 1.5
loss = 1.2

''' Solver time steps '''
num_steps = 10
max_time = 10

''' Wigner plot parameters '''
num_points = 25
xvec = np.linspace(-2, 2, num_points)

''' Initial condition '''
psi0 = qt.tensor(qt.fock(N, 0), qt.fock(N, 0))


date = list(str(datetime.datetime.now())[:19])
date[13] = '-'
date[16] = '-'



''' SUPER IMPORTANT: change the filepath to wherever you want the plots saved '''
qutip_filepath = 'C:/Users/Wang Lab/Documents/qutip/'
plot_filepath = qutip_filepath + 'out/two_cat/' + ''.join(date) + '/'
data_filepath = qutip_filepath + 'data/two_cat/'

if not os.path.exists(plot_filepath):
    os.makedirs(plot_filepath)
if not os.path.exists(data_filepath):
    os.makedirs(data_filepath)


''' Build or load displacement libraries '''
if 0:
    m_lib = build_m_lib(N, num_points, xvec, data_filepath)
else:
    print('loading m_lib')
    try:
        m_lib = qt.fileio.qload(data_filepath + 'm_lib_[' + str(min(xvec)) + ',' + str(max(xvec)) + ']_' + str(num_points))
    except:
        print('m_lib needs to be built')
        bla

''' Optional profiling '''
if 0:
    alpha = 1.8
    psi0 = (qt.tensor(qt.coherent(N, alpha), qt.coherent(N, alpha)) -
        qt.tensor(qt.coherent(N, -alpha), qt.coherent(N, -alpha))).unit()
    profile.run("wigner4d(psi0, xvec, 'reA-reB', dis_lib); print()")
    profile.run("wigner4d(psi0, xvec, 'reA-imA', dis_lib); print()")

''' Setup Hamiltonian '''
a = qt.tensor(qt.destroy(N), qt.qeye(N))
b = qt.tensor(qt.qeye(N), qt.destroy(N))

H = drive * (a + b)**2 + drive.conjugate() * ((a + b)**2).dag()

''' Solve the system or load from save'''
if 1:
    times = np.linspace(0.0, max_time, num_steps)
    opts = qt.Options(store_states=True, nsteps=100000)
    print(opts)
    print('solving...')
    result = qt.mesolve(H, psi0, times, [loss * (a + b)**2], 
                        [a.dag() * a, b.dag() * b], 
                        options=opts, progress_bar = True)
    print('solved!')
    qt.fileio.qsave(m_lib, name = data_filepath + 'result')
else:
    print('loading result')
    try:
        m_lib = qt.fileio.qload(data_filepath + 'result')
    except:
        print('result needs to be solved')




print('plotting')
t_start = time.time()
fig = pl.figure(figsize=(5 * num_steps, 20))
for i in range(num_steps):
    print('time = ' + str(times[i]) + '/' + str(times[-1]))
    ''' reA vs imA '''
    pl.subplot(4, num_steps, i+1)
    W = wigner4d(result.states[i], xvec, 0, m_lib)
    pl.contourf(xvec, xvec, W, np.linspace(-1.0, 1.0, 41, endpoint=True), cmap=mpl.cm.RdBu_r)
    pl.title(' reA vs imA')
    pl.colorbar(ticks = np.linspace(-1.0, 1.0, 11, endpoint=True))
    
    ''' reB vs imB '''
    pl.subplot(4, num_steps, i+1 + num_steps)
    W = wigner4d(result.states[i], xvec, 1, m_lib)
    pl.contourf(xvec, xvec, W, np.linspace(-1.0, 1.0, 41, endpoint=True), cmap=mpl.cm.RdBu_r)
    pl.title('reB vs imB')
    pl.colorbar(ticks = np.linspace(-1.0, 1.0, 11, endpoint=True))
    
    ''' reA vs reB '''
    pl.subplot(4, num_steps, i+1 + 2*num_steps)
    W = wigner4d(result.states[i], xvec, 2, m_lib)
    pl.contourf(xvec, xvec, W, np.linspace(-1.0, 1.0, 41, endpoint=True), cmap=mpl.cm.RdBu_r)
    pl.title('reA vs reB')
    pl.colorbar(ticks = np.linspace(-1.0, 1.0, 11, endpoint=True))
    
    ''' imA vs imB '''
    pl.subplot(4, num_steps, i+1 + 3*num_steps)
    W = wigner4d(result.states[i], xvec, 3, m_lib)
    pl.contourf(xvec, xvec, W, np.linspace(-1.0, 1.0, 41, endpoint=True), cmap=mpl.cm.RdBu_r)
    pl.title('imA vs imB')
    pl.colorbar(ticks = np.linspace(-1.0, 1.0, 11, endpoint=True))

print(str(4 * num_steps) + ' plots created in ' + str(time.time()-t_start) + ' sec')

pl.savefig(plot_filepath + '4d_wigner_cuts.png')
pl.clf()

''' Saving textfile with information about the run '''
np.savetxt(plot_filepath + 'header.txt', [0], 
         header = 
         'N = ' + str(N) + 
         ', drive = ' + str(drive) + 
         ', loss = ' + str(loss) +  
         ', num_points = ' + str(num_points) +
         ', num_steps = ' + str(num_steps) + 
         ', max_time = ' + str(max_time))



