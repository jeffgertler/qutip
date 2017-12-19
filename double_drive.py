import qutip as qt
import matplotlib.pyplot as pl
import matplotlib as mpl
import numpy as np
import datetime
import os
import itertools
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
        DB = qt.tensor(I, qt.displace(N, xvec[j] * 1j))
        m_lib[3][i][j] = DA * DB * PJ * DB.dag() * DA.dag()
    
    filepath = data_filepath + 'm_lib_[' + str(N) + ',' + str(min(xvec)) + ',' + str(max(xvec)) + ']_' + str(num_points)
    qt.fileio.qsave(m_lib, name = filepath)
    print('saved to ' + filepath)
    print('build time: ' + str(time.time()-t_start) + ' sec')
    return m_lib







''' Parameters '''
N = 20
joint_drive = -4j
loss = 0.
joint_loss = 2
confinment_loss = 2

''' Solver time steps '''
num_steps = 5
max_time = .3

''' Wigner plot parameters '''
num_points = 25
xvec = np.linspace(-2, 2, num_points)

''' Initial condition '''
#psi0 = qt.tensor(qt.fock(N, 0), qt.fock(N, 0))
alpha = 1
beta = 0
psi0 = (qt.tensor(qt.coherent(N, alpha), qt.coherent(N, beta)) - qt.tensor(qt.coherent(N, -alpha), qt.coherent(N, -beta))).unit()
#psi0 = qt.tensor(qt.fock(N, alpha), qt.fock(N, beta))


''' Setup operators '''
a = qt.tensor(qt.destroy(N), qt.qeye(N))
b = qt.tensor(qt.qeye(N), qt.destroy(N))

drive_term = (a - b)**2
confinment_term = b ** 2

loss_ops = [joint_loss * drive_term, loss * a, loss * b, confinment_term * confinment_loss]

H = joint_drive * drive_term + joint_drive.conjugate() * drive_term.dag()
times = np.linspace(0.0, max_time, num_steps, endpoint=True)


date = list(str(datetime.datetime.now())[:19])
date[13] = '-'
date[16] = '-'

''' SUPER IMPORTANT: change the filepath to wherever you want the plots saved '''
qutip_filepath = 'C:/Users/Wang Lab/Documents/qutip/'
plot_filepath = qutip_filepath + 'out/double_drive/' + ''.join(date) + '/'
data_filepath = qutip_filepath + 'data/'

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
        m_lib = qt.fileio.qload(data_filepath + 'm_lib_[' + str(N) + ',' + str(min(xvec)) + ',' + str(max(xvec)) + ']_' + str(num_points))
    except:
        print('m_lib needs to be built')
        bla


''' Solve the system or load from save'''
if 1:
    opts = qt.Options(store_states=True, nsteps=100000)
    print(opts)
    print('solving...')
#    result = qt.mesolve(H, psi0, times, [loss * a**2], 
#                        [a.dag() * a, b.dag() * b], 
#                        options=opts, progress_bar = True)
    result = qt.mesolve(H, psi0, times, loss_ops, [a.dag() * a, b.dag() * b], 
                        options=opts, progress_bar = True)
    print('solved!')
    qt.fileio.qsave(result, name = data_filepath + 'result')
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
    print('time = ' + str(i+1) + '/' + str(num_steps))
    ''' reA vs imA '''
    pl.subplot(4, num_steps, i+1)
    W = wigner4d(result.states[i], xvec, 0, m_lib)
    pl.contourf(xvec, xvec, W, np.linspace(-1.0, 1.0, 41, endpoint=True), cmap=mpl.cm.RdBu_r)
    pl.title('reA vs imA t=' + str(times[i]))
    pl.colorbar(ticks = np.linspace(-1.0, 1.0, 11, endpoint=True))
    
    ''' reB vs imB '''
    pl.subplot(4, num_steps, i+1 + num_steps)
    W = wigner4d(result.states[i], xvec, 1, m_lib)
    pl.contourf(xvec, xvec, W, np.linspace(-1.0, 1.0, 41, endpoint=True), cmap=mpl.cm.RdBu_r)
    pl.title('reB vs imB t=' + str(times[i]))
    pl.colorbar(ticks = np.linspace(-1.0, 1.0, 11, endpoint=True))
    
    ''' reA vs reB '''
    pl.subplot(4, num_steps, i+1 + 2*num_steps)
    W = wigner4d(result.states[i], xvec, 2, m_lib)
    pl.contourf(xvec, xvec, W, np.linspace(-1.0, 1.0, 41, endpoint=True), cmap=mpl.cm.RdBu_r)
    pl.title('reA vs reB t=' + str(times[i]))
    pl.colorbar(ticks = np.linspace(-1.0, 1.0, 11, endpoint=True))
    
    ''' imA vs imB '''
    pl.subplot(4, num_steps, i+1 + 3*num_steps)
    W = wigner4d(result.states[i], xvec, 3, m_lib)
    pl.contourf(xvec, xvec, W, np.linspace(-1.0, 1.0, 41, endpoint=True), cmap=mpl.cm.RdBu_r)
    pl.title('imA vs imB t=' + str(times[i]))
    pl.colorbar(ticks = np.linspace(-1.0, 1.0, 11, endpoint=True))

print(str(4 * num_steps) + ' plots created in ' + str(time.time()-t_start) + ' sec')

pl.savefig(plot_filepath + '4d_wigner_cuts.png')
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

pl.savefig(plot_filepath + 'Occupation.png')
pl.clf()

num_points = 40
xvec = np.linspace(-4, 4, num_points)

fig = pl.figure(figsize=(2.5 * num_steps, 10))
for i in range(num_steps):
    ''' Cavity A wigner '''
    pl.subplot(4, num_steps, i+1)
    W = qt.wigner(result.states[i].ptrace(0), xvec, xvec) * np.pi
    pl.contourf(xvec, xvec, W, np.linspace(-1.0, 1.0, 41, endpoint=True), cmap=mpl.cm.RdBu_r)
    pl.title('Cavity A wigner')
    pl.colorbar(ticks = np.linspace(-1.0, 1.0, 11, endpoint=True))


    ''' Cavity B wigner '''
    pl.subplot(4, num_steps, i+1 + num_steps)
    W = qt.wigner(result.states[i].ptrace(1), xvec, xvec) * np.pi
    pl.contourf(xvec, xvec, W, np.linspace(-1.0, 1.0, 41, endpoint=True), cmap=mpl.cm.RdBu_r)
    pl.title('Cavity B wigner')
    pl.colorbar(ticks = np.linspace(-1.0, 1.0, 11, endpoint=True))

pl.savefig(plot_filepath + '2d_wigner.png')
pl.clf()


''' Saving textfile with information about the run '''
np.savetxt(plot_filepath + 'header.txt', [0], 
         header = 
         'N = ' + str(N) + 
         ', joint_drive = ' + str(joint_drive) + 
         ', confinment_loss = ' + str(confinment_loss) + 
         ', loss = ' + str(loss) +  
         ', joint_loss = ' + str(joint_loss) +  
         ', num_points = ' + str(num_points) +
         ', num_steps = ' + str(num_steps) + 
         ', max_time = ' + str(max_time))



