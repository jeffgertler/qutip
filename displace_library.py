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
        rho = qt.ket2dm(rho)
    num_points = len(xvec)
    W = np.empty((num_points, num_points), dtype = float)
    for i, j in itertools.product(range(num_points), range(num_points)):
        W[i,j] = np.real((rho * m_lib[cut_dim][i][j]).tr())
    return W



N = 20
alpha = 1.8

date = list(str(datetime.datetime.now())[:19])
date[13] = '-'
date[16] = '-'

''' SUPER IMPORTANT: change the filepath to wherever you want the plots saved '''
qutip_filepath = 'C:/Users/Wang Lab/Documents/qutip/'
plot_filepath = qutip_filepath + 'out/displace_library/' + ''.join(date) + '/'
data_filepath = qutip_filepath + 'data/'

if not os.path.exists(plot_filepath):
    os.makedirs(plot_filepath)
if not os.path.exists(data_filepath):
    os.makedirs(data_filepath)

psi0 = (qt.tensor(qt.coherent(N, alpha), qt.coherent(N, alpha)) -
        qt.tensor(qt.coherent(N, -alpha), qt.coherent(N, -alpha))).unit()



num_points = 25
xvec = np.linspace(-2, 2, num_points)



''' Build or load displacement libraries '''
I = qt.qeye(N)
a = qt.tensor(qt.destroy(N), qt.qeye(N))
b = qt.tensor(qt.qeye(N), qt.destroy(N))
PJ = (1j * np.pi * a.dag() * a).expm() * (1j * np.pi * b.dag() * b).expm()
if 0:
    print('building m_lib')
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
        
    qt.fileio.qsave(m_lib, name= data_filepath + 'm_lib_[' + str(min(xvec)) + ',' + str(max(xvec)) + ']_' + str(num_points))
    print('build time: ' + str(time.time()-t_start) + ' sec')
else:
    print('loading m_lib')
    try:
        m_lib = qt.fileio.qload(data_filepath + 'm_lib_[' + str(min(xvec)) + ',' + str(max(xvec)) + ']_' + str(num_points))
    except:
        print('m_lib needs to be built')



if 0:
    profile.run("wigner4d(psi0, xvec, 'reA-reB', dis_lib); print()")
    profile.run("wigner4d(psi0, xvec, 'reA-imA', dis_lib); print()")


print('plotting')
t_start = time.time()
fig = pl.figure(figsize=(5, 20))
''' reA vs imA '''
pl.subplot(4, 1, 1)
W = wigner4d(psi0, xvec, 0, m_lib)
pl.contourf(xvec, xvec, W, np.linspace(-1.0, 1.0, 41, endpoint=True), cmap=mpl.cm.RdBu_r)
pl.title(' reA vs imA')
pl.colorbar(ticks = np.linspace(-1.0, 1.0, 11, endpoint=True))

''' reB vs imB '''
pl.subplot(4, 1, 2)
W = wigner4d(psi0, xvec, 1, m_lib)
pl.contourf(xvec, xvec, W, np.linspace(-1.0, 1.0, 41, endpoint=True), cmap=mpl.cm.RdBu_r)
pl.title('reB vs imB')
pl.colorbar(ticks = np.linspace(-1.0, 1.0, 11, endpoint=True))


''' reA vs reB '''
pl.subplot(4, 1, 3)
W = wigner4d(psi0, xvec, 2, m_lib)
pl.contourf(xvec, xvec, W, np.linspace(-1.0, 1.0, 41, endpoint=True), cmap=mpl.cm.RdBu_r)
pl.title('reA vs reB')
pl.colorbar(ticks = np.linspace(-1.0, 1.0, 11, endpoint=True))

''' imA vs imB '''
pl.subplot(4, 1, 4)
W = wigner4d(psi0, xvec, 3, m_lib)
pl.contourf(xvec, xvec, W, np.linspace(-1.0, 1.0, 41, endpoint=True), cmap=mpl.cm.RdBu_r)
pl.title('imA vs imB')
pl.colorbar(ticks = np.linspace(-1.0, 1.0, 11, endpoint=True))

print('4 plots created in ' + str(time.time()-t_start) + ' sec')

pl.savefig(plot_filepath + '4d_wigner_cuts.png')
pl.clf()

