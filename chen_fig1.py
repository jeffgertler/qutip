import numpy as np
import matplotlib.pyplot as pl
import matplotlib as mpl
from scipy.misc import factorial
import itertools
import qutip as qt
from scipy.optimize import curve_fit
import imageio



def qfunc_rere(N, rho, x_vec):
	#if(rho.type == 'ket'):
	#	rho = qt.ket2dm(rho)
	#elif(rho.type is not 'oper'):
	#	print('invalid type')

	num_points = len(xvec)
	Q = np.zeros((num_points, num_points), dtype = complex)
	for i, j in itertools.product(range(num_points), range(num_points)):
		#a = qt.tensor(qt.coherent(N, x_vec[i]), qt.qeye(N))
		#b = qt.tensor(qt.qeye(N), qt.coherent(N, x_vec[j]))
		alpha = np.array(qt.tensor(qt.coherent(N, x_vec[i]), qt.coherent(N, x_vec[j])).full())

		Q[i][j] = np.absolute(np.matmul(np.matmul(alpha.T, rho), alpha))
		#Q[i][j] = (2*np.pi)**2 * a.dag()*(b.dag()*rho*b)*a
		#Q[i][j] = (2*np.pi)**2 * a
	return Q

def poisson(x, alpha):
	return np.exp(-alpha) * np.power(alpha, x) / np.sqrt(factorial(x))

def P_coh(x, alpha):
	return np.exp(-np.abs(alpha)**2 / 2) * np.power(alpha, x) / np.sqrt(factorial(x))

mpl.rcParams.update({'font.size': 12})


''' Plot of coherent state probablility '''
if 0:
	N = 20
	alpha = 2

	plot_cutoff = 13

	psi = qt.coherent(N, alpha)

	n = np.arange(plot_cutoff)
	pl.figure(figsize=(7, 5))
	mpl.rcParams.update({'font.size': 15})
	pl.bar(n, np.power(np.abs(psi.full()), 2)[:plot_cutoff], color='b')
	x = np.linspace(0, plot_cutoff, 100)
	pl.plot(x, P_coh(x, alpha)**2, 
			label=r'$e^{-|\alpha|^2}\alpha^{2n}\frac{1}{n!}$', color='k')
	pl.xlabel('|n>')
	pl.ylabel(r'|<$\psi$|n>|$^2$')
	pl.xticks(n)
	pl.legend()
	pl.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
	pl.savefig('out/chen_fig1.png')
	pl.clf()

''' Even cat state probablility '''
if 0:
	N = 20
	alpha = 2

	plot_cutoff = 13

	psi = (qt.coherent(N, alpha) + qt.coherent(N, -alpha)).unit()

	n = np.arange(plot_cutoff)
	pl.figure(figsize=(7, 5))
	mpl.rcParams.update({'font.size': 15})
	pl.bar(n, np.power(np.abs(psi.full()), 2)[:plot_cutoff], color='b')
	x = np.linspace(0, plot_cutoff, 100)
	pl.plot(x, 2*P_coh(x, alpha)**2, 
			label=r'$2e^{-|\alpha|^2}\alpha^{2n}\frac{1}{n!}$', color='k')
	pl.xlabel('|n>')
	pl.ylabel(r'|<$\psi$|n>|$^2$')
	pl.xticks(n)
	pl.legend()
	pl.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
	pl.savefig('out/chen_fig2.png')
	pl.clf()

''' Odd cat state probability '''
if 0:
	N = 20
	alpha = 2

	plot_cutoff = 13

	psi = (qt.coherent(N, alpha) - qt.coherent(N, -alpha)).unit()

	n = np.arange(plot_cutoff)
	pl.figure(figsize=(7, 5))
	mpl.rcParams.update({'font.size': 15})
	pl.bar(n, np.power(np.abs(psi.full()), 2)[:plot_cutoff], color='b')
	x = np.linspace(0, plot_cutoff, 100)
	pl.plot(x, 2*P_coh(x, alpha)**2, 
			label=r'$2e^{-|\alpha|^2}\alpha^{2n}\frac{1}{n!}$', color='k')
	pl.xlabel('|n>')
	pl.ylabel(r'|<$\psi$|n>|$^2$')
	pl.xticks(n)
	pl.legend()
	pl.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
	pl.savefig('out/chen_fig3.png')
	pl.clf()


''' Simulation of single photon loss on coherent state '''
if 0:
	N = 40
	alpha = 2
	kappa = 1

	psi0 = qt.coherent(N, alpha)
	times = np.linspace(0, .7, 100, endpoint='True')
	a = qt.destroy(N)
	H = qt.qeye(N)
	loss_ops = [np.sqrt(kappa) * a]

	opts = qt.Options(store_states=True, nsteps=100000)
	result = qt.mesolve(H, psi0, times, loss_ops, 
						options=opts, progress_bar = True)
	psi = result.states[-1]
	#psi = qt.ket2dm(psi0)

	plot_cutoff = 13
	
	x_fit = np.arange(N)
	y_fit = np.abs(psi.diag())

	#popt, pcov = curve_fit(lambda x_fit,a_fit : np.exp(-a_fit*2) * np.power(a_fit, 2*x_fit) / factorial(x_fit), x_fit, y_fit, p0 = (1))
	popt, pcov = curve_fit(P_coh, x_fit, y_fit)
	print('alpha=', popt[0], ' <N>=', qt.expect(psi, a.dag()*a), 'fidelity=', qt.fidelity(psi, qt.coherent(N, popt[0])))

	x = np.linspace(0, plot_cutoff, 100)
	popt[0] = np.sqrt(qt.expect(psi, a.dag()*a))
	pl.figure(figsize=(7, 5))
	mpl.rcParams.update({'font.size': 15})
	pl.plot(x, P_coh(x, popt[0])**4, label='alpha=' + str(popt[0]) + '\nN=' + str(popt[0]**2), color='k')
	n = np.arange(plot_cutoff)
	pl.bar(n, np.power(np.abs(psi.diag()), 2)[:plot_cutoff], color='b')
	x = np.linspace(0, plot_cutoff, 100)
	pl.xlabel('|n>')
	pl.ylabel(r'|<$\psi$|n>|$^2$')
	pl.xticks(n)
	pl.legend()
	pl.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
	pl.savefig('out/chen_fig4.png')
	pl.clf()

''' |4> fock state probability '''
if 0:
	N = 20
	alpha = 2

	plot_cutoff = 7

	psi = qt.fock(N, 4)

	n = np.arange(plot_cutoff)
	pl.figure(figsize=(7, 5))
	mpl.rcParams.update({'font.size': 15})
	pl.bar(n, np.power(np.abs(psi.full()), 2)[:plot_cutoff], color='b')
	pl.xlabel('|n>')
	pl.ylabel(r'|<$\psi$|n>|$^2$')
	pl.xticks(n)
	pl.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
	pl.savefig('out/chen_fig5.png')
	pl.clf()

''' |4> fock state single photon loss '''
if 0:
	N = 20
	alpha = 2
	kappa = 1

	plot_cutoff = 7

	psi0 = qt.fock(N, 4)
	times = np.linspace(0, .7, 100, endpoint='True')
	a = qt.destroy(N)
	H = qt.qeye(N)
	loss_ops = [np.sqrt(kappa) * a]

	opts = qt.Options(store_states=True, nsteps=100000)
	result = qt.mesolve(H, psi0, times, loss_ops, 
						options=opts, progress_bar = True)
	psi = result.states[-1]

	n = np.arange(plot_cutoff)


	probs = np.round(np.abs(psi.diag())[:5], 3)*100
	pl.figure(figsize=(30, 10))
	mpl.rcParams.update({'font.size': 30})
	for i in range(5):
		pl.subplot(1,5,i+1)
		psi_plot = qt.ket2dm(qt.fock(N, i))
		pl.bar(n, np.abs(psi_plot.diag())[:plot_cutoff], color='b', label=str(probs[i])+'%')
		pl.xlabel('|n>')
		pl.ylabel(r'|<$\psi$|n>|$^2$')
		pl.xticks(n)
		pl.legend()
	pl.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
	pl.savefig('out/chen_fig6.png')
	pl.clf()


if 0:
	num_steps = 20
	N = 20
	alpha_max = 2
	eps = -alpha_max**2 * 2.0j
	kappa = 1
	xvec = np.linspace(-5, 5, 30, endpoint='True')

	times = np.linspace(0, .15, num_steps, endpoint='True')
	a = qt.tensor(qt.destroy(N), qt.qeye(N))
	b = qt.tensor(qt.qeye(N), qt.destroy(N))
	H = eps*(a+b)**2 + (eps*(a+b)**2).dag()
	loss_ops = [np.sqrt(kappa) * (a+b)**2]

	psi0 = qt.tensor(qt.fock(N, 0), qt.fock(N, 0))

	opts = qt.Options(store_states=True, nsteps=100000)
	result = qt.mesolve(H, psi0, times, loss_ops, 
						options=opts, progress_bar = True)


	images = []
	for i in range(num_steps):
		Q = qfunc_rere(N, result.states[i].full(), xvec)
		pl.figure(figsize=(7, 5))
		mpl.rcParams.update({'font.size': 15})
		pl.contourf(xvec, xvec, Q, np.linspace(-1.0, 1.0, 41, endpoint=True), cmap=mpl.cm.RdBu_r)
		pl.colorbar(ticks = np.linspace(-1.0, 1.0, 11, endpoint=True))
		#pl.savefig('out/chen_fig6.png')
		#pl.clf()

		pl.xlabel('Re[A]')
		pl.ylabel('Re[B]')
		pl.title('t='+str(np.round(times[i], 3)))
		pl.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
		pl.savefig('out/temp.png')
		pl.clf()
		images.append(imageio.imread('out/temp.png'))
	imageio.mimsave('out/chen_gif1.gif', images, fps=4) 


if 1:
	num_steps = 20
	N = 20
	alpha = 2
	eps = -alpha**2 * 2.0j
	kappa = 1
	xvec = np.linspace(-5, 5, 30, endpoint='True')

	beta_arr = np.linspace(alpha, 0, num_steps, endpoint=True)
	sqrt_beta_arr = np.linspace(alpha*np.sqrt(2), 0, num_steps, endpoint=True)

	images = []
	for i in range(num_steps):
		pl.figure(figsize=(7, 5))
		mpl.rcParams.update({'font.size': 15})
		x = np.linspace(-6, 6, 10)
		y = -x + alpha * np.sqrt(2)

		pl.plot(x, y, 'k', label=r'$(a+b)^2=\alpha^2$')
		pl.plot(x, y-2*(alpha*np.sqrt(2)), 'k')
		x_points = np.array([sqrt_beta_arr[i], sqrt_beta_arr[i], -sqrt_beta_arr[i], -sqrt_beta_arr[i]])
		y_points = np.array([-sqrt_beta_arr[i] + alpha * np.sqrt(2), -sqrt_beta_arr[i] + alpha * np.sqrt(2)-2*(alpha*np.sqrt(2)),
							sqrt_beta_arr[i] + alpha * np.sqrt(2), sqrt_beta_arr[i] + alpha * np.sqrt(2)-2*(alpha*np.sqrt(2))])
		s = 100.0 * np.ones_like(x_points)
		pl.scatter(x_points, y_points, s=s, color='r', label='steady states')

		pl.xlim(-6, 6)
		pl.ylim(-6, 6)
		pl.axvline(sqrt_beta_arr[i], label=r'$a^2=\beta^2$')
		pl.axvline(-sqrt_beta_arr[i])

		pl.xlabel('Re[A]')
		pl.ylabel('Re[B]')
		pl.legend()
		pl.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
		pl.savefig('out/temp.png')
		pl.clf()
		images.append(imageio.imread('out/temp.png'))
	imageio.mimsave('out/chen_gif2.gif', images, fps=4) 


if 1:
	num_steps = 20
	N = 20
	alpha = 2
	eps = -alpha**2 * 2.0j
	kappa = 1
	xvec = np.linspace(-5, 5, 30, endpoint='True')

	beta_arr = np.linspace(alpha/2, alpha, num_steps, endpoint=True)
	sqrt_beta_arr = np.linspace(alpha/2*np.sqrt(2), alpha * np.sqrt(2), num_steps, endpoint=True)

	images = []
	for i in range(num_steps):
		pl.figure(figsize=(7, 5))
		mpl.rcParams.update({'font.size': 15})
		x = np.linspace(-6, 6, 10)
		y = -x + alpha * np.sqrt(2)

		pl.plot(x, y, 'k', label=r'$(a+b)^2=\alpha^2$')
		pl.plot(x, y-2*(alpha*np.sqrt(2)), 'k')
		x_points = np.array([sqrt_beta_arr[i], -sqrt_beta_arr[i]])
		y_points = np.array([-sqrt_beta_arr[i] + alpha * np.sqrt(2), sqrt_beta_arr[i] + alpha * np.sqrt(2)-2*(alpha*np.sqrt(2))])
		s = 100.0 * np.ones_like(x_points)
		pl.scatter(x_points, y_points, s=s, color='r', label='steady states')

		pl.xlim(-6, 6)
		pl.ylim(-6, 6)
		pl.axvline(sqrt_beta_arr[i], label=r'$a^2=\beta^2$')
		pl.axvline(-sqrt_beta_arr[i])

		pl.xlabel('Re[A]')
		pl.ylabel('Re[B]')
		pl.legend()
		pl.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
		pl.savefig('out/temp.png')
		pl.clf()
		images.append(imageio.imread('out/temp.png'))
	imageio.mimsave('out/chen_gif3.gif', images, fps=4)
