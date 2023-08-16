import matplotlib
matplotlib.use('TKAgg')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage import convolve
from numba import jit 
from tqdm import tqdm
import sys 
from scipy.optimize import curve_fit





def update(phi_grid, a, k, dx, M, dt):

    mu = -a * phi_grid + a * phi_grid**3 - (k/(dx**2))*( np.roll(phi_grid, 1, axis = 0) + np.roll(phi_grid, -1, axis = 0) + np.roll(phi_grid, 1, axis = 1) + np.roll(phi_grid, -1, axis = 1) - 4*phi_grid )

    phi_grid += M * (dt/(dx**2)) * (np.roll(mu, 1, axis = 0) + np.roll(mu, -1, axis = 0) + np.roll(mu, 1, axis = 1) + np.roll(mu, -1, axis = 1) - 4*mu)

    return phi_grid


def free_energy(a, phi_grid, k, dx):

    grad_phix = (np.roll(phi_grid, 1, axis = 0) - np.roll(phi_grid, -1, axis = 0))/2*dx
    grad_phiy = (np.roll(phi_grid, 1, axis = 1) - np.roll(phi_grid, -1, axis = 1))/2*dx

    fe_grid = -(a/2) * phi_grid**2 + (a/4) * phi_grid**4 + (k/2) * (grad_phix**2 + grad_phiy**2)

    return np.sum(fe_grid) 







phi_0 = float(input('Phi_0: '))
Lgrid = int(input('Length of grid: '))

phi_grid = np.random.uniform( phi_0 - 0.1, phi_0 + 0.1, (Lgrid, Lgrid) )

#fig = plt.figure()
#im = plt.imshow(phi_grid, animated=True, cmap = 'RdBu_r')
#plt.colorbar()

a = 0.1
k = 0.1
M = 0.1

dt = 2
dx = 1

nsteps = 1000001
#nsteps = 100000
FEs = []


for n in tqdm(range(nsteps)):

    fe = free_energy(a, phi_grid, k, dx)
    FEs.append(fe)

    phi_grid = update(phi_grid, a, k, dx, M, dt)


    


    #if (n%500 == 0):

        #plt.cla()
        #im = plt.imshow(phi_grid, animated=True, cmap = 'RdBu_r')
        #plt.draw()
        #plt.pause(0.00001)
        
plt.clf()
plt.plot(FEs, color = 'lime')
plt.xlabel('# Steps')
plt.ylabel('Free energy')
plt.savefig('{}_Feplot.png'.format(phi_0))
plt.show()


file = open('{}_Fe.txt'.format(phi_0), 'w')
file.write('step' + ' ' + 'Fe')
file.write('\n')
for i in range(len(FEs)):
    file.write(str(i) + ' ' + str(FEs[i]))
    file.write('\n')
file.close()