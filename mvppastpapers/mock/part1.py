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

def linear(x, a, b):
    return a*x + b

def get_rho(Lgrid, sigma):

    length = np.linspace(0, Lgrid-1, Lgrid)
    grid = np.meshgrid(length, length, indexing = 'ij')

    coords = np.array(grid).T.reshape(Lgrid, Lgrid, 2)    
    r = np.linalg.norm(coords - Lgrid//2, axis = 2)
    rho = np.exp(-(r/sigma)**2)

    return rho, r



def update(phi_grid, D_const, dt, dx, k, rho):

    phi_grid += D_const * (dt/(dx**2)) * (np.roll(phi_grid, 1, axis = 0) + np.roll(phi_grid, -1, axis = 0) + np.roll(phi_grid, 1, axis = 1) + np.roll(phi_grid, -1, axis = 1) - 4*phi_grid) + rho - k*phi_grid

    return phi_grid



phi_0 = 0.5
Lgrid = 50 

phi_grid = np.random.uniform( phi_0 - 0.1, phi_0 + 0.1, (Lgrid, Lgrid) )

#fig = plt.figure()
#im = plt.imshow(phi_grid, vmax = 1, vmin = -1, animated=True, cmap = 'plasma')
#plt.colorbar()

D_const = 1
dt = 0.1
dx = 1
k = abs(float(input('k: ')))
sigma = abs(float(input('sigma: ')))


nsteps = 3000 

rho, r = get_rho(Lgrid, sigma)


steps = np.arange(nsteps)
phi_avg = []

for n in tqdm(steps):

    phi_avg.append(np.sum(phi_grid)/(Lgrid*Lgrid))

    phi_grid = update(phi_grid, D_const, dt, dx, k, rho)

    #plt.cla()
    #im = plt.imshow(phi_grid, vmax = 1, vmin = -1, animated=True, cmap = 'plasma')
    #plt.draw()
    #plt.pause(0.00001)

#plt.show()

plt.plot(steps, phi_avg)
plt.show()

x, y = np.log(r.flatten()), np.log(phi_grid.flatten())
sorted = x.argsort()

popt, pcov = curve_fit(linear, x[sorted][1:70], y[sorted][1:70])

plt.plot(x, y, 'r')
plt.plot(x[sorted][0:70], linear(x[sorted][0:70], *popt), 'b')
plt.text(0.5, -1, 'y = {:.4f} x + {:.4f}'.format(*popt), fontsize = 15)
plt.show()