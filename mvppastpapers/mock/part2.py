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



def update(phi_grid, D_const, dt, dx, k, rho, v0):

    phi_grid += D_const * (dt/(dx**2)) * (np.roll(phi_grid, 1, axis = 0) + np.roll(phi_grid, -1, axis = 0) + np.roll(phi_grid, 1, axis = 1) + np.roll(phi_grid, -1, axis = 1) - 4*phi_grid) + rho - k*phi_grid
    
    grad_phix = (np.roll(phi_grid, 1, axis = 0) - np.roll(phi_grid, -1, axis = 0))/2*dx
    length = np.linspace(0, Lgrid-1, Lgrid)
    
    y_grid = np.tile(length, (Lgrid, 1))

    velocity_term = -v0 * np.sin(2 * np.pi * y_grid / Lgrid)

    print (grad_phix)

    return phi_grid - velocity_term * grad_phix



phi_0 = 0.5
Lgrid = 50 

phi_grid = np.random.uniform( phi_0 - 0.1, phi_0 + 0.1, (Lgrid, Lgrid) )

fig = plt.figure()
im = plt.imshow(phi_grid, animated=True, cmap = 'gnuplot')
plt.colorbar()

D_const = 1
dt = 0.1
dx = 1
k = abs(float(input('k: ')))
sigma = abs(float(input('sigma: ')))
v0 = float(input('V0: '))

nsteps = 3000 

rho, r = get_rho(Lgrid, sigma)


steps = np.arange(nsteps)
phi_avg = []

for n in tqdm(steps):

    phi_avg.append(np.sum(phi_grid)/(Lgrid*Lgrid))

    phi_grid = update(phi_grid, D_const, dt, dx, k, rho, v0)

    plt.cla()
    im = plt.imshow(phi_grid, animated=True, cmap = 'gnuplot')
    plt.draw()
    plt.pause(0.00001)

#plt.show()