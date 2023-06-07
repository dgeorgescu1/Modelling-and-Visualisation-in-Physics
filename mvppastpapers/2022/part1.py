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
import matplotlib.colors as colors



def initialise(Lgrid):

    a = np.random.uniform(0, float(1/3), (Lgrid, Lgrid))
    b = np.random.uniform(0, float(1/3), (Lgrid, Lgrid))
    c = np.random.uniform(0, float(1/3), (Lgrid, Lgrid))

    return a, b, c



def update(a, b, c, D, dt, dx, q, p):

    a_new = a + D * (dt/(dx**2)) * (np.roll(a, 1, axis = 0) + np.roll(a, -1, axis = 0) + np.roll(a, 1, axis = 1) + np.roll(a, -1, axis = 1) - 4*a) + q * a * (1 - a - b - c) - p * a * c 
    b_new = b + D * (dt/(dx**2)) * (np.roll(b, 1, axis = 0) + np.roll(b, -1, axis = 0) + np.roll(b, 1, axis = 1) + np.roll(b, -1, axis = 1) - 4*b) + q * b * (1 - a - b - c) - p * a * b
    c_new = c + D * (dt/(dx**2)) * (np.roll(c, 1, axis = 0) + np.roll(c, -1, axis = 0) + np.roll(c, 1, axis = 1) + np.roll(c, -1, axis = 1) - 4*c) + q * c * (1 - a - b - c) - p * b * c 

    return a_new, b_new, c_new




def field_update(field, a, b, c):

    d = 1 - a - b - c
    
    stacked = np.stack(arrays = (d, a, b, c), axis = 2)
    field = np.argmax(stacked, axis = 2)
    #for i in range(Lgrid):
     #   for j in range(Lgrid):

 #           vals = np.array([d[i, j], a[i, j], b[i, j], c[i, j]])
  #          field[i, j] = np.argmax(vals)
    
    return field

             


Lgrid = 50
a, b, c = initialise(Lgrid)
field = np.zeros((Lgrid, Lgrid))



dx = 1
dt = 0.1
D = 1
q = 1
p = 0.5

nsteps = 10000

cmap = colors.ListedColormap(['grey', 'red', 'green', 'blue'])

fig = plt.figure()
im = plt.imshow(field, animated=True, cmap = cmap)
plt.colorbar()

for n in range(nsteps):

    a, b, c = update(a, b, c, D, dt, dx, q, p)

    field = field_update(field, a, b, c)

    if (n%100 == 0):
        plt.cla()
        im = plt.imshow(field, animated=True, cmap = cmap)
        plt.draw()
        plt.pause(0.00001)
