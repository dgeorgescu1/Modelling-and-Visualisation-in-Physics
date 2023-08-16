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










def update(grid):

    kernel = np.array([[1, 1, 1], 
                   [1, 0, 1],
                   [1, 1, 1]], dtype = int)

    neighbours = convolve(grid, kernel, mode = 'wrap')

    grid = (
            ((grid == 1) & (neighbours > 1) & (neighbours < 4))
            | ((grid == 0) & (neighbours == 3))
           ).astype(int)
    
    return grid 










#Random grid
@jit(nopython = True)
def random(Lgrid):
    
    grid = np.zeros((Lgrid, Lgrid),dtype = np.uint8)

    for i in range(Lgrid):
        for j in range(Lgrid):
            r = np.random.random()
            if (r < 0.5): grid[i,j] = 0
            if (r >= 0.5): grid[i,j] = 1

    return grid

#Glider on grid
@jit(nopython = True )
def glider(center, grid):

    grid[0 + center, 0 + center] = 1
    grid[1 + center, 0 + center] = 1
    grid[2 + center, 0 + center] = 1
    grid[2 + center, 1 + center] = 1
    grid[1 + center, 2 + center] = 1

    return grid

#Oscillator on grid
@jit(nopython = True )
def oscillator(center, Lgrid):

    grid = np.zeros((Lgrid, Lgrid),dtype = np.uint8)

    for i in range(3):
        for j in range(3):
            grid[i + center, j + center] = 1
            grid[-1 - i + center, -1 - j + center] = 1

    return grid


@jit(nopython = True)
def act_sites(grid):
    
    return np.sum(grid)

    







nstep = 3000
Lgrid = 50



type = input('Plot (P) or Animation (A) or Glider Speed (G): ')
if (type != 'P') and (type != 'A') and (type != 'G'):
    print ('P/A/G for plot or animation or glider speed')
    sys.exit()



if type == 'A':

    init = input('Random (R), Glider (G), Oscillator (O): ')
    if (init != 'R') and (init != 'G') and (init != 'O'):
        print ('R / G / O')
        sys.exit() 


    if init == 'R':

        grid = random(Lgrid)

    elif init == 'G':
        
        grid = np.zeros((Lgrid, Lgrid),dtype = np.uint8)

        grid = glider(10, grid)
        grid = glider(15, grid)
        grid = glider(20, grid)
        grid = glider(25, grid)
        grid = glider(30, grid)
        grid = glider(35, grid)
        grid = glider(40, grid)

    elif init == 'O':

        center = 30

        grid = oscillator(center, Lgrid)

    fig = plt.figure()
    im = plt.imshow(grid, animated=True, cmap = 'binary')

    for n in range(nstep):

        grid = update(grid)

    #show animation
        plt.cla()
        im = plt.imshow(grid, animated=True, cmap = 'binary')
        plt.draw()
        plt.pause(0.00001)

if type == 'P':

    equil_steps = []

    for i in tqdm(range(1000)):

        c = 0
        n = 0

        grid = random(Lgrid)
        sites_i = act_sites(grid)

        while (c <= 9) and n < 10000:
            
            grid = update(grid)
            sites_f = act_sites(grid)

            n += 1

            if sites_i == sites_f: c += 1
            if sites_i != sites_f: c = 0

            sites_i = sites_f

        if c >= 10: equil_steps.append(n - 10)
    
    plt.hist(equil_steps, bins = 50, facecolor = 'magenta', edgecolor = 'black', linewidth = 0.5)

    plt.xlabel('Steps to equilibrium')
    plt.ylabel('Counts')
    plt.title('Game of Life')
    plt.savefig('GOLHist.png')

    file = open('GOLHist.txt', 'w')
    file.write('Equilsteps')
    file.write('\n')
    for i in range(len(equil_steps)):
        file.write(str(equil_steps[i]))
        file.write('\n')
    file.close()

if type == 'G':

    xs = []
    ys = []

    grid = np.zeros((Lgrid, Lgrid),dtype = np.uint8)
    grid = glider(25, grid)

    for n in tqdm(range(299)):

        y, x = grid.nonzero()
        
        difx = abs(x.min() - x.max())
        dify = abs(y.min() - y.max())
        
        if (difx >= Lgrid/2) or (dify >= Lgrid/2):
           
            grid = update(grid)

        else:

            COM = np.array([x.sum(), y.sum()])/len(x)

            xs.append(COM[0])
            ys.append(COM[1])

            grid = update(grid)

    def linear(x, a, b):
        return a*x + b
    
    poptx, pcovx = curve_fit(linear, np.arange(100, 250, 1), xs[100:250])
    popty, pcovy = curve_fit(linear, np.arange(100, 250, 1), ys[100:250])

    print (poptx)
    print (popty)

    fig, ax = plt.subplots(2)
    ax[0].plot(xs, 'bx')
    ax[0].plot(np.arange(100, 250, 1), linear(np.arange(100, 250, 1), *poptx), 'black')
    ax[1].plot(ys, 'rx')
    ax[1].plot(np.arange(100, 250, 1), linear(np.arange(100, 250, 1), *popty), 'black')
    plt.savefig('Gliderspeed.png')

    file = open('Gliderspeed.txt', 'w')
    file.write('step' + ' ' + 'x' + ' ' + 'y')
    file.write('\n')
    for i in range(len(xs)):
        file.write(str(i) + ' ' + str(xs[i]) + ' ' + str(ys[i]))
        file.write('\n')
    file.write('x speed: ' + str(poptx[0]) + 'y speed: ' + str(popty[0]))
    file.write('\n')
    file.write(str((np.sqrt(poptx[0]**2 + popty[0]**2))))
    file.close()
