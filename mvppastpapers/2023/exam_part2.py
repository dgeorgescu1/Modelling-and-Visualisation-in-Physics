#import matplotlib
#matplotlib.use('TKAgg')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage import convolve
from numba import jit 
from tqdm import tqdm
import sys 
from scipy.optimize import curve_fit

#####
#####
@jit(nopython = True)
def rand():
    return np.random.random()

#####
#####
@jit(nopython = True)
def randint(Lgrid):
    return np.random.randint(0, Lgrid)

@jit(nopython = True)
def update(grid, Lgrid, p1, p2, p3):
    
    #random and sequential update rule. Only difference now is that also need to meet probability 
    #condition to chage state
    
    for i in range(Lgrid):
        for j in range(Lgrid):

            i = randint(Lgrid)
            j = randint(Lgrid)

            state = grid[i, j]
            
            nbs = [grid[(i+1)%Lgrid, j], grid[(i-1)%Lgrid, j], grid[i, (j+1)%Lgrid], grid[i, (j-1)%Lgrid], grid[(i+1)%Lgrid, (j+1)%Lgrid], grid[(i-1)%Lgrid, (j-1)%Lgrid], grid[(i-1)%Lgrid, (j+1)%Lgrid], grid[(i+1)%Lgrid, (j-1)%Lgrid]]
            nbs_array = np.array(nbs)

            #Rock
            if state == 1:
                no_P = np.count_nonzero( nbs_array == 0 )

                if (no_P > 0) and (rand() <= p1):
                    #Make paper
                    grid[i, j] = 0
            
            if state == 0:
                no_S = np.count_nonzero( nbs_array == 2 )

                if (no_S > 0) and (rand() <= p2):
                    grid[i, j] = 2
            
            if state == 2:
                no_R = np.count_nonzero( nbs_array == 1)

                if (no_R > 0) and (rand() <= p3):
                    grid[i, j] = 1
    
    return grid






#Creates random grid
def random(Lgrid):

    grid = np.random.choice(np.array([1, 0, 2]), size = (Lgrid, Lgrid))
    
    return grid




Lgrid = int(input('Lgrid: '))

print ('A/P/H for Animation or Plots or heatmap')
run = str(input())

if (run != 'A') and (run != 'P') and (run != 'H'):
    print ('Type of run either A/P/H')
    sys.exit()






if run == 'A':
    
    grid = random(Lgrid)
    
    nsteps = 100
    
    fig = plt.figure()
    im = plt.imshow(grid, animated=True, cmap = 'bwr', vmin = -1, vmax = 2)
    
    p1 = float(input('p1: '))
    p2 = float(input('p2: '))
    p3 = float(input('p3: '))


    for n in range(nsteps):
    
        grid = update(grid, Lgrid, p1, p2, p3)
    
        #show animation
        plt.cla()
        im = plt.imshow(grid, animated=True, cmap = 'bwr', vmin = -1, vmax = 2)
        plt.draw()
        plt.pause(0.00001)
        
    
if run == 'P':
    
    p1 = 0.5
    p2 = 0.5
    
    nsteps = 10001
    
    p3s = np.arange(0, 0.1005, 0.005)
    
    avg_fractions = []
    Variances = []
    
    for p3 in tqdm(p3s):
        
        fractions_R = []
        fractions_P = []
        fractions_S = []
        
        grid = random(Lgrid)
        
        for n in range(nsteps):
            
            grid = update(grid, Lgrid, p1, p2, p3)
            
            count_R = np.count_nonzero( grid == 1)
            count_P = np.count_nonzero( grid == 0)
            count_S = np.count_nonzero( grid == 2)
    
            fractions_R.append(count_R)
            fractions_P.append(count_P)
            fractions_S.append(count_S)
        
        #Averages
        avg_R = np.mean(np.array(fractions_R))
        avg_P = np.mean(np.array(fractions_P))
        avg_S = np.mean(np.array(fractions_S))
        
        if (avg_R < avg_P) and (avg_R < avg_S):
            avg_fractions.append(avg_R/ Lgrid**2)
            Variances.append(np.var(np.array(fractions_R))/ Lgrid**2)
        
        if (avg_P < avg_R) and (avg_P < avg_S):
            avg_fractions.append(avg_P/ Lgrid**2)
            Variances.append(np.var(np.array(fractions_P))/ Lgrid**2)
        
        if (avg_S < avg_P) and (avg_S < avg_R):
            avg_fractions.append(avg_S/ Lgrid**2)
            Variances.append(np.var(np.array(fractions_S))/ Lgrid**2)
    
    plt.plot(p3s, avg_fractions, 'bx')
    plt.xlabel('p3 probability')
    plt.ylabel('average fraction of minority phase')
    plt.show()
    
    plt.plot(p3s, Variances, 'bx')
    plt.xlabel('p3 probability')
    plt.ylabel('Variance of fraction of minority phase')
    plt.show()
            
if run == 'H':
   
    nsteps = 2001
    
    p1 = 0.5
    p2s = np.arange(0, 0.302, 0.02)
    p3s = np.arange(0, 0.302, 0.02)
    
    hmap_vals = np.zeros(shape = ( len(p2s), len(p3s) ))
    
    for i in tqdm(range(len(p2s))):
        p2 = p2s[i]
        for j in range(len(p3s)):
            p3 = p3s[j]
            
            grid = random(Lgrid)
            fractions_R = []
            fractions_P = []
            fractions_S = []
            
            for n in range(nsteps):
                
                grid = update(grid, Lgrid, p1, p2, p3)
                
                count_R = np.count_nonzero( grid == 1)
                count_P = np.count_nonzero( grid == 0)
                count_S = np.count_nonzero( grid == 2)
        
                fractions_R.append(count_R)
                fractions_P.append(count_P)
                fractions_S.append(count_S)
                
            avg_R = np.mean(np.array(fractions_R))
            avg_P = np.mean(np.array(fractions_P))
            avg_S = np.mean(np.array(fractions_S))
            
            if (avg_R < avg_P) and (avg_R < avg_S):
                hmap_vals[j, i] = avg_R/(Lgrid*Lgrid)
            
            if (avg_P < avg_R) and (avg_P < avg_S):
                hmap_vals[j, i] = avg_P/(Lgrid*Lgrid)
            
            if (avg_S < avg_P) and (avg_S < avg_R):
                hmap_vals[j, i] = avg_S/(Lgrid*Lgrid)
    
    plt.imshow(hmap_vals, cmap='plasma', origin='lower', extent = [0, 0.3, 0, 0.3])
    plt.xlabel('p2')
    plt.ylabel('p3')
    plt.colorbar()
    plt.title('Heatmap')
    plt.show()
                
                
    
    
    
