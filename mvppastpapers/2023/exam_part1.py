#import matplotlib
#matplotlib.use('TKAgg')

import numpy as np
import matplotlib.pyplot as plt
from numba import jit 


#Deterministic version
#jit to speed up 
@jit(nopython = True)
def update(grid):
    
    
    #iterate over entire grid
    for i in range(Lgrid):
        for j in range(Lgrid):
            
            #Measure if Rck, paper, 
            state = grid[i, j]

            nbs = [grid[(i+1)%Lgrid, j], grid[(i-1)%Lgrid, j], grid[i, (j+1)%Lgrid], grid[i, (j-1)%Lgrid], grid[(i+1)%Lgrid, (j+1)%Lgrid], grid[(i-1)%Lgrid, (j-1)%Lgrid], grid[(i-1)%Lgrid, (j+1)%Lgrid], grid[(i+1)%Lgrid, (j-1)%Lgrid]]
            nbs_array = np.array(nbs)

            #Rock
            if state == 1:
                no_P = np.count_nonzero( nbs_array == 0 )

                if no_P > 2:
                    #Make paper
                    grid[i, j] = 0
            
            #Paper
            if state == 0:
                no_S = np.count_nonzero( nbs_array == 2 )

                if no_S > 2:
              
                    grid[i, j] = 2
            #Scizors
            if state == 2:
                no_R = np.count_nonzero( nbs_array == 1)

                if no_R > 2:
                    grid[i, j] = 1
    
    return grid





#Grid split into three horizontal strips
def split(Lgrid):
    
    grid = np.zeros((Lgrid, Lgrid) ,dtype = np.uint8)

    choices = np.array([1, 0, 2])
    np.random.shuffle(choices)

    for i in range(Lgrid):
        for j in range(Lgrid):
            if i < Lgrid/3:
                grid[i, j] = choices[0]

            if (i < 2*Lgrid/3) and (i > Lgrid/3):
                grid[i, j] = choices[1]

            if (i < Lgrid) and (i > 2*Lgrid/3):
                grid[i, j] = choices[2]

    return grid

#Random grid with approx. equal of each state
def random(Lgrid):

    grid = np.random.choice(np.array([1, 0, 2]), size = (Lgrid, Lgrid))
    
    return grid




Lgrid = int(input('Lgrid: '))
nsteps = 100

## A grid split into three equal strips or a random grid of each of the three states
#grid = split(Lgrid)
grid = random(Lgrid)

fig = plt.figure()
im = plt.imshow(grid, animated=True, cmap = 'bwr')

steps = np.arange(nsteps)
R_count = []

for n in range(nsteps):

    grid = update(grid)

    #show animation
    plt.cla()
    im = plt.imshow(grid, animated=True, cmap = 'bwr')
    plt.draw()
    plt.pause(0.00001)
    
    #count number of R states
    R_states = np.count_nonzero( grid == 1)
    R_count.append(R_states)
    
plt.show()

plt.plot(steps, R_count, 'b')
plt.xlabel('Step')
plt.ylabel('Number of Rock states')
Plt.show()

    
    