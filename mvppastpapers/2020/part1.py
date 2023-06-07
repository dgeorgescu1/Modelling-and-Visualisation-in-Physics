import matplotlib
matplotlib.use('TKAgg')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numba import jit 
from tqdm import tqdm
import sys 






#####
#####
@jit(nopython = True)
def randint(Lgrid):
    return np.random.randint(0, Lgrid)

@jit(nopython = True)
def rand():
    return np.random.random()
#####
#####








def update(grid, Lgrid, p):

    i = randint(Lgrid)
    j = randint(Lgrid)

    site = grid[i, j]

    #Inactive (Healthy)
    if site == 0: None

    #Active (Infected)
    else: 

        if rand() <= 1 - p:

            grid[i, j] = 0

        else:

            indices = [((i+1)%Lgrid, j), ((i-1)%Lgrid, j), (i, (j+1)%Lgrid), (i, (j-1)%Lgrid)]
            nb_choice = np.random.choice([0, 1, 2, 3])
            nb_indices = indices[nb_choice]

            grid[nb_indices[0], nb_indices[1]] = 1

    return grid    




def update_sweep(grid, Lgrid, p):

    for q in range(Lgrid * Lgrid):

        grid = update(grid, Lgrid, p)

    return grid





@jit(nopython = True)
def init_grid(Lgrid):

    grid = np.random.choice(np.array([1, 0]), size = (Lgrid, Lgrid))

    return grid 




@jit(nopython=True)
def bootstrap(vals):

    boot_vars = []
    vals_a = np.array(vals)
    for i in range (1000):
        bsample = np.random.choice(vals_a, size = len(vals))
        bvar = np.var(bsample)
        boot_vars.append(bvar)
    
    err = np.std(np.array(boot_vars))

    return err





Lgrid = int(input('Size of Grid: '))  

print ('A/F/S for Animation or Fraction plot or survival')
run = str(input())

if (run != 'A') and (run != 'F') and (run != 'S'):
    print ('Type of run either A/F/S')
    sys.exit()


if run == 'A':

    p = float(input('P: '))

    grid = init_grid(Lgrid)
    nsweeps = 400
    fig = plt.figure()
    im = plt.imshow(grid, animated=True, vmin = -1, vmax = 1, cmap = 'prism')

    steps = np.arange(nsweeps)
    fractions = []

    for n in tqdm(range(nsweeps)):
        
        grid = update_sweep(grid, Lgrid, p)

        if(n%1==0):
            plt.cla()
            im = plt.imshow(grid, animated=True, vmin = -1, vmax = 1, cmap = 'prism')
            plt.draw()
            plt.pause(0.08)
        
        active_count = np.count_nonzero(grid)

        fraction = active_count/ (Lgrid**2)
        fractions.append(fraction)

    plt.show()

    plt.plot(steps, fractions, 'b')
    plt.show()



if run == 'F':

    nsweeps = 2001 
    ps = np.arange(0.55, 0.705, 0.005)
    avg_fractions = []
    variances = []
    Errors = []

    for p in tqdm(ps):
        
        active_counts = []

        grid = init_grid(Lgrid)

        for n in range(nsweeps):

            grid = update_sweep(grid, Lgrid, p)

            a_count = np.count_nonzero(grid)

            active_counts.append(a_count)

        if a_count == 0:
            avg_fractions.append(0)
            variances.append(0)
            Errors.append(0)
        else:

            avg_fractions.append((np.array(active_counts).mean())/(Lgrid**2))
            variances.append((np.array(active_counts).var())/(Lgrid**2))
            Errors.append(bootstrap(active_counts)/(Lgrid**2))
        
    plt.plot(ps, avg_fractions, 'b')
    plt.show()

    plt.errorbar(ps, variances, yerr = Errors, marker = 'x', color = 'red', linestyle = '')
    plt.show()

if run == 'S':

    p = float(input('P: '))

    

    nsweeps = 300
    steps = np.arange(nsweeps)

    num_sim = 1000

    sp = np.zeros((nsweeps, num_sim))

    for sim in tqdm(range(num_sim)):
        
        grid = np.zeros((Lgrid, Lgrid))
        i = randint(Lgrid)
        j = randint(Lgrid)
        grid[i, j] = 1

        for swp in range(nsweeps):
            a_count = np.count_nonzero(grid)
            grid = update_sweep(grid, Lgrid, p)

            #a_count = np.count_nonzero(grid)

            if a_count != 0:
                sp[swp, sim] = 1

    means = np.mean(sp, axis = 1)

    plt.plot(np.log(steps), np.log(means), 'b')
    plt.show()
                


            



        


    











