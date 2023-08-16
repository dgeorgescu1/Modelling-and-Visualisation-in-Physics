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





@jit(nopython=True)
def update(grid, Lgrid, p1, p2, p3):

    i = randint(Lgrid)
    j = randint(Lgrid)

    site = grid[i, j]

    #Suseptible
    if site == 0:

        if rand() <= p1:

            neighbours = np.array([ grid[(i+1)%Lgrid, j], grid[(i-1)%Lgrid, j], grid[i, (j+1)%Lgrid], grid[i, (j-1)%Lgrid] ])

            check = neighbours[neighbours.argsort()]

            if check[0] < -0.5:

                grid[i, j] = -int(1)
            
    #Infected
    if site == -1:
        if rand() <= p2:
            grid[i, j] = int(1)

    #Recovered
    if site == 1:
        if rand() <= p3:
            grid[i, j] = int(0)

    return grid    


@jit(nopython = True)
def update_sweep(grid, Lgrid, p1, p2, p3):

    for q in range(Lgrid * Lgrid):

        grid = update(grid, Lgrid, p1, p2, p3)

    return grid


@jit(nopython = True)
def init_grid(Lgrid):

    grid = np.random.choice(np.array([1, 0, -1]), size = (Lgrid, Lgrid))

    return grid 


@jit(nopython = True)
def count_infected(grid):

    count = np.count_nonzero(grid == -1)

    return count 



@jit(nopython = True)
def immunity_grid(grid, N_immune):

    for q in range(N_immune):

        i = randint(Lgrid)
        j = randint(Lgrid)

        site = grid[i, j]
        
        while site == 3:

            i = randint(Lgrid)
            j = randint(Lgrid)

            site = grid[i, j]
        
        grid[i, j] = 3

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













if(len(sys.argv) != 5):
    print ("Usage python SIR.py N p1 p2 p3 (p values just for animation)")
    sys.exit()



Lgrid = int(sys.argv[1]) 
p1 = float(sys.argv[2]) 
p2 = float(sys.argv[3]) 
p3 = float(sys.argv[4]) 




print ('A/H/V/I for Animation or Heatmap or Variance plot or Immunity plot')
run = str(input())

if (run != 'A') and (run != 'H') and (run != 'V') and (run != 'I'):
    print ('Type of run either A/H/V/I')
    sys.exit()








if run == 'A':

    grid = init_grid(Lgrid)
    nsweeps = 1000
    fig = plt.figure()
    im = plt.imshow(grid, animated=True, vmin = -1, vmax = 1, cmap = 'prism')

    for n in range(nsweeps):
        
        grid = update_sweep(grid, Lgrid, p1, p2, p3)

        if(n%1==0):
            plt.cla()
            im = plt.imshow(grid, animated=True, vmin = -1, vmax = 1, cmap = 'prism')
            plt.draw()
            plt.pause(0.08)




if run == 'H':

    nsweeps = 1101
    p2 = 0.5 
    p1s = np.arange(0, 1.05, 0.05)
    p3s = np.arange(0, 1.05, 0.05)

    hmap_vals = np.zeros(shape = ( len(p1s), len(p3s) ))

    for i in tqdm(range(len(p1s))):
        p1 = p1s[i]
        for j in tqdm(range(len(p3s))):
            p3 = p3s[j]
            


            grid = init_grid(Lgrid)
            infected_peeps = []


            for n in range(nsweeps):
                
                grid = update_sweep(grid, Lgrid, p1, p2, p3)

                if n >= 99:

                    count = count_infected(grid)
                    infected_peeps.append(count)

                    if count == 0:
                        break
            
            avg_infected = np.mean(np.array(infected_peeps))
            hmap_vals[j, i] = avg_infected/(Lgrid*Lgrid)

    plt.imshow(hmap_vals, cmap='plasma', origin='lower', extent = [0, 1, 0, 1])
    plt.xlabel('p1')
    plt.ylabel('p3')
    plt.colorbar()
    plt.title('Heatmap')
    plt.savefig('Hmap.png')


    file = open('Heatmap.txt', 'w')
    file.write('p1' + ' ' + 'p3' + ' ' + '<I>/N')
    file.write('\n')
    for i in range(len(p1s)):
        for j in range(len(p3s)):
            file.write(str(p1s[i]) + ' ' + str(p3s[j]) + ' ' + str(hmap_vals[i, j]))
            file.write('\n')
    file.close()


if run == 'V':
    
    nsweeps = 10101
    p2 = 0.5
    p3 = 0.5

    p1s = np.arange(0.2, 0.51, 0.01)
    vars = []
    Errors = []

    for p1 in tqdm(p1s):
        

        grid = init_grid(Lgrid)
        infected_peeps = []


        for n in tqdm(range(nsweeps)):

            grid = update_sweep(grid, Lgrid, p1, p2, p3)

            if n >= 499: 

                count = count_infected(grid)
                infected_peeps.append(count)

                if count == 0:
                        break

        infected_var = np.var(np.array(infected_peeps))
        vars.append(infected_var/(Lgrid*Lgrid))
        Errors.append(bootstrap(infected_peeps)/(Lgrid*Lgrid))

    plt.errorbar(p1s, vars, yerr = Errors, marker = 'x', color = 'blue', linestyle = '')
    plt.xlabel('p1')
    plt.ylabel('Var(I)/N')
    plt.title('Variance plot')
    plt.savefig('VarPlot.png')

    file = open('VariancePlot.txt', 'w')
    file.write('p1' + ' ' + 'Var(Infected)/N>' + ' ' + 'Bootstrap error')
    file.write('\n')
    for i in range(len(p1s)):
        file.write(str(p1s[i]) + ' ' + str(vars[i]) + ' ' + str(Errors[i]))
        file.write('\n')
    file.close()


if run == 'I':

    nsweeps = 10101
    p1 = p2 = p3 = 0.5

    fractions = np.arange(0, 1.01, 0.01)
    avgs = []
    

    for frac in tqdm(fractions):

        N_immune = Lgrid*Lgrid*frac

        grid = init_grid(Lgrid)
        grid = immunity_grid(grid, N_immune)

        infected_peeps = []

        for n in tqdm(range(nsweeps)):
            
            grid = update_sweep(grid, Lgrid, p1, p2, p3)

            if n >= 99:
                
                count = count_infected(grid)
                infected_peeps.append(count)

                if count == 0:
                    break
        
        avg_infected = np.mean(np.array(infected_peeps))
        avgs.append(avg_infected/(Lgrid*Lgrid))
    
    plt.plot(fractions, avgs, 'bx')
    plt.xlabel('Fraction of immune people')
    plt.ylabel('Fraction of <I>')
    plt.savefig('ImmunePlot.png')
        
    file = open('Immunity.txt', 'w')
    file.write('fraction' + ' ' + 'Fraction of <Infected>')
    file.write('\n')
    for i in range(len(fractions)):
        file.write(str(fractions[i]) + ' ' + str(avgs[i]))
        file.write('\n')
    file.close()

    