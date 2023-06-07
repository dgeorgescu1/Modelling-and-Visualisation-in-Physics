# Importing packages
import matplotlib
matplotlib.use('TKAgg')

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numba import jit 
from tqdm import tqdm









###
### Speed things up a bit
@jit(nopython=True)
def randint(lx):
    return np.random.randint(0, lx)

@jit(nopython=True)
def rand():
    return np.random.random()
###
###









@jit(nopython=True)
def Glauber(lx, spin, kT):

    """

    Glauber updating rule using metropolis algorithm. Flip spin depending on energy from NNs

    Input: lx (int, length of spin grid), spin (np.array, grid of spins), kT (float, temp. constant)
    Return: Updated grid of spins (np.array)

    """

    #Selecting random point on grid
    i = randint(lx)
    j = randint(lx)
    spin_old = spin[i,j]

    #Calculate energy from NNs
    d_E = 2 * spin_old * ( spin[(i+1)%lx, j] + spin[(i-1)%lx, j] + spin[i, (j+1)%lx] + spin[i, (j-1)%lx] )

    #Update depending on value of energy 
    if d_E <= 0:

        spin[i, j] = -spin_old

    elif rand() <= np.exp(-d_E/kT):

        spin[i, j] = -spin_old
    
    return spin



@jit(nopython=True)
def Kawasaki(lx, spin, kT):

    """

    Kawasaki updating rule. Select two different but random spins on grid and swap depending on their NN energies.
    Energy correction if two selected spins NNs

    Input: lx (int, length of spin grid), spin (np.array, grid of spins), kT (float, temp. constant)
    Return: Updated grid of spins (np.array)

    """

    #Select one random grid point
    i1 = randint(lx)
    j1 = randint(lx)
    spin1 = spin[i1, j1]

    (i2, j2) = (i1, j1)
    
    #Ensures spins are not the exact same point
    while (i2, j2) == (i1, j1):

        i2 = randint(lx)
        j2 = randint(lx)
    
    spin2 = spin[i2, j2]

    #Doesnt bother with calculation if both points have same spin
    if spin1 == spin2: None

    else:

        #NN calculation
        nbs1 = ( spin[(i1+1)%lx, j1] + spin[(i1-1)%lx, j1] + spin[i1, (j1+1)%lx] + spin[i1, (j1-1)%lx])
        nbs2 = ( spin[(i2+1)%lx, j2] + spin[(i2-1)%lx, j2] + spin[i2, (j2+1)%lx] + spin[i2, (j2-1)%lx])

        d_E = 2 * spin1 * (nbs1 - nbs2)

        #Nearest Neighbour correction
        diff = np.array([(i1-i2), (j1-j2)], dtype = 'float')
        
        #Nearest Neighbour correction
        if np.linalg.norm(diff) == 1: d_E += 4

        #Nearest Neighbour correction
        elif np.linalg.norm(diff) == (lx-1): 
            if ((i1-i2) == 0) or ((j1-j2) == 0):
                d_E += 4

        #Swap spins depending on energies
        if d_E <= 0:

            spin[i1, j1] = -spin1
            spin[i2, j2] = -spin2

        elif rand() <= np.exp(-d_E/kT):

            spin[i1, j1] = -spin1
            spin[i2, j2] = -spin2
    
    return spin







def InitAnim(lx, ly):

    spin = np.zeros((lx,ly), dtype=float)
    fig = plt.figure()
    im = plt.imshow(spin, animated=True, cmap = 'cool')

    for i in range(lx):
        for j in range(ly):
            r = rand()
            if(r<0.5): spin[i,j] = -1
            if(r>=0.5): spin[i,j] = 1
    
    return spin, fig, im


@jit(nopython=True)
def Energy(spin, lx, J):

    E = 0
    for i in range(lx):
        for j in range(lx):
            E += - J * spin[i, j] * ( spin[(i+1)%lx, j] + spin[(i-1)%lx, j] + spin[i, (j+1)%lx] + spin[i, (j-1)%lx] )

    return E/2


@jit(nopython=True)
def bootstrap(vals, factor):

    boot_vars = []
    vals_a = np.array(vals)
    for i in range (1000):
        bsample = np.random.choice(vals_a, size = len(vals))
        bvar = np.var(bsample)
        boot_vars.append(bvar)
    
    err = np.std(np.array(boot_vars))

    return err/ factor





J=1.0
nstep = 10001

if(len(sys.argv) != 3):
    print ("Usage python ising.animation.py N T (T just for animation)")
    sys.exit()

lx=int(sys.argv[1]) 
ly=lx 
kT=float(sys.argv[2]) 

print ('K/G for Kawasaki or Glauber')
method = str(input())

if (method != 'K') and (method != 'G'):
    print ('Method K/G')
    sys.exit()

print ('A/T for Animation or Temperature analysis')
run = str(input())

if (run != 'A') and (run != 'T'):
    print ('Type of run either A/T')
    sys.exit()













if run == 'A':

    spin, fig, im = InitAnim(lx, ly)

    for n in range(nstep):
        for q in range(ly*ly):
            
            #Kawasaki method
            if method == 'K':
                spin = Kawasaki(lx, spin, kT)

            #Glauber
            if method == 'G':
                spin = Glauber(lx, spin, kT)
                    
    #occasionally plot or update measurements, eg every 10 sweeps
        if(n%10==0) and (run == 'A'): 
    #       update measurements
    #       dump output
            f=open('spinskaw.dat','w')
            for i in range(lx):
                for j in range(ly):
                    f.write('%d %d %lf\n'%(i,j,spin[i,j]))
            f.close()
    #       show animation
            plt.cla()
            im=plt.imshow(spin, animated=True, vmin = -1, vmax = 1, cmap = 'winter')
            plt.draw()
            plt.pause(0.0001)


elif run == 'T':
    
    if method == 'G':
        spin = np.ones((lx, ly), dtype=float)

    if method == 'K':
        spin = np.ones((lx, ly), dtype=float)
        split = int(lx/2)
        spin[split:, :] = -1
    
    Temps = np.arange(1, 3.1, 0.1)

    AvgM = []
    Sus = []

    AvgE = []
    Spe = []

    SusErr = []
    SpeErr = []

    for kT in tqdm(Temps):

        Ms = []
        Ms2 = []
        Es = []

        for n in tqdm(range(nstep)):
            for q in range(ly*ly):

               #Kawasaki method
                if method == 'K':
                    spin = Kawasaki(lx, spin, kT)

                #Glauber
                if method == 'G':
                    spin = Glauber(lx, spin, kT)

            if (n >= 99) and (n%10 == 0):
                
                M = np.sum(spin)
                Ms.append(abs(M))

                Ms2.append(M)

                Es.append(Energy(spin, lx, J))


        if method == 'G':

            AvgM.append(np.mean( np.array(Ms) ))
            VarM = np.var( np.array(Ms2) )
            Sus.append(VarM / (kT*lx*lx))
            SusErr.append(bootstrap(Ms2, kT*lx*lx))

        AvgE.append(np.mean( np.array(Es) ))
        VarE = np.var( np.array(Es) )
        Spe.append(VarE / (kT*kT*lx*lx))
        SpeErr.append(bootstrap(Es, kT*kT*lx*lx))


    if method == 'G':

        fig, ax = plt.subplots(2, 2)

        ax[0, 0].plot(Temps, AvgM, 'bx')
        ax[0, 0].set_title('Magnetization')
        ax[0, 0].set_xlabel('T')
        ax[0, 0].set_ylabel('<|M|>')
        ax[1, 0].errorbar(Temps, Sus, yerr = SusErr, marker = 'x', color = 'blue', linestyle = '')
        ax[1, 0].set_title('Susceptibility')
        ax[1, 0].set_xlabel('T')
        ax[1, 0].set_ylabel('X')

        ax[0, 1].plot(Temps, AvgE, 'bx')
        ax[0, 1].set_title('Energy')
        ax[0, 1].set_xlabel('T')
        ax[0, 1].set_ylabel('<E>')
        ax[1, 1].errorbar(Temps, Spe, yerr = SpeErr, marker = 'x', color = 'blue', linestyle = '')
        ax[1, 1].set_title('Heat Capacity')
        ax[1, 1].set_xlabel('T')
        ax[1, 1].set_ylabel('Cv')
        fig.tight_layout()
        fig.savefig('GlauberPlots.png')

        file = open('Glauber.txt', 'w')
        file.write('T' + ' ' + '<|M|>' + ' ' + '<E>' + ' ' + 'X' + ' '  + 'XErr' + ' '  + 'Cv' + ' '  + 'CvErr')
        file.write('\n')
        for i in range(len(Temps)):
            file.write(str(Temps[i]) + ' ' + str(AvgM[i]) + ' ' + str(AvgE[i]) + ' ' + str(Sus[i]) + ' ' + str(SusErr[i]) + ' ' + str(Spe[i]) + ' ' + str(SpeErr[i]))
            file.write('\n')
        file.close()


    elif method == 'K':

        fig, ax = plt.subplots(2)
        
        ax[0].plot(Temps, AvgE, 'bx')
        ax[0].set_title('Energy')
        ax[0].set_xlabel('T')
        ax[0].set_ylabel('<E>')
        ax[1].errorbar(Temps, Spe, yerr = SpeErr, marker = 'x', color = 'blue', linestyle = '')
        ax[1].set_title('Heat Capacity')
        ax[1].set_xlabel('T')
        ax[1].set_ylabel('Cv')
        fig.tight_layout()
        fig.savefig('KawasakiPlots.png')

        file = open('Kawasaki.txt', 'w')
        file.write('T' + ' '  + '<E>' + ' ' + 'Cv' + ' '  + 'CvErr')
        file.write('\n')
        for i in range(len(Temps)):
            file.write(str(Temps[i]) + ' ' + str(AvgE[i]) + ' ' + str(Spe[i]) + ' ' + str(SpeErr[i]))
            file.write('\n')
        file.close()

    plt.show()


                