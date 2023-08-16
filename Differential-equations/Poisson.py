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













##### Makes 3D True/False checkerboard
def checkerboard(N):

    size = (N+2, N+2, N+2)
    checker = (np.indices(size).sum(axis=0) % 2).astype(np.bool)
    
    return checker, ~checker


##### Jacobian updater 
def update_jacob(phi, rho, field_type):

    old_phi = np.copy(phi)

    update_phi = (1/6)*(

                np.roll(phi, 1, axis = 0) +
                np.roll(phi, -1, axis = 0) +
                np.roll(phi, 1, axis = 1) +
                np.roll(phi, -1, axis = 1) +
                np.roll(phi, 1, axis = 2) +
                np.roll(phi, -1, axis = 2) +
                rho

                )
    
    if field_type == 'E':

        middle = update_phi[1:-1, 1:-1, 1:-1]
        padder = ((1, 1), (1, 1), (1, 1))

    if field_type == 'M':

        middle = update_phi[1:-1, 1:-1, :]
        padder = ((1, 1), (1, 1), (0, 0))
        
    phi = np.pad(middle, padder)
    error = np.sum(np.abs(phi - old_phi))

    return phi, error


#### Gauss_Seidel (with SOR) method
def Gauss_Seidel(phi, rho, black, white, field_type, omega):

    old_phi = np.copy(phi)

    phi[black] = (1/6)*(

                np.roll(phi, 1, axis = 0) +
                np.roll(phi, -1, axis = 0) +
                np.roll(phi, 1, axis = 1) +
                np.roll(phi, -1, axis = 1) +
                np.roll(phi, 1, axis = 2) +
                np.roll(phi, -1, axis = 2) +
                rho

                )[black]
    
    if field_type == 'E':

        middle = phi[1:-1, 1:-1, 1:-1]
        padder = ((1, 1), (1, 1), (1, 1))

    if field_type == 'M':

        middle = phi[1:-1, 1:-1, :]
        padder = ((1, 1), (1, 1), (0, 0))
        
    phi = np.pad(middle, padder)

    phi[black] = (1 - omega) * old_phi[black] + omega * phi[black]

    phi[white] = (1/6)*(

                np.roll(phi, 1, axis = 0) +
                np.roll(phi, -1, axis = 0) +
                np.roll(phi, 1, axis = 1) +
                np.roll(phi, -1, axis = 1) +
                np.roll(phi, 1, axis = 2) +
                np.roll(phi, -1, axis = 2) +
                rho

                )[white]
    
    if field_type == 'E':

        middle = phi[1:-1, 1:-1, 1:-1]
        padder = ((1, 1), (1, 1), (1, 1))

    if field_type == 'M':

        middle = phi[1:-1, 1:-1, :]
        padder = ((1, 1), (1, 1), (0, 0))
        
    phi = np.pad(middle, padder)

    phi[white] = (1 - omega) * old_phi[white] + omega * phi[white]

    error = np.sum(np.abs(phi - old_phi))

    return phi, error


#### Calculates the vector field (magnetic or electric)
def Field(phi, dx):

    F_x = -(np.roll(phi, 1, axis = 0) - np.roll(phi, -1, axis = 0)) / (2*dx)
    F_y = -(np.roll(phi, 1, axis = 1) - np.roll(phi, -1, axis = 1)) / (2*dx)
    F_z = -(np.roll(phi, 1, axis = 2) - np.roll(phi, -1, axis = 2)) / (2*dx)

    return F_x, F_y, F_z

#For fitting
def linear(x, a, b):
        return a*x + b






dx = 1
N = int(input('Integer length of grid: '))

Method = input('Jacobian (J) or Gauss-Seidel (G) or Succesive over-relaxation method (S): ')
if (Method != 'J') and (Method != 'G') and (Method != 'S'):
    print ('J/G for Jacobian or Gauss-seidel update methods or S for SOR calculation')
    sys.exit()

field_type = input('Electric (E) or Magnetic (M): ')
if (field_type != 'E') and (field_type != 'M'):
    print ('E/M for electric or magnetic field calculation')
    sys.exit()





phi = np.zeros((int(N+2), int(N+2), int(N+2)))
rho = np.zeros((int(N+2), int(N+2), int(N+2)))
black, white = checkerboard(N)

if field_type == 'E':
    rho[(N+2)//2, (N+2)//2, (N+2)//2] = 1

if field_type == 'M':
    rho[(N+2)//2, (N+2)//2, :] = 1



if (Method == 'J') or (Method == 'G'):

    n=0
    error = 90

    while error > 1e-3: 
        n+=1

        if Method == 'J':
            phi, error = update_jacob(phi, rho, field_type)

        if Method == 'G':
            phi, error = Gauss_Seidel(phi, rho, black, white, field_type, 1)

    print (n)


if Method == 'S':

    no_sweeps = []
    omega_list = np.arange(1.1, 2, 0.02)
    for omega in tqdm(omega_list):

        n=0
        error = 90

        while error > 1e-3: 
            n+=1

            phi, error = Gauss_Seidel(phi, rho, black, white, field_type, omega)

        phi = np.zeros((int(N+2), int(N+2), int(N+2)))
        no_sweeps.append(n)

    omega_min = omega_list[np.argmin(no_sweeps)]


    plt.plot(omega_list, no_sweeps, 'mx')
    plt.xlabel('Omega')
    plt.ylabel('# Sweeps to reach 1e-3 accuracy')
    plt.text(1.2, 200, 'min: omega = {:.2f}'.format(omega_min), fontsize = 12)
    plt.title('{} field'.format(field_type))
    plt.savefig('omega_min.png')
    plt.show()


if (field_type == 'E') and (Method != 'S'):


    middle = phi[1:-1, 1:-1, 1:-1]

    #Contour plot
    plt.imshow(middle[:, :, N//2], cmap='gnuplot')
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('E potential contour plot')
    plt.savefig('E_potential_contour.png')
    plt.show()



    # Potential vs dist from charge
    length = np.linspace(0, N-1, N)
    grid = np.meshgrid(length, length, indexing = 'ij')
    coords = np.array(grid).T.reshape(-1, 2)

    dist_from_centre = np.linalg.norm(coords - N//2, axis = 1)

    x, y = np.log(dist_from_centre), np.log(middle[:, :, N//2].flatten())
    plt.plot(x, y, 'bx')
    plt.xlabel('log distance from central charge')
    plt.ylabel('log E potential strength')

    sorted = x.argsort()
    popt, pcov = curve_fit(linear, x[sorted][5:200], y[sorted][5:200])
    plt.plot(x[sorted][5:200], linear(x[sorted][5:200], *popt), 'r')
    plt.text(1, -10, 'y = {:.4f} x + {:.4f}'.format(*popt), fontsize = 15)
    plt.savefig('E_pot_vs_distfromcharge.png')
    plt.show()


    #E field lines
    Ex, Ey, Ez = Field(middle, dx)

    Ex_slice = Ex[:, :, N//2].flatten()
    Ey_slice = Ey[:, :, N//2].flatten()
    Ez_slice = Ez[:, :, N//2].flatten()

    Mag_E = np.sqrt(Ex_slice**2 + Ey_slice**2 + Ez_slice**2)
    y2 = np.log(Mag_E)

    #E field strength vs dist from charge
    plt.plot(x, y2, 'bx')
    plt.xlabel('log distance from central charge')
    plt.ylabel('log E field strength')

    popt2, pcov2 = curve_fit(linear, x[sorted][4:300], y2[sorted][4:300])
    plt.plot(x[sorted][4:300], linear(x[sorted][4:300], *popt2), 'r')
    plt.text(1, -10, 'y = {:.4f} x + {:.4f}'.format(*popt2), fontsize = 15)

    plt.savefig('E_Fieldstrength_vs_distfromcharge.png')
    plt.show()

    #E field vector plot
    plt.quiver(coords.T[1], coords.T[0], -Ex_slice/Mag_E, -Ey_slice/Mag_E, angles = 'xy', scale_units = 'xy', scale = 1)
    plt.title('E field vector plot')
    plt.savefig('E_field_vector_plot.png')
    plt.show()


    #Save datafiles
    np.savetxt('E_pot_contour.txt', np.column_stack((coords, dist_from_centre, x, Ex_slice, Ey_slice, Ez_slice)), delimiter = ' ', header = '(x y) dist_from_charge phi Ex Ey Ez')

    




if (field_type == 'M') and (Method != 'S'):

    middle = phi[1:-1, 1:-1, 1:-1]

    #Contour 
    plt.imshow(middle[:, :, (N//2)])
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('M potential contour plot')
    plt.savefig('M_potential_contour.png')
    plt.show()

    length = np.linspace(0, N-1, N)
    coords = np.array(np.meshgrid(length, length, indexing = 'xy')).T.reshape(-1, 2)

    dist_from_centre = np.linalg.norm(coords - N//2, axis = 1)

    #M potential vs dist from charge
    x, y = np.log(dist_from_centre), np.log(middle[:, :, N//2].flatten())
    plt.plot(x, y, 'rx')
    plt.xlabel('log distance from central charge')
    plt.ylabel('log M potential strength')

    sorted = x.argsort()
    popt, pcov = curve_fit(linear, x[sorted][5:200], y[sorted][5:200])
    plt.plot(x[sorted][5:200], linear(x[sorted][5:200], *popt), 'b')
    plt.text(1, -7, 'y = {:.4f} x + {:.4f}'.format(*popt), fontsize = 15)
    plt.savefig('M_pot_vs_distfromcharge.png')
    plt.show()

    Mx, My, Mz = Field(middle, dx)

    Mx_slice = Mx[:, :, N//2].flatten()
    My_slice = My[:, :, N//2].flatten()

    Mag_M = np.sqrt(Mx_slice**2 + My_slice**2)
    y2 = np.log(Mag_M)

    #E field strength vs dist from charge
    plt.plot(x, y2, 'rx')
    plt.xlabel('log distance from central charge')
    plt.ylabel('log M field strength')

    popt2, pcov2 = curve_fit(linear, x[sorted][4:300], y2[sorted][4:300])
    plt.plot(x[sorted][4:300], linear(x[sorted][4:300], *popt2), 'b')
    plt.text(1, -7, 'y = {:.4f} x + {:.4f}'.format(*popt2), fontsize = 15)

    plt.savefig('M_Fieldstrength_vs_distfromcharge.png')
    plt.show()

    plt.quiver(coords.T[0], coords.T[1], My_slice/Mag_M, -Mx_slice/Mag_M, angles = 'xy', scale_units = 'xy', scale = 1)
    plt.title('M field vector plot')
    plt.savefig('M_field_vector_plot.png')
    plt.show()

    np.savetxt('M_pot_contour.txt', np.column_stack((coords, dist_from_centre, x, Mx_slice, My_slice)), delimiter = ' ', header = '(x y) dist_from_charge phi Mx My Mz')

    print (x.shape)
    print (y.shape)




    
