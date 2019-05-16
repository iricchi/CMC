"""Plot results"""

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from cmc_robot import ExperimentLogger
from save_figures import save_figures
from parse_args import save_plots
import scipy.integrate as integrate


def plot_positions(times, link_data):
    """Plot positions"""
    for i, data in enumerate(link_data.T):
        if i==0:
            plt.plot(times, data, label=["x", "y", "z"][i])
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Distance [m]")
    plt.grid(True)

def plot_trajectory(link_data):
    """Plot positions"""
    plt.plot(link_data[:, 0], link_data[:, 2])
    plt.xlabel("x [m]")
    plt.ylabel("z [m]")
    plt.axis("equal")
    plt.grid(True)
    

def plot_spine_angle(times,link_data):
    plt.figure('Spine angle')
    for i in range(10):
        plt.plot(times,link_data[:,i,0]+i*0.3)
    plt.title('Spine angles')
    plt.xlabel('time[s]')
    plt.ylabel('angle')
    
    plt.figure('Spine angle legs')
    for i in range(4):
        plt.plot(times,link_data[:,10+i,0]+i*0.3)
    plt.title('Spine angles legs')
    plt.xlabel('time[s]')
    plt.ylabel('angle')
    
  

 
 
def plot_energy(listamplitude, listphaselag, energymat):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X, Y = np.meshgrid(listamplitude, listphaselag)
    surf = ax.plot_surface(X, Y, energymat.T, cmap=cm.coolwarm,linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel('Phase offset')
    ax.set_ylabel('Amplitude')
    ax.set_zlabel('Energy')
    
    
    
def plot_vitesse(listamplitude, listphaselag, vitessemat):
    fig = plt.figure()
    #ax=plt.axes(projection='3d')
    #ax.plot_trisurf(listamplitude, listphaselag,vitessemat, cmap='viridis', edgecolor='none')
    
    
    ax = fig.gca(projection='3d')
    X, Y = np.meshgrid(listamplitude, listphaselag)
    surf = ax.plot_surface(X, Y, vitessemat.T, cmap=cm.coolwarm,linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel('Phase offset')
    ax.set_ylabel('Amplitude')
    ax.set_zlabel('Velocity')     
 
 
def compute_energy(joints_data, times):
    velocity = joints_data[:,:,1]
    torque = joints_data[:,:,3]
    power = velocity*torque
    energy = np.zeros(power.shape[1])
    for i in range(power.shape[1]):
        y = np.absolute(power[:,i])
        energy[i] = integrate.trapz(y,times) 
    return(sum(energy))

def compute_velocity(times, link_data):
    vitesse=(link_data[-1]-link_data[int(len(link_data)/2)])/(times[-1]-times[int(len(times)/2)])
    return (np.linalg.norm(vitesse))




def plot_2d(results, labels, n_data=300, log=False, cmap=None):
    """Plot result

    results - The results are given as a 2d array of dimensions [N, 3].

    labels - The labels should be a list of three string for the xlabel, the
    ylabel and zlabel (in that order).

    n_data - Represents the number of points used along x and y to draw the plot

    log - Set log to True for logarithmic scale.

    cmap - You can set the color palette with cmap. For example,
    set cmap='nipy_spectral' for high constrast results.

    """
    xnew = np.linspace(min(results[:, 0]), max(results[:, 0]), n_data)
    ynew = np.linspace(min(results[:, 1]), max(results[:, 1]), n_data)
    grid_x, grid_y = np.meshgrid(xnew, ynew)
    results_interp = griddata(
        (results[:, 0], results[:, 1]), results[:, 2],
        (grid_x, grid_y),
        method='linear'  # nearest, cubic
    )
    extent = (
        min(xnew), max(xnew),
        min(ynew), max(ynew)
    )
    plt.plot(results[:, 0], results[:, 1], "r.")
    imgplot = plt.imshow(
        results_interp,
        extent=extent,
        aspect='auto',
        origin='lower',
        interpolation="none",
        norm=LogNorm() if log else None
    )
    if cmap is not None:
        imgplot.set_cmap(cmap)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    cbar = plt.colorbar()
    cbar.set_label(labels[2])


def main(nbsimu, listamplitude, listphaselag, plot=True):
    """Main"""
    
    energymat=np.zeros((listamplitude.size, listphaselag.size))
    vitessemat=np.zeros((listamplitude.size, listphaselag.size))
    
    for i in range(nbsimu):
        indexamplitude=int(i//listphaselag.size)
        indexphaselag=int(i%listphaselag.size)
        
        # Load data
    with np.load('logs/9b/simulation_0.npz') as data:
        timestep = float(data["timestep"])
        amplitude = data["amplitude_value"]
        phase_lag = data["phase_lag"]
        link_data = data["links"][:, 0, :]
        joints_data = data["joints"]
        
    times = np.arange(0, timestep*np.shape(link_data)[0], timestep)
        
    energymat[indexamplitude, indexphaselag]=compute_energy(joints_data, times)
    vitessemat[indexamplitude, indexphaselag]=compute_velocity(times, link_data)
        
    #print(compute_velocity(times, link_data))
    
    # Plot data
    plt.figure("Positions")
    plot_positions(times, link_data)
        
    plt.figure("Trajectory")
    plot_trajectory(link_data)
    
        
    # Plot energy
    plt.figure('Energy')
    plot_energy( listamplitude, listphaselag, energymat)
    
    #Plot vitesse
    plt.figure('Vitesse')
    plot_vitesse( listamplitude, listphaselag, vitessemat)
    
    plot_spine_angle(times,joints_data)
    # Show plots
    if plot:
        plt.show()
    else:
        save_figures()

    if __name__ == '__main__':
        main(plot=not save_plots())

