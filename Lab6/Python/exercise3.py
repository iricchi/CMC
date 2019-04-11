""" Lab 6 Exercise 3
This file implements the pendulum system with two muscles attached driven
by a neural network

To obtain figure 15, 



"""

""" 
-------------------------------------------------------------------------------------------------------
HOW TO OBTAIN THE FIGURES IN THE REPORT
-------------------------------------------------------------------------------------------------------
To obtain figures 13 and 14, just run the code
To obtain figure 15, run the code a first time, then without closing the plot, 
change the 0. by 1. in the lines 134 to 137 and run it again.

To obtain figure 16, change the 0. in the previous lines by 0.2, 0., 0.2, 0.

To obtain Figure 17, change the 0.2 by 0.7 in the previous instruction
"""

import numpy as np
from matplotlib import pyplot as plt
import math
import cmc_pylog as pylog
from cmcpack import DEFAULT
from cmcpack.plot import save_figure
from muscle import Muscle
from muscle_system import MuscleSytem
from neural_system import NeuralSystem
from pendulum_system import PendulumSystem
from system import System
from system_animation import SystemAnimation
from system_parameters import (MuscleParameters, NetworkParameters,
                               PendulumParameters)
from system_simulation import SystemSimulation

# Global settings for plotting
# You may change as per your requirement
plt.rc('lines', linewidth=2.0)
plt.rc('font', size=12.0)
plt.rc('axes', titlesize=14.0)     # fontsize of the axes title
plt.rc('axes', labelsize=14.0)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=14.0)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14.0)    # fontsize of the tick labels


def exercise3():
    """ Main function to run for Exercise 3.

    Parameters
    ----------
        None

    Returns
    -------
        None
    """
    # Define and Setup your pendulum model here
    # Check Pendulum.py for more details on Pendulum class
    P_params = PendulumParameters()  # Instantiate pendulum parameters
    P_params.L = 0.5  # To change the default length of the pendulum
    P_params.m = 1.  # To change the default mass of the pendulum
    pendulum = PendulumSystem(P_params)  # Instantiate Pendulum object

    #### CHECK OUT Pendulum.py to ADD PERTURBATIONS TO THE MODEL #####

    pylog.info('Pendulum model initialized \n {}'.format(
        pendulum.parameters.showParameters()))

    # Define and Setup your pendulum model here
    # Check MuscleSytem.py for more details on MuscleSytem class
    M1_param = MuscleParameters()  # Instantiate Muscle 1 parameters
    M1_param.f_max = 1500  # To change Muscle 1 max force
    M2_param = MuscleParameters()  # Instantiate Muscle 2 parameters
    M2_param.f_max = 1500  # To change Muscle 2 max force
    M1 = Muscle(M1_param)  # Instantiate Muscle 1 object
    M2 = Muscle(M2_param)  # Instantiate Muscle 2 object
    # Use the MuscleSystem Class to define your muscles in the system
    muscles = MuscleSytem(M1, M2)  # Instantiate Muscle System with two muscles
    pylog.info('Muscle system initialized \n {} \n {}'.format(
        M1.parameters.showParameters(),
        M2.parameters.showParameters()))

    # Define Muscle Attachment points
    m1_origin = np.array([-0.17, 0.0])  # Origin of Muscle 1
    m1_insertion = np.array([0.0, -0.17])  # Insertion of Muscle 1

    m2_origin = np.array([0.17, 0.0])  # Origin of Muscle 2
    m2_insertion = np.array([0.0, -0.17])  # Insertion of Muscle 2

    # Attach the muscles
    muscles.attach(np.array([m1_origin, m1_insertion]),
                   np.array([m2_origin, m2_insertion]))

    ##### Neural Network #####
    # The network consists of four neurons
    N_params = NetworkParameters()  # Instantiate default network parameters
    #N_params.D = 2.  # To change a network parameter
    N_params.D = 1.
    # Similarly to change w -> N_params.w = (4x4) array
    
    N_params.w=[[0.,-5.,-5.,0.],
                [-5.,0.,0.,-5.],
                [5.,-5.,0.,0.],
                [-5.,5.,0.,0.]]
    
    
    # Create a new neural network with above parameters
    neural_network = NeuralSystem(N_params)
    
    pylog.info('Neural system initialized \n {}'.format(
        N_params.showParameters()))

    # Create system of Pendulum, Muscles and neural network using SystemClass
    # Check System.py for more details on System class
    sys = System()  # Instantiate a new system
    sys.add_pendulum_system(pendulum)  # Add the pendulum model to the system
    sys.add_muscle_system(muscles)  # Add the muscle model to the system
    # Add the neural network to the system
    sys.add_neural_system(neural_network)
    
    

    ##### Time #####
    t_max = 10  # Maximum simulation time
    time = np.arange(0., t_max, 0.001)  # Time vector
    
    
    
    
    externalinput=np.ones((len(time), 4))
    #PLAY WITH EXTERNAL INPOUT HERE (excitation of only 1 neuron for example)

    externalinput[:, 0]=0.*externalinput[:,0]
    externalinput[:, 1]=0.*externalinput[:,1]
    externalinput[:, 2]=0.*externalinput[:,2]
    externalinput[:, 3]=0.*externalinput[:,3]
    
    

    ##### Model Initial Conditions #####
    x0_P = np.array([0., 0.])  # Pendulum initial condition

    # Muscle Model initial condition
    x0_M = np.array([0., M1.L_OPT, 0., M2.L_OPT])

    x0_N = np.array([-0.5, 1, 0.5, 1])  # Neural Network Initial Conditions

    x0 = np.concatenate((x0_P, x0_M, x0_N))  # System initial conditions

    ##### System Simulation #####
    # For more details on System Simulation check SystemSimulation.py
    # SystemSimulation is used to initialize the system and integrate
    # over time

    sim = SystemSimulation(sys)  # Instantiate Simulation object
    
    
    # Add external inputs to neural network
    #sim.add_external_inputs_to_network(externalinput)
    #-------------------------------------------------------------------------
    
    sim.add_external_inputs_to_network(externalinput)
    
    # sim.add_external_inputs_to_network(ext_in)

    sim.initalize_system(x0, time)  # Initialize the system state

    # Integrate the system for the above initialized state and time
    sim.simulate()

    # Obtain the states of the system after integration
    # res is np.array [time, states]
    # states vector is in the same order as x0
    res = sim.results()

    # Obtain the states of the system after integration
    # res is np.array [time, states]
    # states vector is in the same order as x0
    #res = sim.results()
    
    # In order to obtain internal states of the muscle
    # you can access the results attribute in the muscle class
    muscle1_results = sim.sys.muscle_sys.Muscle1.results
    muscle2_results = sim.sys.muscle_sys.Muscle2.results
    
    
    #print (np.size(muscle1_results))
    #plt.figure('activation')
    #plt.plot(muscle1_results)
    #☻plt.plot(muscle2_results)

    plt.figure('output neural network')
    plt.title('output of the neural network')
    #plt.plot(res[:, 0], res[:, :2])
    plt.plot(res[:, 0], res[:, -1])
    plt.plot(res[:, 0], res[:, -2])
    plt.xlabel('time (s)')
    plt.ylabel('amplitude (.)')
    #plt.legend(['external input=0.0','external input=1.0'])
    

    # Plotting the results
    plt.figure('Pendulum')
    plt.title('Pendulum Phase')
    #plt.plot(res[:, 0], res[:, :2])
    plt.plot(res[:, 1], res[:, 2])
    #plt.plot(res[:, 1], res[:, 2])
    #plt.plot(res[:, 0])
    #plt.plot(res[:, :2])
    plt.xlabel('Position [rad]')
    plt.ylabel('Velocity [rad.s]')
    plt.grid()

    if DEFAULT["save_figures"] is False:
        plt.show()
    else:
        figures = plt.get_figlabels()
        pylog.debug("Saving figures:\n{}".format(figures))
        for fig in figures:
            plt.figure(fig)
            save_figure(fig)
            plt.close(fig)

    # To animate the model, use the SystemAnimation class
    # Pass the res(states) and systems you wish to animate
    simulation = SystemAnimation(
        res,
        sim.sys.pendulum_sys,
        sim.sys.muscle_sys,
        sim.sys.neural_sys)
    # To start the animation
    simulation.animate()


if __name__ == '__main__':
    from cmcpack import parse_args
    parse_args()
    exercise3()

