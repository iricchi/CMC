""" Lab 6 Exercise 2
This file implements the pendulum system with two muscles attached
"""
""" 
-------------------------------------------------------------------------------------------------------
                              HOW TO OBTAIN THE FIGURES IN THE REPORT
-------------------------------------------------------------------------------------------------------
 
To obtain the plots of Figure4, Figure 5 and Figure6 in the report, 
run "Computeandplotmusclelength".

To see the activation waves (figure 7), uncomment lines 180-186
To obtain the figure 11, just run the code as it is, or change the activation between square and sin.

To obtain the figures 8, 9 and 10, modify lines 191 to modify the activation, and 206 to put or not a perturbation

"""
import math
from math import sqrt
import cmc_pylog as pylog
import numpy as np
from matplotlib import pyplot as plt
import scipy
from scipy import signal
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
plt.rc('axes', labelsize=14.0)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=14.0)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14.0)    # fontsize of the tick labels
DEFAULT["save_figures"] = True


def exercise2():
    """ Main function to run for Exercise 2.

    Parameters
    ----------
        None

    Returns
    -------
        None
    """

    # Define and Setup your pendulum model here
    # Check PendulumSystem.py for more details on Pendulum class
    pendulum_params = PendulumParameters()  # Instantiate pendulum parameters
    pendulum_params.L = 0.5  # To change the default length of the pendulum
    pendulum_params.m = 1.  # To change the default mass of the pendulum
    pendulum = PendulumSystem(pendulum_params)  # Instantiate Pendulum object

    #### CHECK OUT PendulumSystem.py to ADD PERTURBATIONS TO THE MODEL #####

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

    # Create a system with Pendulum and Muscles using the System Class
    # Check System.py for more details on System class
    sys = System()  # Instantiate a new system
    sys.add_pendulum_system(pendulum)  # Add the pendulum model to the system
    sys.add_muscle_system(muscles)  # Add the muscle model to the system

    ##### Time #####
    t_max = 3  # Maximum simulation time
    time = np.arange(0., t_max, 0.001)  # Time vector

    ##### Model Initial Conditions #####
    x0_P = np.array([0., 0.])  # Pendulum initial condition

    # Muscle Model initial condition
    x0_M = np.array([0., M1.L_OPT, 0., M2.L_OPT])

    x0 = np.concatenate((x0_P, x0_M))  # System initial conditions

    ##### System Simulation #####
    # For more details on System Simulation check SystemSimulation.py
    # SystemSimulation is used to initialize the system and integrate
    # over time

    simsin = SystemSimulation(sys)  # Instantiate Simulation object
    
    #simsquare = SystemSimulation(sys)
    
    # Add muscle activations to the simulation
    # Here you can define your muscle activation vectors
    # that are time dependent
        
    label_test=[]
    
    
    """" definition of different kinds of activation for each muscle.
    Amplitude1 and amplitude2 allows to play with the amplitude of activation on each muscle (RMS value for the sinus activation)
    
    act1 and act2 activates the muscle all the time.
    actsin activates with sin(wi) if sin(wi)>0 (no negative activation). The 2 muscles are in opposition of phase.
    actsquare does the same with a square signal.
    
    
    
    """
    
    
    amplitude1=1.
    amplitude2=1.
    
    #declaration of the activations
    act1 = np.ones((len(time), 1))*amplitude1
    act2 = np.ones((len(time), 1))*amplitude2
    actsin = np.ones((len(time), 1))
    actsin2 = np.ones((len(time), 1))
    actsquare = np.ones((len(time), 1))
    actsquare2 = np.ones((len(time), 1))
    
    wlist=[0.1,0.05, 0.01, 0.005]
    
    k=0
    
    
    for w in wlist:
        #generation of the signals at pulsation w
        for i in range(len(actsin)):  
            if math.sin(w*i)<=0:
                actsin[i]=0
                actsin2[i]=abs(amplitude2*math.sqrt(2)*math.sin(w*i))
            else:
                actsin[i]=abs(amplitude1*math.sqrt(2)*math.sin(w*i))
                actsin2[i]=0
            
        for i in range(len(actsquare)):
            
            if i%(2*math.pi/w)<=math.pi/w:
                actsquare[i]=amplitude1
                actsquare2[i]=0
            else:
                actsquare[i]=0
                actsquare2[i]=amplitude2
 

        """ uncomment this to plot the activation signals"""               
#        #Plot of the activation through time    
#        plt.figure
#        plt.plot(actsquare)
#        plt.plot(actsin)
#        plt.title("Activations wave forms used")
#        plt.xlabel("Time (s)")
#        plt.ylabel("Activation amplitude (.)")
    
    
    
        """ put as parameters the activation you want (act1/2, actsin1/2 or actsquare1/2)"""
        activationssin = np.hstack((actsquare, actsquare2))
        #activationssquare = np.hstack((actsquare, actsquare2))

        # Method to add the muscle activations to the simulation

        simsin.add_muscle_activations(activationssin)
        #simsquare.add_muscle_activations(activationssquare)
        # Simulate the system for given time

        simsin.initalize_system(x0, time)  # Initialize the system state
        #simsquare.initalize_system(x0, time)
        #: If you would like to perturb the pedulum model then you could do
        # so by
        
        """perturbation of the signal"""
        simsin.sys.pendulum_sys.parameters.PERTURBATION = False
        #simsquare.sys.pendulum_sys.parameters.PERTURBATION = True
        # The above line sets the state of the pendulum model to zeros between
        # time interval 1.2 < t < 1.25. You can change this and the type of
        # perturbation in
        # pendulum_system.py::pendulum_system function

        # Integrate the system for the above initialized state and time
        simsin.simulate()
        #simsquare.simulate()
        # Obtain the states of the system after integration
        # res is np.array [time, states]
        # states vector is in the same order as x0
        ressin = simsin.results()
        #ressquare = simsquare.results()

        # In order to obtain internal states of the muscle
        # you can access the results attribute in the muscle class
        muscle1_results = simsin.sys.muscle_sys.Muscle1.results
        muscle2_results = simsin.sys.muscle_sys.Muscle2.results

        # Plotting the results
        plt.figure('Pendulum')
        plt.title('Pendulum Phase')
        plt.plot(ressin[:, 1], ressin[:, 2])
        label_test.append('w='+str(wlist[k]))
        k=k+1
        #plt.plot(ressquare[:, 1], ressquare[:, 2])
        plt.xlabel('Position [rad]')
        plt.ylabel('Velocity [rad.s]')
        plt.legend(label_test)
        plt.grid()

        # To animate the model, use the SystemAnimation class
        # Pass the res(states) and systems you wish to animate
        simulationsin = SystemAnimation(ressin, pendulum, muscles)
        #simulationsquare = SystemAnimation(ressquare, pendulum, muscles)
    
        # To start the animation
        if DEFAULT["save_figures"] is False:
            simulationsin.animate()
        #simulationsquare.animate()
        if not DEFAULT["save_figures"]:
            plt.show()
        else:
            figures = plt.get_figlabels()
            pylog.debug("Saving figures:\n{}".format(figures))
            for fig in figures:
                plt.figure(fig)
                save_figure(fig)
                plt.close(fig)
                    
if __name__ == '__main__':
    from cmcpack import parse_args
    parse_args()
    exercise2()
    

