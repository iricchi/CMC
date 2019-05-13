"""Exercise 9d"""

import numpy as np
from run_simulation import run_simulation
from simulation_parameters import SimulationParameters
import plot_results


def exercise_9d1(world, timestep, reset):
    """Exercise 9d1"""
    # Parameters
    n_joints = 10
    listuple=[]
    Rtail=np.linspace(0.4, 0.4, 1)
    Rhead=np.linspace(0.4, 0.4, 1)
    for i in range(Rtail.size):
        for j in range(len(Rhead)):
            listuple.append((Rtail[i], Rhead[j]))

    nbsimu=Rhead.size*Rtail.size
    
    parameter_set = [SimulationParameters(
            simulation_duration=20,
            #drive=drive,
            amplitude_gradient=None,
            amplitude=0.3,
            phase_lag=2/3*np.pi/10,
            #
            turn=0.,
            backward=True,
            # ...
        )for grad in listuple]

    # Grid search
   
    for simulation_i, parameters in enumerate(parameter_set):
        print(simulation_i)
        reset.reset()
        run_simulation(
            world,
            parameters,
            timestep,
            int(1000*parameters.simulation_duration/timestep),
            logs="./logs/9b/simulation_{}.npz".format(simulation_i)     
        )
    plot_results.main(nbsimu,Rtail, Rhead, plot=True)


def exercise_9d2(world, timestep, reset):
    """Exercise 9d2"""
    pass

