"""Exercise 9b"""

import numpy as np
from run_simulation import run_simulation
from simulation_parameters import SimulationParameters
import plot_results

def exercise_9b(world, timestep, reset):
    """Exercise example"""
    # Parameters
    n_joints = 10
    listamplitude=np.linspace(0.1,0.5, 6)
    listphaselag=np.linspace(0.1, 2., 7)*np.pi/10
    listuple=[]
    
    for i in range(len(listamplitude)):
        for j in range(len(listphaselag)):
            listuple.append((listamplitude[i], listphaselag[j]))
    nbsimu=len(listamplitude)* len(listphaselag)
    parameter_set = [SimulationParameters(
            simulation_duration=5,
            #drive=drive,
            amplitude=amplitude,
            phase_lag=2/3*np.pi/10,
            #
            turn=0,
            # ...
        )for amplitude, phase_lag in listuple]

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
    plot_results.main(nbsimu,listamplitude, listphaselag, plot=True)
        
   

