"""Exercise 9b"""

import numpy as np
from run_simulation import run_simulation
from simulation_parameters import SimulationParameters
import plot_results

def exercise_9b(world, timestep, reset):
    """Exercise example"""
    # Parameters
    n_joints = 10
    listamplitude=np.linspace(0.05,0.4, 4)
    listphaselag=np.linspace(0., 0.9, 6)
    listuple=[]
    
    for i in range(len(listamplitude)):
        for j in range(len(listphaselag)):
            listuple.append((listamplitude[i], listphaselag[j]))
            
    nbsimu=len(listamplitude)* len(listphaselag)
    
    parameter_set = [SimulationParameters(
        simulation_duration=15, 
        amplitude_body_value=amp,
        phase_lag=phasel, 
        drive_mlr=None)
        for amp, phasel in listuple]

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
        
   

