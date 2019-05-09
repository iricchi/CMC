"""Exercise 9b"""

import numpy as np
from run_simulation import run_simulation
from simulation_parameters import SimulationParameters
import plot_results

def exercise_9b(world, timestep, reset):
    """Exercise example"""
    # Parameters
    n_joints = 10
    parameter_set = [SimulationParameters(
            simulation_duration=1,
            #drive=drive,
            amplitude=amplitude,
            phase_lag=2/3*np.pi/10,
            #
            turn=0,
            # ...
        )for amplitude, phase_lag in zip([0.1, 0.2, 0.3, 0.4],[2/3*np.pi/10])  
    ]

    # Grid search
   
    for simulation_i, parameters in enumerate(parameter_set):
        
        reset.reset()
        run_simulation(
            world,
            parameters,
            timestep,
            int(1000*parameters.simulation_duration/timestep),
            logs="./logs/example/simulation_{}.npz".format(simulation_i)     
        )
    plot_results.main(plot=True)
        
   

