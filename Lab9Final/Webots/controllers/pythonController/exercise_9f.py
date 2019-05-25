"""Exercise 9f"""

import numpy as np
from run_simulation import run_simulation
from simulation_parameters import SimulationParameters
import plot_results


def exercise_9f(world, timestep, reset):
    """Exercise 9f"""
    # Parameters
    n_joints = 10
    
    
    Phase_offset=np.linspace(-3,3, 10)
    amp=np.linspace(0.3, 0.3, 2)
    nbsimu=Phase_offset.size*amp.size
    
    listuple=[]
    for i in range(Phase_offset.size):
        for j in range(len(amp)):
            listuple.append((Phase_offset[i], amp[j]))
            
            
    nbsimu=Phase_offset.size*amp.size
    
    parameter_set = [SimulationParameters(
            simulation_duration=15,
            drive_mlr=1,
            phaseoffset=phaseof,
            amplitude_value=ampli,
            phase_lag=2*np.pi/10,
        )for phaseof, ampli in listuple]

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
    plot_results.main(nbsimu,Phase_offset, amp, plot=True)

