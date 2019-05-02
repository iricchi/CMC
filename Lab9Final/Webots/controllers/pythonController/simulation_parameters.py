"""Simulation parameters"""
import numpy as np

class SimulationParameters(dict):
    """Simulation parameters"""

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def __init__(self, **kwargs):
        super(SimulationParameters, self).__init__()
        # Default parameters
        self.n_body_joints = 10
        self.n_legs_joints = 0
        self.n_oscillators = 20
        
        self.simulation_duration = 30
        self.phase_lag = 0.5
        self.amplitude_gradient = None
        self.amplitude_value = 1
        self.coupling_weights = np.zeros((self.n_oscillators,self.n_oscillators))
        self.freqs = np.ones((self.n_oscillators,1))*2
        # walking
        # self.freqs[:] = 0.8
        
        for i in range(self.n_oscillators):
            for j in range(1,self.n_oscillators):
                if j-i == 1:
                    self.coupling_weights[i,j] = 10
                    self.coupling_weights[j,i] = 10
                    
        
        self.coupling_weights[0:self.n_body_joints,self.n_body_joints:2*self.n_body_joints] = 10*np.eye(self.n_body_joints)
        self.coupling_weights[self.n_body_joints:2*self.n_body_joints,0:self.n_body_joints] = 10*np.eye(self.n_body_joints) 
        
        self.phase_bias = np.zeros((self.n_oscillators,self.n_oscillators))
        for i in range(self.n_oscillators):
            for j in range(1,self.n_oscillators-1):
                if j-i == 1:
                    self.phase_bias[i,j] = self.phase_lag
                    self.phase_bias[j,i] = -self.phase_lag
        
        if self.amplitude_gradient == None:
            self.nominal_amplitudes = np.ones((self.n_oscillators,1)) * self.amplitude_value
        else:
            nominal_amplitude = np.zeros((self.n_body_joints,1))
            nominal_amplitude[0] = self.amplitude_value
            for i in range(1,self.n_body_joints):
                nominal_amplitude[i] = nominal_amplitude[i-1]+ self.amplitude_gradient
             
            self.nominal_amplitudes = np.vstack((nominal_amplitude,nominal_amplitude))
        
        self.amplitude_rate = np.ones((self.n_oscillators,1))*10
        # Feel free to add more parameters (ex: MLR drive)
        # self.drive_mlr = ...
        # ...
        # Update object with provided keyword arguments
        self.update(kwargs)  # NOTE: This overrides the previous declarations

