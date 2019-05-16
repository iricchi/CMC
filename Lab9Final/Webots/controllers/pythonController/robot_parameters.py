"""Robot parameters"""

import numpy as np
import cmc_pylog as pylog


class RobotParameters(dict):
    """Robot parameters"""

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def __init__(self, parameters):
        super(RobotParameters, self).__init__()

        # Initialise parameters
        self.n_body_joints = parameters.n_body_joints
        self.n_legs_joints = parameters.n_legs_joints
        self.n_joints = self.n_body_joints + self.n_legs_joints
        self.n_oscillators_body = 2*self.n_body_joints
        self.n_oscillators_legs = self.n_legs_joints
        self.n_oscillators = self.n_oscillators_body + self.n_oscillators_legs
        self.drive_mlr=parameters.drive_mlr
        self.freqs = parameters.freqs
        
        self.turn=parameters.turn
        self.backward=parameters.backward
        
        self.phaseoffset=parameters.phaseoffset
        
        self.coupling_weights = np.zeros([self.n_oscillators,self.n_oscillators])
        self.phase_bias = np.zeros([self.n_oscillators, self.n_oscillators])
        self.rates = parameters.amplitude_rate
        
        self.nominal_amplitudes = np.zeros(self.n_oscillators)
        
        self.amplitude_leg_nominal=parameters.amplitude_leg_value
        self.update(parameters)
        
        
        
        

    def update(self, parameters):
        """Update network from parameters"""
        
        self.set_frequencies(parameters)  # f_i
        self.set_amplitudes_rate(parameters)  # a_i
        self.set_coupling_weights(parameters)  # w_ij
        self.set_phase_bias(parameters)  # theta_i
        self.set_nominal_amplitudes(parameters)  # R_i
       
        
    def set_frequencies(self, parameters):
        """Set frequencies"""   
        self.freqs = parameters.freqs
            
        
            
    def set_nominal_amplitudes(self, parameters):
        """Set nominal amplitudes"""
        self.nominal_amplitudes = parameters.nominal_amplitudes
            
            
    def set_coupling_weights(self, parameters):
        """Set coupling weights"""
        # Set coupling weights
        self.coupling_weights = parameters.coupling_weights
      

    def set_phase_bias(self, parameters):
        """Set phase bias"""
        self.phase_bias = parameters.phase_bias
                               
    def set_amplitudes_rate(self, parameters):
        """Set amplitude rates"""
        self.rates = parameters.amplitude_rate
        
        #self.rates = parameters.rates
        
    
