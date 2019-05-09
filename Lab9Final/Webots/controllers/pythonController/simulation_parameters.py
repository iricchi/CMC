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
        self.n_legs_joints = 4
        self.n_oscillators = 24
        self.simulation_duration = 60
        self.phase_lag = 2/3*np.pi/10
        self.amplitude_gradient = None
        self.amplitude_value = 0.15
        self.amplitude = 0
        self.amplitude_leg_nominal=np.pi/2
        
        self.coupling_weights = np.zeros((self.n_oscillators,self.n_oscillators))
        self.freqs = (np.ones((self.n_oscillators,1))*2)[:,0] #2Hz
        # walking
        # self.freqs[:] = 0.8
        
        self.coupling_weights[0:self.n_body_joints,self.n_body_joints:2*self.n_body_joints] = 10*np.eye(self.n_body_joints)
        self.coupling_weights[self.n_body_joints:2*self.n_body_joints,0:self.n_body_joints] = 10*np.eye(self.n_body_joints) 
   
        for i in range(self.n_oscillators-4):
            for j in range(1,self.n_oscillators-4):
                if j-i == 1:
                    self.coupling_weights[i,j] = 10
                    self.coupling_weights[j,i] = 10
        self.coupling_weights[20,21]=10
        self.coupling_weights[21,20]=10
        self.coupling_weights[22,23]=10
        self.coupling_weights[23,22]=10
        self.coupling_weights[20,22]=10
        self.coupling_weights[22,20]=10
        self.coupling_weights[21,23]=10
        self.coupling_weights[23,21]=10
        self.coupling_weights[20,0:5]=30
        self.coupling_weights[0:5,20]=30
        self.coupling_weights[21,10:15]=30
        self.coupling_weights[10:15,21]=30
        self.coupling_weights[22,5:10]=30
        self.coupling_weights[5:10,22]=30
        self.coupling_weights[23,15:20]=30
        self.coupling_weights[15:20,23]=30
        
        self.phase_bias = np.zeros((self.n_oscillators,self.n_oscillators))
        for i in range((self.n_oscillators-4)//2):
            self.phase_bias[i,i+10] = np.pi
            self.phase_bias[i+10,i] = np.pi
        
            for j in range((self.n_oscillators-4)//2):
                if j-i == 1:
                    self.phase_bias[i,j] = +self.phase_lag
                    self.phase_bias[j,i] = -self.phase_lag
                    self.phase_bias[10+i,10+j] = -self.phase_lag
                    self.phase_bias[10+j,10+i] = +self.phase_lag
                    
        self.phase_bias[20,21]=np.pi  
        self.phase_bias[21,20]=np.pi
        
        self.phase_bias[20,22]=np.pi  
        self.phase_bias[22,20]=np.pi   
        
        self.phase_bias[22,23]=np.pi  
        self.phase_bias[23,22]=np.pi
                    
        self.phase_bias[21,23]=np.pi  
        self.phase_bias[23,21]=np.pi
        
       
        
        if self.amplitude_gradient == None:
            nominal_amplitude = np.ones(self.n_body_joints*2) * self.amplitude_value
            
            self.nominal_amplitudes=np.concatenate((nominal_amplitude, np.ones(4)*self.amplitude_leg_nominal), axis=None)
            
        else:
            nominal_amplitude = np.zeros(self.n_body_joints)
            nominal_amplitude[0] = self.amplitude_value
            for i in range(1,self.n_body_joints):
                nominal_amplitude[i] = nominal_amplitude[i-1]+ self.amplitude_gradient
            nominal_amplitudes = np.concatenate((nominal_amplitude,nominal_amplitude), axis=None)
            self.nominal_amplitudes=np.concatenate((nominal_amplitudes,np.ones(4)*self.amplitude_leg_nominal), axis=None)
            
        self.amplitude_rate = np.ones((self.n_oscillators,1))*10
        # Feel free to add more parameters (ex: MLR drive)
        # self.drive_mlr = ...
        # ...
        # Update object with provided keyword arguments
        self.update(kwargs)  # NOTE: This overrides the previous declarations

