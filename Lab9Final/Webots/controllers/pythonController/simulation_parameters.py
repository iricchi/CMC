"""Simulation parameters"""
import numpy as np
import matplotlib.pyplot as plt

class SimulationParameters(dict):
    """Simulation parameters"""

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def __init__(self, **kwargs):
        super(SimulationParameters, self).__init__()
        #Set Default parameters
        self.n_body_joints = 10
        self.n_legs_joints = 4
        self.n_oscillators = 24
        self.simulation_duration = 60
        
        self.amplitude_rate = 72
        
        self.phase_lag = 2/3*np.pi/10
        self.freqvaluebody=1
        self.freqvaluelimb=0
        self.amplitude_body_value = 0.
        self.amplitude_leg_value=0.
        
        self.drive_mlr = None
        
        self.turn=0.
        self.backward=False
        self.amplitude_gradient = None
        self.phaseoffset=0.
        self.freqs=(np.ones((self.n_oscillators,1))*2)[:,0]
        
        
        #Update of the paramters
        self.update(kwargs)
        #
        
        vsatbody=0.
        Rsatbody=0.
        cv1body=0.2
        cv0body=0.3
        cr1body=0.065
        cr0body=0.196
        
        vsatlimb=0.
        Rsatlimb=0.
        
        cv1limb=1
        cv0limb=0.
        cr1limb=1
        cr0limb=0.
        
        dlowbody=1.
        dlowlimb=1.
        dhighbody=5.
        dhighlimb=3.
        
        d = self.drive_mlr
        
        if d !=None:
            if d >=dlowbody and d <=dhighbody:
                # v = c1*d + c0
                vref_body = cv1body*d + cv0body
                Rrefbody=cr1body*d+cr0body    
            else:
                vref_body=vsatbody
                Rrefbody=Rsatbody
            
            
            if d >=dlowlimb and d <=dhighlimb:
                # v = c1*d + c0
                vref_limb = cv1limb*d + cv0limb
                Rreflimb=cr1limb*d+cr0limb    
            else:
                vref_limb=vsatlimb
                Rreflimb=Rsatlimb
                
            
            self.freqvaluebody=vref_body
            self.freqvaluelimb=vref_limb
            
            self.amplitude_body_value=Rrefbody
            self.amplitude_leg_value=Rreflimb
            
            self.freqsbody = (np.ones((self.n_oscillators,1))*self.freqvaluebody)[:,0]
            self.freqslimb = (np.ones((4,1))*self.freqvaluelimb)[:,0]
            self.freqs=np.concatenate((self.freqsbody, self.freqslimb), axis=None)
            #print (self.freqs)
            
                    
        self.coupling_weights = np.zeros((self.n_oscillators,self.n_oscillators))
        
        #DEFINITION OF THE COUPLING WEIGHT MATRIX
        self.coupling_weights[0:self.n_body_joints,self.n_body_joints:2*self.n_body_joints] = 10*np.eye(self.n_body_joints)
        self.coupling_weights[self.n_body_joints:2*self.n_body_joints,0:self.n_body_joints] = 10*np.eye(self.n_body_joints) 
        for i in range(self.n_oscillators-4):
            for j in range(1,self.n_oscillators-4):
                if j-i == 1:
                    self.coupling_weights[i,j] = 10
                    self.coupling_weights[j,i] = 10
        self.coupling_weights[9,10]=0
        self.coupling_weights[10,9]=0
        #legs --legs        
        self.coupling_weights[20,21]=10
        self.coupling_weights[21,20]=10
        self.coupling_weights[22,23]=10
        self.coupling_weights[23,22]=10
        self.coupling_weights[20,22]=10
        self.coupling_weights[22,20]=10
        self.coupling_weights[21,23]=10
        self.coupling_weights[23,21]=10
        #legs=>body
        self.coupling_weights[20,0:5]=30
        self.coupling_weights[21,10:15]=30
        self.coupling_weights[22,5:10]=30
        self.coupling_weights[23,15:20]=30
        
        self.coupling_weights=self.coupling_weights.T
        
        #Plot of the matrix
        #plt.matshow(self.coupling_weights)
        #plt.show
        
        #DEFINITION OF THE PHASE BIAS MATRIX
        self.phase_bias = np.zeros((self.n_oscillators,self.n_oscillators))
        for i in range((self.n_oscillators-4)//2):
            self.phase_bias[i,i+10] = np.pi
            self.phase_bias[i+10,i] = -np.pi
            for j in range((self.n_oscillators-4)//2):
                if j-i == 1:
                    self.phase_bias[i,j] = -self.phase_lag
                    self.phase_bias[j,i] = +self.phase_lag
                    self.phase_bias[10+i,10+j] = -self.phase_lag
                    self.phase_bias[10+j,10+i] = +self.phase_lag            
        self.phase_bias[9,10]=0
        self.phase_bias[10,9]=0

        self.phase_bias[20,21]=-np.pi  
        self.phase_bias[21,20]=np.pi
        
        self.phase_bias[20,22]=-np.pi  
        self.phase_bias[22,20]=np.pi   
        
        self.phase_bias[22,23]=-np.pi  
        self.phase_bias[23,22]=np.pi
                   
        self.phase_bias[21,23]=-np.pi  
        self.phase_bias[23,21]=np.pi
        
        #adding the phase offset
        self.phase_bias[20,0:5]=self.phaseoffset
        self.phase_bias[21,10:15]=self.phaseoffset
        self.phase_bias[22,5:10]=self.phaseoffset
        self.phase_bias[23,15:20]=self.phaseoffset
        
        #plot of the matrix
        #plt.matshow(self.phase_bias)
        #plt.show
        
        #To go backward
        if(self.backward):
            self.phase_bias=self.phase_bias.T
        
        if self.amplitude_gradient == None:
            #definition of the amplitude vector
            nominal_amplitude = np.ones(self.n_body_joints*2) * self.amplitude_body_value
            self.nominal_amplitudes=np.concatenate((nominal_amplitude, np.ones(4)*self.amplitude_leg_value), axis=None)    
                
        else:
            #creating one half of the amplitude vector
            nominal_amplitude_left = np.zeros(self.n_body_joints)
            nominal_amplitude_left[0] = self.amplitude_body_value
            for i in range(1,self.n_body_joints):
                nominal_amplitude_left[i] = nominal_amplitude_left[i-1]+ self.amplitude_gradient
            #concatenate with itself to have all the limbs
            nominal_amplitude_body = np.concatenate((nominal_amplitude_left,nominal_amplitude_left), axis=None)
            
            #concatenate with the legs
            self.nominal_amplitudes=np.concatenate((nominal_amplitude_body, np.ones(4)*self.amplitude_leg_value), axis=None)
            #print('nomamp', self.nominal_amplitudes)
            
        
        
