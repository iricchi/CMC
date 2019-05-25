"""Oscillator network ODE"""

import numpy as np
from simulation_parameters import SimulationParameters
from solvers import euler, rk4
from robot_parameters import RobotParameters


def network_ode(_time, state, parameters):
    """Network_ODE
    returns derivative of state (phases and amplitudes)
    """
    
    phases = np.array(state[:parameters.n_oscillators])

    amplitudes = np.array(state[parameters.n_oscillators:2*parameters.n_oscillators])
    
    #turn
    amplitudes[0:10] = (1-parameters.turn)*amplitudes[:10]
    amplitudes[10:20]= (1+parameters.turn)*amplitudes[10:20]
    
    #computation of the derivative of the phase of each oscillator  
    dphase = np.zeros(parameters.n_oscillators)
    
    
    for i in range(parameters.n_oscillators):
        sum = 0
        for j in range(parameters.n_oscillators):
            sum += parameters.coupling_weights[i,j]*amplitudes[j]*np.sin(phases[j]-phases[i]-parameters.phase_bias[i,j])
        dphase[i]=2*np.pi*parameters.freqs[i]+sum
    
    damplitude = parameters.rates*((parameters.nominal_amplitudes-amplitudes))
    return np.concatenate((dphase, damplitude), axis=None)
    
    #a changer avec equations du cours


def motor_output(phases, amplitudes,parameters):
    """Motor output"""
    q = amplitudes[:parameters.n_body_joints]*(1+np.cos(phases[:parameters.n_body_joints])) - amplitudes[parameters.n_body_joints:2*parameters.n_body_joints]*(1+np.cos(phases[parameters.n_body_joints:2*parameters.n_body_joints]))
    #print(phases[2*parameters.n_body_joints:])
    q2=amplitudes[2*parameters.n_body_joints:]*(1+np.cos(phases[2*parameters.n_body_joints:]))
    
    q=np.concatenate((np.array(q), np.array(q2)), axis=None)
    return (q)

class ODESolver(object):
    """ODE solver with step integration"""

    def __init__(self, ode, timestep, solver=rk4):
        super(ODESolver, self).__init__()
        self.ode = ode
        self.solver = solver
        self.timestep = timestep
        self._time = 0

    def integrate(self, state, *parameters):
        """Step"""
        diff_state = self.solver(
            self.ode,
            self.timestep,
            self._time,
            state,
            *parameters
        )
        self._time += self.timestep
        return diff_state

    def time(self):
        """Time"""
        return self._time


class RobotState(np.ndarray):
    """Robot state"""

    def __init__(self, *_0, **_1):
        super(RobotState, self).__init__()
        self[:] = 0.0

    @classmethod
    def salamandra_robotica_2(cls):
        """State of Salamandra robotica 2"""
        return cls(2*24, dtype=np.float64, buffer=np.zeros(2*24))  #24

    @property
    def phases(self):
        """Oscillator phases"""
        return self[:24] # 24

    @phases.setter
    def phases(self, value):
        self[:24] = value  # 24

    @property
    def amplitudes(self):
        """Oscillator phases"""
        return self[24:] # 24

    @amplitudes.setter
    def amplitudes(self, value):
        self[24:] = value #24


class SalamanderNetwork(ODESolver):
    """Salamander oscillator network"""

    def __init__(self, timestep, parameters):
        super(SalamanderNetwork, self).__init__(
            ode=network_ode,
            timestep=timestep,
            solver=rk4  # Feel free to switch between Euler (euler) or
                        # Runge-Kutta (rk4) integration methods
        )
        # States
        self.state = RobotState.salamandra_robotica_2()
        # Parameters
        self.parameters = RobotParameters(parameters)
        # Set initial state
        self.state.phases = 1e-4*np.random.ranf(self.parameters.n_oscillators)

    def step(self):
        """Step"""
        self.state += self.integrate(self.state, self.parameters)

    def get_motor_position_output(self):
        """Get motor position"""
        return motor_output(self.state.phases, self.state.amplitudes,self.parameters)

