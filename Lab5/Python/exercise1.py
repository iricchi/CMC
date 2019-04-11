""" Lab 5 - Exercise 1 """

import matplotlib.pyplot as plt
import numpy as np

import cmc_pylog as pylog
from muscle import Muscle
from mass import Mass
from cmcpack import DEFAULT, parse_args
from cmcpack.plot import save_figure
from system_parameters import MuscleParameters, MassParameters
from isometric_muscle_system import IsometricMuscleSystem
from isotonic_muscle_system import IsotonicMuscleSystem

DEFAULT["label"] = [r"$\theta$ [rad]", r"$d\theta/dt$ [rad/s]"]

# Global settings for plotting
# You may change as per your requirement
plt.rc('lines', linewidth=2.0)
plt.rc('font', size=12.0)
plt.rc('axes', titlesize=14.0)     # fontsize of the axes title
plt.rc('axes', labelsize=14.0)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=14.0)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14.0)    # fontsize of the tick labels

DEFAULT["save_figures"] = True


def exercise1a():
    """ Exercise 1a
    The goal of this exercise is to understand the relationship
    between muscle length and tension.
    Here you will re-create the isometric muscle contraction experiment.
    To do so, you will have to keep the muscle at a constant length and
    observe the force while stimulating the muscle at a constant activation."""

    # Defination of muscles
    parameters = MuscleParameters()
    pylog.warning("Loading default muscle parameters")
    pylog.info(parameters.showParameters())
        
    # Create muscle object
    muscle = Muscle(parameters)
    

    # Instatiate isometric muscle system
    sys = IsometricMuscleSystem()

    # Add the muscle to the system
    sys.add_muscle(muscle)
    
    # Evalute for a single muscle stimulation
    muscle_stimulation = 0.5

    # Create time vector
    t_start = 0.0
    t_stop = 0.3
    time_step = 0.001    
    time = np.arange(t_start, t_stop, time_step)
    
    # Set the initial condition
    x0 = [0.0, sys.muscle.L_OPT]
    
    # Create stretch coefficient vector
    dL=np.linspace(-0.5, 0.5, num=50)
    len_ce = np.zeros(dL.size)
    # Create a steady state muscle force vector
    R_glob_1a=np.zeros((50))
    R_pass_1a=np.zeros((50))
    R_activ_1a=np.zeros((50))

    """ 1.a) Effect of stretching coefficient on the muscle force """
    
    for i in range (0,len(dL)):

        # Evalute for a single muscle stretch
        muscle_stretch = (sys.muscle.L_OPT+sys.muscle.L_SLACK)*(1+dL[i])
        
        # Run the integration
        result = sys.integrate(x0=x0,
                                time=time,
                                time_step=time_step,
                                stimulation=muscle_stimulation,
                                muscle_length=muscle_stretch)  
        R_glob_1a[i]=result.tendon_force[-1]
        R_pass_1a[i]=result.passive_force[-1]
        R_activ_1a[i]=result.active_force[-1]
        len_ce[i] = result.l_ce[-1]
        
#        # Plot Isometric muscle experiment
#        plt.figure('Isometric muscle experiment')
#        plt.plot(result.time, result.tendon_force)
#        plt.title('Isometric muscle experiment')
#        plt.xlabel('Time [s]')
#        plt.ylabel('Muscle force')
    
    # Plot force-length curve 
    plt.figure('Forces as a function of stretching coefficient')
    # Passive Force
    plt.plot(dL,R_pass_1a)
    # Active Force
    plt.plot(dL,R_activ_1a)
    # Total force
    plt.plot(dL,R_glob_1a)
    
    plt.xlabel('Stretch coeffcient [%]')
    plt.ylabel('Muscle force [N]') 
    plt.legend(['Passive Force', 'Active Force','Total Force'])
    plt.ylim([0, 4000]) 
    plt.xlim([-0.4,0.4])          
    
    plt.figure('Forces as a function of the contractile element length')
    # Passive Force
    plt.plot(len_ce,R_pass_1a)
    # Active Force
    plt.plot(len_ce,R_activ_1a)
    # Total force
    plt.plot(len_ce,R_glob_1a)
    
    plt.ylim([0,4000])
    plt.xlim([0.06,0.18])
    plt.xlabel('Contractile element Lenght [m]')
    plt.ylabel('Muscle force [N]')
    plt.legend(['Passive Force', 'Active Force','Total Force'])
    
    
    
    #plt.savefig('Forces_stretch_coeff')
    """1.b) Effect of muscle stimulation [-1,0] on muscle force as a function of stretch coefficient"""
    
    # Create a steady state muscle force vector
    R_glob_1b=np.zeros((5,50))
    R_pass_1b=np.zeros((5,50))
    R_act_1b=np.zeros((5,50))
    # Create muscle activation vector
    dS=np.linspace(0,1,num=5)
    
    for i in range (0,len(dL)):
        for j in range(0,len(dS)):
            
            # Evalute for a single muscle stimulation
            muscle_stimulation = dS[j]
            
            # Evalute for a single muscle stretch
            muscle_stretch = (sys.muscle.L_OPT+sys.muscle.L_SLACK)*(1+dL[i])
            
            # Run the integration
            result = sys.integrate(x0=x0,
                                    time=time,
                                    time_step=time_step,
                                    stimulation=muscle_stimulation,
                                    muscle_length=muscle_stretch)  
            R_glob_1b[j,i]=result.tendon_force[-1]
            R_pass_1b[j,i]=result.passive_force[-1]
            R_act_1b[j,i]=result.active_force[-1]
    # Plot force-length curve for different muscle activation
    plt.figure('Forces as a function of dL for different muscle stimulation')
    plt.plot(dL,R_glob_1b[0,:])
    plt.plot(dL,R_glob_1b[1,:])
    plt.plot(dL,R_glob_1b[2,:])
    plt.plot(dL,R_glob_1b[3,:])
    plt.plot(dL,R_glob_1b[4,:])
    plt.xlabel('Stretch coeffcient [%]')
    plt.ylabel('Muscle force [N]')
    plt.legend(['Stimulation = 0','Stimulation = 0.25','Stimulation = 0.5','Stimulation = 0.75','Stimulation = 1'])
    plt.title('Total forces')
    
    plt.ylim([0,4000])
    
    
    plt.figure('Active forces as a function of dL for different muscle stimulation')
    plt.plot(dL,R_act_1b[0,:])
    plt.plot(dL,R_act_1b[1,:])
    plt.plot(dL,R_act_1b[2,:])
    plt.plot(dL,R_act_1b[3,:])
    plt.plot(dL,R_act_1b[4,:])
    plt.xlabel('Stretch coeffcient [%]')
    plt.ylabel('Muscle force [N]')
    plt.legend(['Stimulation = 0','Stimulation = 0.25','Stimulation = 0.5','Stimulation = 0.75','Stimulation = 1'])
    plt.title('Active forces')

    plt.figure('Passive forces as a function of dL for different muscle stimulation')
    plt.plot(dL,R_pass_1b[0,:])
    plt.plot(dL,R_pass_1b[1,:])
    plt.plot(dL,R_pass_1b[2,:])
    plt.plot(dL,R_pass_1b[3,:])
    plt.plot(dL,R_pass_1b[4,:])
    plt.xlabel('Stretch coeffcient [%]')
    plt.ylabel('Muscle force [N]')
    plt.legend(['Stimulation = 0','Stimulation = 0.25','Stimulation = 0.5','Stimulation = 0.75','Stimulation = 1'])
    plt.title('Passive forces')
    
    
    
    """ 1.c) Effect of fiber length on force-length curve """
    
    # Evalute for a single muscle stimulation
    muscle_stimulation = 0.5
    # Create fiber length vector
    #dl=np.linspace(0.1,1,num=10)
    dl = [0.07, 0.11, 0.15]
    # Create a steady state muscle force vectors
    R_glob_1c=np.zeros((len(dl),len(dL)))
    # Create active force vectors
    R_act_1c = np.zeros((len(dl),len(dL)))
    # Create passive force vectors
    R_pas_1c = np.zeros((len(dl),len(dL)))
    # Create contractile element length vectors
    len_ce = np.zeros((len(dl),len(dL)))
    
    
    for i in range (0,len(dL)):
        for j in range(0,len(dl)):
            
            # Change the fiber length
            sys.muscle.L_OPT=dl[j]
            
            # Evalute for a single muscle stretch
            muscle_stretch = (sys.muscle.L_OPT+sys.muscle.L_SLACK)*(1+dL[i])
            
            # Run the integration
            result = sys.integrate(x0=x0,
                                    time=time,
                                    time_step=time_step,
                                    stimulation=muscle_stimulation,
                                    muscle_length=muscle_stretch)  
            R_glob_1c[j,i]=result.tendon_force[-1]
            R_act_1c[j,i] = result.active_force[-1]
            R_pas_1c[j,i] = result.passive_force[-1]
            len_ce[j,i] = result.l_ce[-1]
            
    plt.figure('Forces as a function of dL for different fiber lengths')
    for i in range(0,len(dl)):
        plt.plot(dL, R_glob_1c[i,:])

    plt.xlabel('Strecth coeffcient')
    plt.ylabel('Muscle force')
    plt.legend(['Opt_len: 0.07','Opt_len: 0.11','Opt_len : 0.15'])
    
    plt.figure('Forces wrt CE length for different optimal lengths')    
    for i in range(0,len(dl)):
        plt.plot(len_ce[i,:], R_glob_1c[i,:])    
        
    plt.legend(['Opt_len: 0.07','Opt_len: 0.11','Opt_len : 0.15'])
    for i in range(0,len(dl)):
        plt.axvline(dl[i], color='r', linestyle='--')
    mvs = np.max(R_act_1c, axis=1)
    mv = np.unique(np.round(mvs))
    plt.axhline(mv, color='black',linestyle = '--')
    plt.ylim([0,3000])
    plt.xlabel('Contractile elememnt length [m]')    
    plt.ylabel('Muscle force [n]')
    
    plt.figure('Active forces')
    for i in range(0,len(dl)):
        plt.plot(len_ce[i,:], R_act_1c[i,:])    
        
    plt.xlabel('Contractile elememnt length [m]')   
    plt.ylabel('Muscle force [n]')
    plt.legend(['Opt_len: 0.07','Opt_len: 0.11','Opt_len : 0.15'])

    plt.figure('Passive forces')    
    for i in range(0,len(dl)):
        plt.plot(len_ce[i,:], R_pas_1c[i,:])    
        
    plt.xlabel('Contractile elememnt length [m]')   
    plt.ylabel('Muscle force [n]')
    plt.legend(['Opt_len: 0.07','Opt_len: 0.11','Opt_len : 0.15'])
    
        


def exercise1d():
    """ Exercise 1d

    Under isotonic conditions external load is kept constant.
    A constant stimulation is applied and then suddenly the muscle
    is allowed contract. The instantaneous velocity at which the muscle
    contracts is of our interest."""
    
   
    # Defination of muscles
    muscle_parameters = MuscleParameters()
    print(muscle_parameters.showParameters())

    mass_parameters = MassParameters()
    print(mass_parameters.showParameters())

    # Create muscle object
    muscle = Muscle(muscle_parameters)

    # Create mass object
    mass = Mass(mass_parameters)

    pylog.warning("Isotonic muscle contraction to be implemented")

    # Instatiate isotonic muscle system
    sys = IsotonicMuscleSystem()

    # Add the muscle to the system
    sys.add_muscle(muscle)

    # Add the mass to the system
    sys.add_mass(mass)
    
    # Evalute for a single muscle stimulation
    muscle_stimulation = 1.
    
    #sys.muscle.L_OPT=0.05

    
    
    # x0[0] - -> activation
    # x0[1] - -> contractile length(l_ce)
    # x0[2] - -> position of the mass/load
    # x0[3] - -> velocity of the mass/load

    # Set the time for integration
    t_start = 0.0
    t_stop = 0.25
    time_step = 0.001
    time_stabilize = 0.2

    time = np.arange(t_start, t_stop, time_step)


    # You can still access the muscle inside the system by doing
    # >>> sys.muscle.L_OPT # To get the muscle optimal 
    
    # Create a V_ce vector
    V_ce_1d=np.zeros(100)
    PF_ce_1d=np.zeros(100)
    AF_ce_1d=np.zeros(100)
    # Create load vector
    Load=np.linspace(0.1,800,num=100)
    
    
    """1.d) velocity-tension analysis """

    for i in range (0,len(Load)):
        # Set the initial condition
        x0 = [muscle_stimulation, sys.muscle.L_OPT,
          sys.muscle.L_OPT + sys.muscle.L_SLACK, 0.0]
        
        # Evalute for a single load
        load = Load[i]
            
        # Run the integration
        result = sys.integrate(x0=x0,
                               time=time,
                               time_step=time_step,
                               time_stabilize=time_stabilize,
                               stimulation=muscle_stimulation,
                               load=load)
        #print(sys.muscle.L_OPT+sys.muscle.L_SLACK)
        #print(result.l_mtc[-1]-(sys.muscle.L_OPT+sys.muscle.L_SLACK))
        
        if (result.l_mtc[-1] < sys.muscle.L_OPT+sys.muscle.L_SLACK):
            V_ce_1d[i]=min(result.v_ce)
        
        else :
            
            V_ce_1d[i] =max(result.v_ce)
            
        PF_ce_1d[i] =result.passive_force[-1]
        AF_ce_1d[i] =result.active_force[-1]  
        
        # Plotting
        plt.figure('Isotonic muscle experiment')
        plt.plot(result.time, result.v_ce)
        plt.title('Isotonic muscle experiment')
        plt.xlabel('Time [s]')
        plt.ylabel('Contractile element velocity')
        plt.grid()
            
            
            
    # Plot velocity versus tension
    plt.figure()
    plt.plot(V_ce_1d,Load)
    plt.title('Isotonic muscle experiment')
    plt.xlabel('Contractile element velocity')
    plt.ylabel('Load')
    plt.grid()
    
    # Plot velocity versus tension
    plt.figure()
    plt.plot(V_ce_1d,PF_ce_1d+AF_ce_1d)
    plt.plot(V_ce_1d, PF_ce_1d)
    plt.plot(V_ce_1d, AF_ce_1d, '--')
    plt.title('Isotonic muscle experiment')
    plt.xlabel('Contractile element velocity')
    plt.ylabel('Total Force')
    plt.grid()
    plt.legend(['total','passive', 'active'])
    plt.axvline(0, color='r', linestyle='--')
    
    
    
    """ 1.f) velocity-tension as a function of muscle activation """
    # Create solution vector
    #R_glob=np.zeros((5,50))
    # Create muscle activation vector
    dS=np.linspace(0.1,1,num=5)
    # Create a V_ce vector
    V_ce=np.zeros((5,100))
    
    PF_ce=np.zeros((5,100))
    AF_ce=np.zeros((5,100))
    for i in range (0,len(Load)):
        for j in range(0,len(dS)):
            
            
            # Evalute for a single load
            load = Load[i]
            
            # Evalute for a single muscle stimulation
            muscle_stimulation = dS[j]
            
            # Set the initial condition
            x0 = [muscle_stimulation, sys.muscle.L_OPT,
                      sys.muscle.L_OPT + sys.muscle.L_SLACK, 0.0]
            
            
        
            # Run the integration
            result = sys.integrate(x0=x0,
                                    time=time,
                                    time_step=time_step,
                                    time_stabilize=time_stabilize,
                                    stimulation=muscle_stimulation,
                                    load=load)
            #R_glob[j,i]=result.tendon_force[len(result.tendon_force)-1]
            #print(sys.muscle.L_OPT+sys.muscle.L_SLACK)
            #print(result.l_mtc[-1]-(sys.muscle.L_OPT+sys.muscle.L_SLACK))
            
            if (result.l_mtc[-1] < sys.muscle.L_OPT+sys.muscle.L_SLACK):
                V_ce[j,i]=min(result.v_ce)
        
            else :
            
                V_ce[j,i] =max(result.v_ce)
            
            PF_ce[j,i] =result.passive_force[-1]
            AF_ce[j,i] =result.active_force[-1]  
                
            # # Plotting
            # plt.figure('Isotonic muscle experiment')
            # plt.plot(result.time, result.v_ce)
            # plt.title('Isotonic muscle experiment')
            # plt.xlabel('Time [s]')
            # plt.ylabel('Contractile element velocity')
            # plt.grid()
            
            
        
    # Plot velocity versus tension
    plt.figure()
    for i in range(0,5):
        plt.plot(V_ce[i,:],PF_ce[i,:]+AF_ce[i,:])
        plt.title('Isotonic muscle experiment')
        plt.xlabel('Contractile element velocity')
        plt.ylabel('Force')
        plt.grid()
    plt.legend(['Stimulation = 0.1','Stimulation = 0.28','Stimulation = 0.46','Stimulation = 0.64','Stimulation = 0.82','Stimulation = 1'])
            
        


def exercise1():
    exercise1a()
    exercise1d()

    if DEFAULT["save_figures"] is False:
        plt.show()
    else:
        figures = plt.get_figlabels()
        print(figures)
        pylog.debug("Saving figures:\n{}".format(figures))
        for fig in figures:
            plt.figure(fig)
            save_figure(fig)
            plt.close(fig)


if __name__ == '__main__':
    from cmcpack import parse_args
    parse_args()
    exercise1()

