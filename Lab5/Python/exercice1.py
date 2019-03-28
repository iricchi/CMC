
"""
Created on Thu Mar 28 17:15:32 2019

@author: BARTHE Lancelot
"""

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
    t_stop = 1
    time_step = 0.001    
    time = np.arange(t_start, t_stop, time_step)
    
    # Set the initial condition
    x0 = [0.0, sys.muscle.L_OPT]
    
    # Create stretch coefficient vector
    dL=np.linspace(-1, 1, num=100)
    
    # Create a steady state muscle force vector
    R_glob_1a=np.zeros((100))

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
        R_glob_1a[i]=result.tendon_force[len(result.tendon_force)-1]
        
        # Plot Isometric muscle experiment
        plt.figure('Isometric muscle experiment')
        plt.plot(result.time, result.tendon_force)
        plt.title('Isometric muscle experiment')
        plt.xlabel('Time [s]')
        plt.ylabel('Muscle force')
    
    # Plot force-length curve
    plt.figure('Forces as a function of stretchning coefficient')
    plt.plot(dL,R_glob_1a)
    plt.xlabel('Strecth coeffcient [%]')
    plt.ylabel('Muscle force')
        
    
            
    
    """1.b) Effect of muscle stimulation [-1,0] on muscle force as a function of stretch coefficient"""
    
    # Create a steady state muscle force vector
    R_glob_1b=np.zeros((5,100))
    
    # Create muscle activation vector
    dS=np.linspace(0,-1,num=5)
    
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
            R_glob_1b[j,i]=result.tendon_force[len(result.tendon_force)-1]
            
            
    # Plot force-length curve for different muscle activation
    plt.figure('Forces as a function of dL for different muscle stimulation')
    plt.plot(dL,R_glob_1b[0,:])
    plt.plot(dL,R_glob_1b[1,:])
    plt.plot(dL,R_glob_1b[2,:])
    plt.plot(dL,R_glob_1b[3,:])
    plt.plot(dL,R_glob_1b[4,:])
    plt.xlabel('Strecth coeffcient')
    plt.ylabel('Muscle force')
    
    
    """ 1.c) Effect of fiber length on force-length curve """
    
    # Evalute for a single muscle stimulation
    muscle_stimulation = 0.5
    # Create a steady state muscle force vector
    R_glob_1c=np.zeros((9,100))
    # Create fiber length vector
    dl=np.linspace(0.1,1,num=9)
    
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
            R_glob_1c[j,i]=result.tendon_force[len(result.tendon_force)-1]
            
            
    plt.figure('Forces as a function of dL for different fiber length')
    plt.plot(dL,R_glob_1c[0,:])
    plt.plot(dL,R_glob_1c[2,:])
    plt.plot(dL,R_glob_1c[4,:])
    plt.plot(dL,R_glob_1c[6,:])
    plt.plot(dL,R_glob_1c[8,:])
    plt.xlabel('Strecth coeffcient')
    plt.ylabel('Muscle force')