# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 13:35:56 2019

@author: Julien
"""
import math
from math import sqrt
import numpy as np
from matplotlib import pyplot as plt

m1_origin = np.array([-0.17, 0.0])  # Origin of Muscle 1
m1_insertion = np.array([0.0, -0.17])  # Insertion of Muscle 1
m2_origin = np.array([0.17, 0.0])  # Origin of Muscle 2
m2_insertion = np.array([0.0, -0.17])  # Insertion of Muscle 2
stepplot=0.01



def computeandplotmusclelength(muscleorigin, muscleinsertion):   
    
    a1=math.fabs(muscleorigin[0])
    a2=abs(muscleinsertion[1]) 
    theta = np.arange(-math.pi/4, math.pi/4, stepplot)
    Length=[]
    for i in range(len(theta)):
        if(muscleorigin[0]<0):
            Length.append(sqrt(a1**2+a2**2+2*a1*a2*math.sin(theta[i])))
        else :
            Length.append(sqrt(a1**2+a2**2+2*a1*a2*math.sin(-theta[i])))
    plt.plot(theta, Length)    
   
computeandplotmusclelength(m2_origin, m2_insertion)