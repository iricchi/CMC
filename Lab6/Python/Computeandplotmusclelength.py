# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 13:35:56 2019

@author: Ilaria
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
theta = np.arange(-math.pi/4, math.pi/4, stepplot)

"""Exercise 2a"""
def compute_length(muscleorigin,muscleinsertion, theta):
    
    a1=math.fabs(muscleorigin[0])
    a2=abs(muscleinsertion[1]) 
    Length=[]
    for i in range(len(theta)):
        if(muscleorigin[0]<0):
            Length.append(sqrt(a1**2+a2**2+2*a1*a2*math.sin(theta[i])))
        else :
            Length.append(sqrt(a1**2+a2**2+2*a1*a2*math.sin(-theta[i])))
    return Length


def compute_moment(Length, theta, muscleorigin, muscleinsertion):
    
    a1=math.fabs(muscleorigin[0])
    a2=abs(muscleinsertion[1]) 
    h = []
    for i in range(len(theta)):
        h.append(a1*a2*math.cos(theta[i])/Length[i])
    
    return h
    
    
L1 = compute_length(m1_origin,m1_insertion, theta)
L2 = compute_length(m2_origin,m2_insertion, theta)

h = compute_moment(L1,theta,m1_origin,m1_insertion)
h2 = compute_moment(L2, theta,m2_origin,m2_insertion)

plt.figure()
plt.plot(theta, L1) 
plt.plot(theta, L2)
plt.legend(['Muscle 1', 'Muscle 2'])
plt.title('Length with respect to angle theta')

plt.figure()
plt.plot(theta,h)
plt.plot(theta,h2)   
plt.legend(['Muscle 1', 'Muscle 2'])
plt.title('Moment with respect to angle theta')