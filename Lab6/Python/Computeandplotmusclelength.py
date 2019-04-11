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


m1_insertions = [np.array([0.0, -0.11]), np.array([0.0, -0.23]), np.array([0.0, -0.32])]
m2_insertions = [np.array([0.0, -0.11]), np.array([0.0, -0.23]), np.array([0.0, -0.32])]

m1_insertion2 = np.array([0.0, -0.17])
m2_insertion2 = np.array([0.0, -0.23])


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
plt.xlabel('Theta [rad]')
plt.ylabel('Length [m]')

plt.legend(['Muscle 1', 'Muscle 2'])
plt.title('Length with respect to angle theta')

plt.figure()
plt.plot(theta,h)
plt.plot(theta,h2)   
plt.xlabel('Theta [rad]')
plt.ylabel('Moment arm [m]')

plt.legend(['Muscle 1', 'Muscle 2'])
plt.title('Moment arm with respect to angle theta')



plt.figure('Varying insertion point')
ax1 = plt.subplot(211)
ax2 = plt.subplot(212)

""" Varying insertion point """
for i in range(len(m1_insertions)):
    L1t = compute_length(m1_origin, m1_insertions[i], theta)
    L2t = compute_length(m2_origin, m2_insertions[i], theta)
    
    h1t = compute_moment(L1t, theta, m1_origin,m1_insertions[i])
    h2t = compute_moment(L2t, theta, m2_origin,m2_insertions[i])
    
    ax1.plot(theta, L1t)
    ax1.plot(theta, L2t)
    ax2.plot(theta, h1t)
    ax2.plot(theta, h2t)
    

ax1.set_xlabel('Theta [rad]')    
ax2.set_xlabel('Theta [rad]') 
ax1.set_ylabel('Length [m]')
ax2.set_ylabel('Moment arm [m]')

ax1.title.set_text('Length')
ax2.title.set_text('Moment Arm')
ax1.legend(['Muscle 1 at ip = [0 -0.11]', 'Muscle 2 at ip = [0 -0.11]','Muscle 1 at ip = [0 -0.23]', 'Muscle 2 at ip = [0 -0.23]',
            'Muscle 1 at ip = [0 -0.32]', 'Muscle 2 at ip = [0 -0.32]'])
ax2.legend(['Muscle 1 at ip = [0 -0.11]', 'Muscle 2 at ip = [0 -0.11]','Muscle 1 at ip = [0 -0.23]', 'Muscle 2 at ip = [0 -0.23]',
            'Muscle 1 at ip = [0 -0.32]', 'Muscle 2 at ip = [0 -0.32]'])
    
    
""" Insertion point not equal in the two muscles """

L12 = compute_length(m1_origin, m1_insertion2, theta)
L22 = compute_length(m2_origin, m2_insertion2, theta)

h = compute_moment(L12,theta,m1_origin,m1_insertion2)
h2 = compute_moment(L22, theta,m2_origin,m2_insertion2)

plt.figure()
plt.plot(theta, L12) 
plt.plot(theta, L22)
plt.xlabel('Theta [rad]')
plt.ylabel('Length [m]')

plt.legend(['Muscle 1', 'Muscle 2'])
plt.title('Length')

plt.figure()
plt.plot(theta,h)
plt.plot(theta,h2)   
plt.xlabel('Theta [rad]')
plt.ylabel('Moment arm [m]')

plt.legend(['Muscle 1', 'Muscle 2'])
plt.title('Moment arm')