#!/usr/bin/env python
# -*- coding: utf-8 -*-
##### IMPORTING MODULES ############################################################################################################
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.widgets import Slider, Button
from random import choice
from scipy.interpolate import interp2d
from string import ascii_letters
from pylab import *

import astropy.constants as aconst
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
import shutil
import subprocess
import sys
import os


##### CHECK FOR ARGUMENTS ##########################################################################################################
try:
    data_dir = sys.argv[1]
except:
    sys.exit('You have to give the data diretory as first argument!')


##### LOAD DIMS.DAT ################################################################################################################
# This is an output file which contains for example the number of grid points
try:
    dims = np.loadtxt(data_dir+'/dims.dat')
except:
    sys.exit('Could not load dims.dat')


##### LOAD USED_RAD.DAT ############################################################################################################
# This file contains the positions of the radial grid cell interfaces
try:
    used_rad = np.loadtxt(data_dir+'/used_rad.dat') * u.AU
except:
    sys.exit('Could not load used_rad.dat!')


##### CREATE GRID ##################################################################################################################
N_t       = int( dims[5] )                            # Number of outputs
N_R       = int( dims[6] )                            # Number of radial gridpoint
N_theta   = int( dims[7] )                            # Number of angular grid points
R_cen     = 0.5 * (used_rad[:-1] + used_rad[1:] )     # Calculate radial cell centers
theta_cen = np.linspace( 0., 2.*const.pi, N_theta )   # Theta grid

print N_R, N_t, N_theta
##### LOAD PLANET0.DAT #############################################################################################################
# The planet data file
try:
    planet0 = np.loadtxt(data_dir+'/planet0.dat')
except:
    sys.exit('Could not load planet0.dat!')
dummy_i = -1
for i in range(planet0.shape[0]):
    if(planet0[i, 0] == dummy_i):
        planet0[i-1:planet0.shape[0]-1, :] = planet0[i:,:]
    dummy_i = planet0[i, 0]
# Cartesian coordinates of the planet
planet_x = planet0[:,1] * u.AU
planet_y = planet0[:,2] * u.AU
# Trannsform to polar coordinates
planet_R     = np.sqrt( planet_x**2 + planet_y**2 )
planet_theta = np.arctan2( planet_y, planet_x )


##### LOAD DENSITIES ###############################################################################################################
# Load the temperature and density
mu = 2.3 * u.g / u.mole
Tunit = mu / aconst.R * aconst.G * aconst.M_sun / u.AU
Pconv = aconst.R / mu
i_image = input("Enter output: ")
T = np.zeros( (N_R, N_theta) ) * Tunit  # Initialize temperature array

try:
    T[:, :] = np.fromfile(data_dir+'temperature/temperature'+repr(i_image)+'.dat', dtype = 'float64').reshape(N_R, N_theta) * Tunit
except:
    print 'Could not load gasTemperature'+repr(i_image)+'.dat'
sigma = np.zeros( (N_R, N_theta) ) * aconst.M_sun / u.AU**2 # Initialize density array
try:
    sigma[:, :] = np.fromfile(data_dir+'dens/dens'+repr(i_image)+'.dat', dtype = 'float64').reshape(N_R, N_theta) * aconst.M_sun / u.AU**2
except:
    print 'Could not load gasdens'+repr(i_image)+'.dat'
vtheta = np.zeros( (N_R, N_theta) )  # Initialize density array
try:
    vtheta[:, :] = np.fromfile(data_dir+'vtheta/vtheta'+repr(i_image)+'.dat', dtype = 'float64').reshape(N_R, N_theta) 
except:
    print 'Could not load gasvtheta'+repr(i_image)+'.dat'



# Calculate pressure
#P = sigma * T * Pconv
P = sigma


# Function to interpolate data
def intpol(theta, r, data):
    f = interp2d(theta_cen, R_cen, data, kind='linear')
    return f(theta, r)


##### PLOTTING #####################################################################################################################
# Create amber-teal colormap
color_dict = {'red':   ((0.00, 0.10, 0.10),
                        (0.33, 0.10, 0.10),
                        (0.67, 1.00, 1.00),
                        (1.00, 1.00, 1.00)),
              'green': ((0.00, 0.10, 0.10),
                        (0.33, 0.10, 0.10),
                        (0.67, 0.50, 0.50),
                        (1.00, 1.00, 1.00)),
              'blue':  ((0.00, 0.10, 0.10),
                        (0.33, 0.50, 0.50),
                        (0.67, 0.10, 0.10),
                        (1.00, 1.00, 1.00))
             }
amber_teal = LinearSegmentedColormap('OrangeTeal1', color_dict)
amber_teal.set_under('#191919')
amber_teal.set_over('#FFFFFF')
colormap = 'jet'

# Constrain the colorbar
P_min  = np.min( P[P>0.].cgs.value )                                                      # Minimum (non-empty) value of sigma
P_max  = np.max( P.cgs.value )                                                            # Maximum value of sigma
levels     = np.linspace( np.floor(np.log10(P_min)), np.ceil(np.log10(P_max)), 100 )      # Levels of colorbar
cbar_ticks = np.arange( np.floor(np.log10(P_min)), np.ceil(np.log10(P_max))+0.1, 0.5 )    # Ticks of colorbar

# Create plot
fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))           # Use polar coordinate system
plt.subplots_adjust(bottom=0.25)                                      # Create space at the bottom for the slider

plot = ax.contourf(theta_cen, R_cen.to(u.AU), np.log10(P[ :, :].cgs.value), cmap=colormap, levels=levels, extend='both' )       # Filled contour plot
collections_list = plot.collections[:]                                                                                            # Collect the collections....lol
#planet, = ax.plot(planet_theta[i_image], planet_R[i_image].to(u.AU), color='cyan', marker='o', markersize=8, markeredgecolor='black') # Cross at planet position

ax.set_rmin(-R_cen[0].to(u.AU).value)                                 # Creating inner hole. Otherwise R=R_min would be in the center
ax.set_rmax(R_cen[-1].to(u.AU).value)                                 # Needed somehow to define outer limit.

ax.tick_params(axis='x', labelbottom='off')                           # Turns off the theta axis labelling
ax.tick_params(axis='y', colors='black')                              # Change radial labelling to white for better visibility

cbar = fig.colorbar(plot)                                             # Show colorbar
cbar.ax.set_ylabel(r'log $\Sigma$  [ g/cm$^2$ ]')
cbar.set_ticks(cbar_ticks)
plt.grid(b=True)

#give more value info to the plot
def format_coord(theta, r):
    value = intpol(theta, r, P[ :, :].cgs)
    return r'theta=%1.4f rad, R=%1.4f AU, Sigma=%1.4f g/' % (theta, r, value)

ax.format_coord = format_coord

plt.show()



















