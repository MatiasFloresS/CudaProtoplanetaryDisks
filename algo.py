#!/usr/bin/python
from numpy import *
from pylab import *
import pylab as m
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import numpy as numpy
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.colors import LogNorm
from matplotlib.colors import ListedColormap
from matplotlib import rc, rcParams
from matplotlib.font_manager import FontProperties
import matplotlib

test1 = "/home/matias/Desktop/CudaProtoplanetaryDisksFloat/out/"
test2 = "/home/matias/Desktop/outputGFARGO/128x384_L_3.2_7.2_OPEN_ADIA_NOSG/out/"
out = "/home/matias/Desktop/CudaProtoplanetaryDisksFloat/fargo.png"

data_dir = test2
output = input("Enter output: ")

Munit = 1.0   #solar mass
Runit = 1.0   #au
grav =  39.4862194 # (au^3) / (solar mass * (yr^2))    
Rgas = 3.67e-4 # //au^2/(yr^2 K)   R =0.000183508935; // R = 4124 [J/kg/K] = 0.000183508935 (au^2) / ((year^2) * kelvin)
Tempunit = 2.35*(grav*Munit/(Rgas*Runit)) # Kelvin

# grid specification
nrad = 128 #128 #500
nsec = 384 # 256 #1500
Rmin = 3.2
Rmax = 7.2

r = np.linspace(Rmin, Rmax, nrad)

rr = []
for i in range(0,nrad):
	rr.append(Rmin*exp(i*log(Rmax/Rmin)/nrad))
	
theta, rad   = np.meshgrid(np.linspace(0., 2.*np.pi, nsec), rr)
xi = rad * np.cos(theta)
yi = rad * np.sin(theta)



temp =  fromfile(data_dir + "temperature/" +"temperature{0:d}.raw".format(output),dtype='float32') #1D array


Temp =         temp.reshape(nrad,nsec)


fig = plt.figure()
plt.pcolormesh(xi,yi,(Temp*Tempunit))
#clim(log10(50),log10(500))
xlabel('x [AU]', fontsize=16)
ylabel('y [AU]', fontsize=16)
cb = plt.colorbar()
cb.set_label('Temperature [K]')

#show()

savefig(out, format='png', dpi=1000, bbox_inches='tight')
#plt.savefig(out, format='eps')
#plt.show()
