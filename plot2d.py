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


path1 = "/bolt/matias/Sombras-SG/Sombras-SG-long_run/out2/"
path2 = "/bolt/matias/Sombras-SG/Sombras-NoSha_long_run/out2/"
path4 = "/bolt/matias/Sombras-NewSet/No-Shadows/SG1-n1/out1/"
sg0_n0 = "/bolt/matias/Sombras-NewSet/SG_Shadows/SG0-n0/out1/"
sg0_n1 = "/bolt/matias/Sombras-NewSet/SG_Shadows/SG0-n1/out1/"
sg0_n1_E0 = "/bolt/matias/Sombras-NewSet/SG_Shadows/SG0-n1-E0/out1/"

sg1_n0 = "/bolt/matias/Sombras-NewSet/SG_Shadows/SG1-n0/out1/"
sg1_n1 = "/bolt/matias/Sombras-NewSet/SG_Shadows/SG1-n1/out1/"
sg1_n05 = "/bolt/matias/Sombras-NewSet/SG_Shadows/SG1-n0.5/out1/"

sg1_n1_E1 = "/bolt/matias/Sombras-NewSet/SG_Shadows/SG1-n1-E0/out1/"
old = "/bolt/matias/Sombras/ideal/Q-sgma-cte2/out1/"


test0 = "/bolt/matias/Sombras-NewSet/Tests/SG0-n1-E1/out1/"

test1 = "/home/matias/Desktop/CudaProtoplanetaryDisks/out/"



data_dir = test1
print data_dir

output = input("Enter output: ")


# # the font of the legend's plot
# fontP = FontProperties()
# fontP.set_size('large')

# # Make use of Tex
# rc('text',usetex=True)
# rc('xtick', labelsize=18)
# rc('ytick', labelsize=18)

# # Change all fonts to 'Computer Modern'
# rc('font',**{'family':'serif','serif':['Computer Modern']})

#thicker line 
#rc('axes', linewidth=2)


# physical ctes
Munit = 1.0   #solar mass
Runit = 1.0   #au
grav =  39.4862194 # (au^3) / (solar mass * (yr^2))    
Timeunit = sqrt(Runit*Runit*Runit/(grav*Munit)) # yr
Lumunit =   Munit*Runit*Runit/(Timeunit*Timeunit*Timeunit) #  Msun*au^2*yr^-3
Lsun = 2.7e-4 # Msun au^2/yr^3 
Lestrella =  (Lsun/Lumunit) # 
Rgas = 3.67e-4 # //au^2/(yr^2 K)   R =0.000183508935; // R = 4124 [J/kg/K] = 0.000183508935 (au^2) / ((year^2) * kelvin)
Tempunit = 2.35*(grav*Munit/(Rgas*Runit)) # Kelvin
densINcgs = 8888035.76 #1 Msun/au^2 = 8888035.76 grams / (cm^2)
DensUnit = Munit / Runit**2.0 * densINcgs # grams / (cm^2)
QplusUnit = 1.6e13 #erg/s/cm^2
Stefam_Bol_cte = 8.9438e-16*Tempunit**4.0*Timeunit*3.0/Munit #   5.67e-5 #cgs erg cm^-2 s^-1 K^4
Boltz_cte = 3.1e-61 #Msun AU^2/yr^2/K   ##1.4e-16 erg/K
sigma_cte_cgs = 5.67e-5 # erg cm^-2 s^-1 K^-4
k_cte_cgs = 1.4e-16      #erg K^-1
proton_mass_cgs = 1.4e-24 #grams

au_in_cm = 1.49597871e13 #cm       1 au = 1.5e13 cm
Timeunit_in_seconds = Timeunit*3.15569e7 # seconds
vel_unit_cgs = au_in_cm/Timeunit_in_seconds  #cm/s


# grid specification
nrad = 128 #128 #500
nsec = 384 # 256 #1500
Rmin = 0.4
Rmax = 2.5

r = np.linspace(Rmin, Rmax, nrad)

rr = []
for i in range(0,nrad):
	rr.append(Rmin*exp(i*log(Rmax/Rmin)/nrad))
	
theta, rad   = np.meshgrid(np.linspace(0., 2.*np.pi, nsec), rr)
xi = rad * np.cos(theta)
yi = rad * np.sin(theta)

R = rad.reshape(nsec*nrad)  #radial vector

#mesh for the dp/dr quantities
r2 = []
for i in range(0,nrad-1):
	r2.append(Rmin*exp(i*log(Rmax/Rmin)/(nrad-1) ))

theta, rad2   = np.meshgrid(np.linspace(0., 2.*np.pi, nsec), r2)
x2 = (rad2) * np.cos(theta)
y2 = (rad2) * np.sin(theta)

R2 = rad2.reshape(nsec*(nrad-1))  #radial vector
Theta2 = theta.reshape(nsec*(nrad-1))

def vector_field(vx,vy, **karg):
    X = R2*cos(Theta2)
    Y = R2*sin(Theta2)
    U = vy*cos(Theta2) - vx*sin(Theta2)
    V = vy*sin(Theta2) + vx*cos(Theta2)
    ax = gca()
    ax.quiver(X,Y,U,V,scale=5,pivot='midle', **karg)

rho0 =   fromfile(data_dir + "dens/" + "dens{0:d}.raw".format(0),dtype='float64')         #1D array
temp0 = fromfile(data_dir + "temperature/" +"temperature{0:d}.raw".format(0),dtype='float64')         #1D array
rho =   fromfile(data_dir + "dens/" +"dens{0:d}.raw".format(output),dtype='float64')         #1D array
temp =  fromfile(data_dir + "temperature/" +"temperature{0:d}.raw".format(output),dtype='float64') #1D array
qplus = fromfile(data_dir + "qplus/" +"qplus{0:d}.raw".format(output),dtype='float64') #1D array
vrad =  fromfile(data_dir + "vrad/" +"vrad{0:d}.raw".format(output),dtype='float64') #1D array
vtheta= fromfile(data_dir + "vtheta/" +"vtheta{0:d}.raw".format(output),dtype='float64') #1D array


deltaRho = (rho-rho0) / rho0
deltaT = (temp - temp0)/temp0

cs0 = sqrt(1.4*temp0)
press0 = rho*pow(cs0,2.0)  

cs = sqrt(1.4*temp) #*vel_unit_cgs  # cm/s  #1D array
press = rho *pow(cs,2.0)   #*densINcgs*vel_unit_cgs**2.0   # in grams/s^2
delta_p = (rho *pow(cs,2.0)  - press0)/press0

centr_acc = (vrad**2.0+vtheta**2.0)/R

v_k = sqrt(1.0/R) # AU/yr
height = cs * R / v_k # in AU
aspect_ratio = height / R  # no units
#toomre_q = cs*v_k/(np.pi*1.0*R*rho) #no units
toomre_q = cs*vtheta/(np.pi*1.0*R*rho) #no units
cooling_time = (3./2.)*k_cte_cgs*(rho*DensUnit)/(2.3*proton_mass_cgs*sigma_cte_cgs *(temp*Tempunit)**3.0)*3.16888e-8
delta = 0.5
cross_time = R*delta/v_k*Timeunit #yr


gradp_acc_r=[]
gradp_acc_theta = []
for i in range(0,nrad-1):
	for j in range(0,nsec):
		l = j + i*nsec
		k = i + nsec;
		#dr.append(R[l+1]-R[l])
		#print R[nsec+i*nsec]-R[nsec+i*nsec-1]
		dp_th = (press[l+1]-press[l])
		dp_r = (press[nsec+i*nsec]-press[nsec+(i-1)*nsec])
		#dr = R[i]-R[i-1]
		dr = (R[nsec+i*nsec]-R[nsec+(i-1)*nsec])
		#print dr
		dtheta = Theta2[j]-Theta2[j-1]
		gradp_acc_r.append((-1.0/rho[l])* (dp_r/dr))
		gradp_acc_theta.append((-1.0/rho[l]) *(1.0/R[l])*(dp_th/dtheta))

gradp_acc_r = array(gradp_acc_r)   #*(densINcgs*vel_unit_cgs**2.0)/au_in_cm
gradp_acc_theta = array(gradp_acc_theta)

print max(gradp_acc_theta)
grap_module = sqrt(gradp_acc_r**2.0 + gradp_acc_theta**2.0) 

gradp_acc_r_unit = gradp_acc_r/grap_module 
gradp_acc_theta_unit = gradp_acc_theta/grap_module


Rho =          rho.reshape(nrad,nsec) 
DeltaRho =	   deltaRho.reshape(nrad,nsec)   
DeltaT    = deltaT.reshape(nrad,nsec)
DeltaP = delta_p.reshape(nrad,nsec)

Temp =         temp.reshape(nrad,nsec)
Qplus =        qplus.reshape(nrad,nsec)
Vrad =         vrad.reshape(nrad,nsec)
Vtheta =       vtheta.reshape(nrad,nsec)
Centr_acc =    centr_acc.reshape(nrad,nsec)
Cs =    	   cs.reshape(nrad,nsec)
Press = 	   press.reshape(nrad,nsec)
dPress =       diff(Press)
Height =       height.reshape(nrad,nsec)
Aspect_Ratio = aspect_ratio.reshape(nrad,nsec)
Toomre_Q =     toomre_q.reshape(nrad,nsec)
Cooling_Time = cooling_time.reshape(nrad,nsec)
Cross_time =   cross_time.reshape(nrad,nsec)

# derived quantities
gradP_acc_r = gradp_acc_r.reshape(nrad-1,nsec)
gradP_acc_theta = gradp_acc_theta.reshape(nrad-1,nsec)
graP_module = grap_module.reshape(nrad-1,nsec)
 
#print min(toomre_q)

#print shape(dPress)
#print shape(DR)

# figure(figsize=(10,10))  # Basic r-theta plot
# imshow(log10(Rho), origin='lower', cmap='cubehelix', aspect='auto', extent=[Rmin,Rmax,Rmin,Rmax])

# figure(figsize=(15,5))
# rho1D = fromfile(data_dir + "/gasdens{0:d}.dat".format(output)).reshape(nrad,nsec).mean(axis=1)
# plot(linspace(Rmin,Rmax,nrad),rho1D)
# xlim(Rmin,Rmax)


# figure(figsize=(15,5))
# for i in range(5):
#     Rho = fromfile(data_dir + "/gasdens{0:d}.dat".format(i)).reshape(nrad,nsec).mean(axis=1)
#     plot(linspace(Rmin,Rmax,nrad),Rho)
# xlim(Rmin,Rmax)
# show()

figure(100)
imshow(log10(Rho*DensUnit),origin='lower',cmap=cm.Oranges_r,aspect='auto')
#clim(1.2,2.25) #clim(25, 240)
xlabel('x [AU]', fontsize=16)
ylabel('y [AU]', fontsize=16)
cb = plt.colorbar()
#plt.colorbar(format='%.0e')
cb.set_label('log Density [$\\rm g$ $\\rm cm^{-2}$]')

figure(17)
pcolormesh(xi,yi,log10(Rho*DensUnit))
#clim(1.2,2.25) #clim(25, 240)
xlabel('x [AU]', fontsize=20)
ylabel('y [AU]', fontsize=20)
cb = plt.colorbar()
#plt.colorbar(format='%.0e')
cb.set_label('log Density [$\\rm g$ $\\rm cm^{-2}$]', size=20)


figure(21)
pcolormesh(xi,yi,(DeltaT))
#clim(1.2,2.25) #clim(25, 240)
xlabel('x [AU]', fontsize=16)
ylabel('y [AU]', fontsize=16)
cb = plt.colorbar()
#plt.colorbar(format='%.0e')
cb.set_label('$\Sigma - \Sigma_0 / \Sigma_0$')

figure()
pcolormesh(xi,yi,(DeltaP))
#clim(1.2,2.25) #clim(25, 240)
xlabel('x [AU]', fontsize=16)
ylabel('y [AU]', fontsize=16)
cb = plt.colorbar()
#plt.colorbar(format='%.0e')
cb.set_label('$(P - P_0)/P_0$')


figure(2)
pcolormesh(xi,yi,Height)
cb = plt.colorbar()
cb.set_label('Scale height h [AU]')
xlabel('x [AU]', fontsize=16)
ylabel('y [AU]', fontsize=16)

figure(3)
pcolormesh(xi,yi,Aspect_Ratio)
xlabel('x [AU]', fontsize=16)
ylabel('y [AU]', fontsize=16)
cb = plt.colorbar()
cb.set_label('Aspect Ratio')


figure(4)
pcolormesh(xi,yi,(Tempunit*Temp))
#clim(log10(50),log10(500))
xlabel('x [AU]', fontsize=16)
ylabel('y [AU]', fontsize=16)
cb = plt.colorbar()
cb.set_label('Temperature [K]')

figure(5)
pcolormesh(xi,yi, log10(Qplus*QplusUnit))
cb = plt.colorbar()
cb.set_label('$Q^+$ $[\\rm erg$ $s^{-1}$ $\\rm cm^{-2}]$ (log)', size=20)
xlabel('x [AU]', fontsize=20)
ylabel('y [AU]', fontsize=20)

figure(6)
pcolormesh(xi,yi,(Toomre_Q))
xlabel('x [AU]', fontsize=20)
ylabel('y [AU]', fontsize=20)
cb = plt.colorbar()
cb.set_label('Toomre parameter', size=20)

figure(7)
pcolormesh(xi,yi,(Cooling_Time))
#clim(0,90)
xlabel('x [AU]', fontsize=16)
ylabel('y [AU]', fontsize=16)
cb = plt.colorbar()
cb.set_label('Cooling Time [yr]')

figure(8)
pcolormesh(xi,yi,(Cross_time))
clim(0,30)
xlabel('x [AU]', fontsize=16)
ylabel('y [AU]', fontsize=16)
cb = plt.colorbar()
cb.set_label('Crossing Time [yr]')

figure(9)
pcolormesh(xi,yi,(Cross_time/Cooling_Time))
#clim(0.3,2.7)
xlabel('x [AU]', fontsize=16)
ylabel('y [AU]', fontsize=16)
cb = plt.colorbar()
cb.set_label('Crossing / Cooling')


figure(10)
pcolormesh(xi,yi, log10(Press*densINcgs*vel_unit_cgs**2.0/au_in_cm))
xlabel('x [AU]', fontsize=16)
ylabel('y [AU]', fontsize=16)
cb = plt.colorbar()
cb.set_label('log Pressure field [$g$ $cm^{-1}$ $s^{-2}$]')


figure(11)
pcolormesh(xi,yi,-Vrad*vel_unit_cgs)
title("sine")
#clim(-30,0)
xlabel('x [AU]', fontsize=16)
ylabel('y [AU]', fontsize=16)
cb = plt.colorbar()
cb.set_label('Radial Velocity [$cm$ $s^{-1}$]')

figure(12)
pcolormesh(xi,yi,Vtheta*vel_unit_cgs)
xlabel('x [AU]', fontsize=16)
ylabel('y [AU]', fontsize=16)
cb = plt.colorbar()
cb.set_label('Azimuthal Velocity')

figure(13)
pcolormesh(xi,yi, Centr_acc * (vel_unit_cgs**2.0/au_in_cm) )
xlabel('x [AU]', fontsize=16)
ylabel('y [AU]', fontsize=16)
cb = plt.colorbar()
cb.set_label('Centrifugal acceleration')

figure(14)
pcolormesh(x2,y2, gradP_acc_r*(densINcgs*vel_unit_cgs**2.0)/au_in_cm)
#clim(-45,45)
xlabel('x [AU]', fontsize=16)
ylabel('y [AU]', fontsize=16)
cb = plt.colorbar()
cb.set_label('grad P acceleration [$g$ $cm^{-2}$ $s^{-2}$]')

figure(15)
pcolormesh(x2,y2, gradP_acc_theta*(densINcgs*vel_unit_cgs**2.0)/au_in_cm)
#clim(-0.08,0.22)
xlabel('x [AU]', fontsize=16)
ylabel('y [AU]', fontsize=16)
cb = plt.colorbar()
cb.set_label('grad P theta acceleration')

figure(16)
pcolormesh(x2,y2, graP_module*(densINcgs*vel_unit_cgs**2.0)/au_in_cm)
xlabel('x [AU]', fontsize=16)
ylabel('y [AU]', fontsize=16)
cb = plt.colorbar()
cb.set_label('grad P module')

# figure(17)
# vector_field(Vrad, Vtheta)


show()

