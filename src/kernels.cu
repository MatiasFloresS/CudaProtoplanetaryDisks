#include "kernels.cuh"
#include <stdio.h>
using namespace std;


__global__ void substep1(float *press, float *dens, float *vradint, float *invdiffRmed, float *pot,
   float *Rinf, float *invRinf, float *vrad, float *vthetaint, float *vtheta, float *Rmed, float dt,
 int nrad, int nsec, float OmegaFrame, bool ZMPlus, float IMPOSEDDISKDRIFT, float SIGMASLOPE, float *powRmed)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = threadIdx.y + blockDim.y*blockIdx.y;
  float gradp, gradphi, vt2;
  i+= 1;
  // i=1->nrad , j=0->nsec

  if (i<nrad && j<nsec)
  {
    gradp = 2.0*(press[i*nsec + j] - press[(i-1)*nsec + j])/(dens[i*nsec + j] + dens[(i-1)*nsec + j])*invdiffRmed[i];
    gradphi = (pot[i*nsec + j]-pot[(i-1)*nsec + j])*invdiffRmed[i];
    vt2 = press[i*nsec + j] + press[(i-1)*nsec + j] + press[i*nsec + (j+1)%nsec] + press[(i-1)*nsec + (j+1)%nsec];
    vt2 = vt2/4.0+Rinf[i]*OmegaFrame;
    vradint[i*nsec + j] = vrad[i*nsec + j] + dt*(-gradp - gradphi + vt2*vt2*invRinf[i]);
  }

  i-=1;

  // i=0->nrad ,   j=0->nsec
  if (i<nrad && j<nsec)
  {
    gradp = 2.0*(press[i*nsec + j] - press[i*nsec + ((j-1)+nsec)%nsec])/(dens[i*nsec +j] +dens[i*nsec + ((j-1)+nsec)%nsec]) \
    *1.0/(2.0*CUDART_PI_F/nsec*Rmed[i]);


    if (ZMPlus)
    {
      gradp *= 1; //gradp *= SG_aniso_coeff;  Definir mas adelante SG_aniso_coeff
    }

    gradphi = (pot[i*nsec+ j] - pot[i*nsec + ((j-1)+nsec)%nsec])*1.0/(2.0*CUDART_PI_F/nsec*Rmed[i]);
    vthetaint[i*nsec + j] = vtheta[i*nsec + j]- dt*(gradp+gradphi);
    vthetaint[i*nsec + j] += dt*IMPOSEDDISKDRIFT*0.5*powRmed[i];

  }

}


__global__ void UpdateVelocities(float *vt, float *vr, float *invRmed, float *Rmed, float *Rsup,
  float *Rinf, float *invdiffRmed, float *invdiffRsup, float *dens, float *invRinf, float *Trr,
  float *Trp, float *Tpp, float DeltaT, int nrad, int nsec)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = threadIdx.y + blockDim.y*blockIdx.y;

  i+=1;
  /* Now we can update velocities
     with the viscous source term
     of Navier-Stokes equation */

  /* vtheta first */
  if (i<nrad-1 && j<nsec)
  {
    vt[i*nsec +j] += DeltaT*invRmed[i]*((Rsup[i]*Trp[(i+1)*nsec+ j]-Rinf[i]*Trp[i*nsec +j])*invdiffRsup[i] + \
    (Tpp[i*nsec +j]-Tpp[i*nsec + ((j-1)+nsec)%nsec])*1.0/(2.0*CUDART_PI_F/nsec) + 0.5*(Trp[i*nsec + j] + Trp[(i+1)*nsec +j]))/ \
    (0.5*(dens[i*nsec +j]+dens[i*nsec + ((j-1)+nsec)%nsec]));
  }

  /* now vrad */
  if (i<nrad && j<nsec)
  {
    vr[i*nsec +j] += DeltaT*invRinf[i]*((Rmed[i]*Trr[i*nsec +j]- Rmed[i-1]*Trr[(i-1)*nsec + j])*invdiffRmed[i] + \
    (Trp[i*nsec + (j+1)%nsec] -Trp[i*nsec + j])*1.0/(2.0*CUDART_PI_F/nsec) - 0.5*(Trp[i*nsec +j] + Trp[(i-1)*nsec + j]))/ \
    (0.5*(dens[i*nsec +j] + dens[(i-1)*nsec + j]));

  }
}


__global__ void InitComputeAccel(float *CellAbscissa, float *CellOrdinate, float *Rmed, float *cosns, float *sinns, int nsec, int nrad)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = threadIdx.y + blockDim.y*blockIdx.y;

  if (i<nrad && j<nsec)
  {
    CellAbscissa[i*nsec+j]= Rmed[i] * cosns[j];
    CellOrdinate[i*nsec+j]= Rmed[i] * sinns[j];
  }
}


__global__ void ComputeSoundSpeed(float *SoundSpeed, float *dens, float *Rmed, float *energy, int nsec, int nrad,
int Adiabaticc, float ADIABATICINDEX, float FLARINGINDEX, float *AspectRatioRmed)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = threadIdx.y + blockDim.y*blockIdx.y;

  if (i<nrad && j<nsec)
  {
    if (~Adiabaticc){
      SoundSpeed[j+i*nsec] = AspectRatioRmed[i]*sqrtf(1.0*1.0/Rmed[i])*powf(Rmed[i], FLARINGINDEX);
    }
    else SoundSpeed[j+i*nsec] = sqrtf(ADIABATICINDEX*(ADIABATICINDEX-1.0)*energy[j+i*nsec]/dens[j+i*nsec]);
  }
}


__global__ void ComputePressureField(float *SoundSpeed, float *dens, float *press, int Adiabaticc, int nrad, int nsec,
  int ADIABATICINDEX, float *energy)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = threadIdx.y + blockDim.y*blockIdx.y;

  if (i<nrad && j<nsec)
  {
    if (~Adiabaticc){
      press[j+i*nsec] = dens[j+i*nsec]*SoundSpeed[j+i*nsec]*SoundSpeed[j+i*nsec];
    }
    else press[j+i*nsec] = (ADIABATICINDEX-1.0)*energy[j+i*nsec];
  }
}

__global__ void ComputeTemperatureField(float *dens, float *temperature, float *press, float *energy, float MU, float R,
float ADIABATICINDEX, int Adiabaticc, int nsec, int nrad)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = threadIdx.y + blockDim.y*blockIdx.y;

  if (i<nrad && j<nsec)
  {
    if (~Adiabaticc){
      temperature[j+i*nsec] = MU/R*press[j+i*nsec]/dens[j+i*nsec];
    }
    else temperature[j+i*nsec] = MU/R*(ADIABATICINDEX-1.0)*energy[j+i*nsec]/dens[j+i*nsec];
  }
}


__global__ void InitLabel (float *label, float xp, float yp, float rhill, float *Rmed, int nrad, int nsec)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = threadIdx.y + blockDim.y*blockIdx.y;

  if (i<nrad && j<nsec)
  {
    float distance, angle, x, y;
    angle = 2.0*CUDART_PI_F*j/nsec;
    x = Rmed[i] * cosf(angle);
    y = Rmed[i] * sinf(angle);
    distance = sqrtf( (x - xp) * (x - xp) + (y - yp)*(y -yp) );
    if (distance < rhill) label[i] = 1.0;
    else label[i] = 0.0;
  }
}

/*__global__ float CircumPlanetaryMass (float *dens, sys)
{
  int i = threadIdx.x + blockDim.x*blockIdx.x;
  int y = threadIdx.y + blockDim.y*blockIdx.y;

  if (i<nrad && j<nsec)
  {

  }
}*/
