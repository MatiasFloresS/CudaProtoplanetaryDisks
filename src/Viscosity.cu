#include "Viscosity.cuh"


/*
__global__ void UpdateVelocities(float *vt, float *vr, float *invRmed, float *Rmed, float *Rsup,
  float *Rinf, float *invdiffRmed, float *invdiffRsup, float *rho, float *invRinf, float *Trr,
  float *Trp, float *Tpp, float DeltaT)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = threadIdx.y + blockDim.y*blockIdx.y;

  i+=1;
  /* Now we can update velocities
     with the viscous source term
     of Navier-Stokes equation */

  /* vtheta first *//*
  if (i<nrad-1 && j<nsec)
  {
    vt[i*nsec +j] += DeltaT*invRmed[i]*((Rsup[i]*Trp[(i+1)*nsec+ j]-Rinf[i]*Trp[i*nsec +j])*invdiffRsup[i] + \
    (Tpp[i*nsec +j]-Tpp[i*nsec + ((j-1)+nsec)%nsec])*1.0/(2.0*M_PI/nsec*) + 0.5*(Trp[i*nsec + j] + Trp[(i+1)*nsec +j]))/ \
    (0.5*(rho[i*nsec +j]+rho[i*nsec + ((j-1)+nsec)%nsec]));
  }

  /* now vrad *//*
  if (i<nrad && j<nsec)
  {
    vr[i*nsec +j] += DeltaT*invRinf[i]*((Rmed[i]*Trr[i*nsec +j]- Rmed[i-1]*Trr[(i-1)*nsec + j])*invdiffRmed[i] + \
    (Trp[i*nsec + (j+1)%nsec] -Trp[i*nsec + j])*1.0/(2.0*M_PI/nsec*) - 0.5*(Trp[i*nsec +j] + Trp[(i-1)*nsec + j]))/ \
    (0.5*(rho[i*nsec +j] + rho[(i-1)*nsec + j]);

  }
}
*/

__host__ void UpdateVelocitiesWithViscosity(float *RadialVelocity, float *AzimuthalVelocity, float *Rho, float DeltaT)
{

  printf("%f\n", nsec);
}
