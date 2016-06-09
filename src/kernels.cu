#include "kernels.cuh"

extern int nrad;
extern int nsec;
extern float OmegaFrame;
extern bool ZMPlus;
extern float IMPOSEDDISKDRIFT;
extern float SIGMASLOPE;

__global__ void substep1(float *press, float *rho, float *vradint, float *invdiffRmed, float *pot,
   float *Rinf, float *invRinf, float *vrad, float *vthetaint, float *Rmed, float *vtheta, float dt,
 int nrad, int nsec, float OmegaFrame, bool ZMPlus, float IMPOSEDDISKDRIFT, float SIGMASLOPE)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = threadIdx.y + blockDim.y*blockIdx.y;
  float gradp, gradphi, vt2;

  i+= 1;
  // i=1->nrad , j=0->nsec

  if (i<nrad && j<nsec)
  {
    gradp = 2.0*(press[i*nsec + j] - press[(i-1)*nsec + j])/(rho[i*nsec + j] + rho[(i-1)*nsec + j])*invdiffRmed[i];
    gradphi = (pot[i*nsec + j]-pot[(i-1)*nsec + j])*invdiffRmed[i];
    vt2 = press[i*nsec + j] + press[(i-1)*nsec + j] + press[i*nsec + (j+1)%nsec] + press[(i-1)*nsec + (j+1)%nsec];
    vt2 = vt2/4.0+Rinf[i]*OmegaFrame;
    vradint[i*nsec + j] = vrad[i*nsec + j] + dt*(-gradp - gradphi + vt2*vt2*invRinf[i]);
  }

  i-=1;

  // i=0->nrad , j=0->nsec

  if (i<nrad && j<nsec)
  {
    gradp = 2.0*(press[i*nsec + j] - press[i*nsec + ((j-1)+nsec)%nsec])/(rho[i*nsec +j] +rho[i*nsec + ((j-1)+nsec)%nsec]) \
    *1.0/(2.0*M_PI/nsec*Rmed[i]);
    if (ZMPlus)
    {
      gradp *= 1; //gradp *= SG_aniso_coeff;  Definir mas adelante SG_aniso_coeff
    }

    gradphi = (pot[i*nsec+ j] - pot[i*nsec + ((j-1)+nsec)%nsec])*1.0/(2.0*M_PI/nsec*Rmed[i]);
    vthetaint[i*nsec + j] = vtheta[i*nsec + j] -dt*(gradp+gradphi);
    vthetaint[i*nsec + j] += dt*IMPOSEDDISKDRIFT*0.5*powf(Rmed[i],-2.5+SIGMASLOPE);
  }

}



__global__ void UpdateVelocities(float *vt, float *vr, float *invRmed, float *Rmed, float *Rsup,
  float *Rinf, float *invdiffRmed, float *invdiffRsup, float *rho, float *invRinf, float *Trr,
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
    (Tpp[i*nsec +j]-Tpp[i*nsec + ((j-1)+nsec)%nsec])*1.0/(2.0*M_PI/nsec) + 0.5*(Trp[i*nsec + j] + Trp[(i+1)*nsec +j]))/ \
    (0.5*(rho[i*nsec +j]+rho[i*nsec + ((j-1)+nsec)%nsec]));
  }

  /* now vrad */
  if (i<nrad && j<nsec)
  {
    vr[i*nsec +j] += DeltaT*invRinf[i]*((Rmed[i]*Trr[i*nsec +j]- Rmed[i-1]*Trr[(i-1)*nsec + j])*invdiffRmed[i] + \
    (Trp[i*nsec + (j+1)%nsec] -Trp[i*nsec + j])*1.0/(2.0*M_PI/nsec) - 0.5*(Trp[i*nsec +j] + Trp[(i-1)*nsec + j]))/ \
    (0.5*(rho[i*nsec +j] + rho[(i-1)*nsec + j]));

  }
}
