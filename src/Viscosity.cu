#include "main.cuh"
#include "kernels.cuh"

using namespace std;

extern int nsec;
extern float TRANSITIONWIDTH, TRANSITIONRADIUS, TRANSITIONRATIO, ASPECTRATIO, LAMBDADOUBLING;
extern float VISCOSITY, ViscosityAlpha, *Rmed, CAVITYRATIO, CAVITYRADIUS, CAVITYWIDTH, *GLOBAL_bufarray, ALPHAVISCOSITY;
float PhysicalTime =0.0, PhysicalTimeInitial= 0.0;

__host__ void UpdateVelocitiesWithViscosity(float *RadialVelocity, float *AzimuthalVelocity, float *Rho, float DeltaT)
{

  /*UpdateVelocities<<<dimGrid, dimBlock>>>(vt_d,vr_d,invRmed_d,Rmed_d, Rsup_d, Rinf_d,
    invdiffRmed_d, invdiffSup_d,  rho_d, invRinf_d, Trr_d, Tpp_d, DeltaT, nrad, nsec);

    float *vt, float *vr, float *invRmed, float *Rmed, float *Rsup,
      float *Rinf, float *invdiffRmed, float *invdiffRsup, float *rho, float *invRinf, float *Trr,
      float *Trp, float *Tpp, float DeltaT, int nrad, int nsec*/
}

__host__ float AspectRatio(float r)
{
  float aspectratio, rmin, rmax, scale;
  aspectratio = ASPECTRATIO;
  rmin = TRANSITIONRADIUS-TRANSITIONWIDTH*ASPECTRATIO;
  rmax = TRANSITIONRADIUS+TRANSITIONWIDTH*ASPECTRATIO;
  scale = 1.0+(PhysicalTime-PhysicalTimeInitial)*LAMBDADOUBLING;
  rmin *= scale;
  rmax *= scale;
  if (r < rmin) aspectratio *= TRANSITIONRATIO;
  if ((r >= rmin) && (r <= rmax)) {
    aspectratio *= expf((rmax-r)/(rmax-rmin)*logf(TRANSITIONRATIO));
  }
  return aspectratio;
}

__host__ float FViscosity(float r)
{
  float viscosity, rmin, rmax, scale;
  int i = 0;
  viscosity = VISCOSITY;
  if (ViscosityAlpha)
  {
    while (Rmed[i] < r) i++;
    viscosity = ALPHAVISCOSITY*GLOBAL_bufarray[i] * GLOBAL_bufarray[i] * powf(r, 1.5);
  }
  rmin = CAVITYRADIUS-CAVITYWIDTH*ASPECTRATIO;
  rmax = CAVITYRADIUS+CAVITYWIDTH*ASPECTRATIO;
  scale = 1.0+(PhysicalTime-PhysicalTimeInitial)*LAMBDADOUBLING;
  rmin *= scale;
  rmax *= scale;
  if (r < rmin) viscosity *= CAVITYRATIO;
  if ((r >= rmin) && (r <= rmax)) viscosity *= expf((rmax-r)/(rmax-rmin)*logf(CAVITYRATIO));
  return viscosity;
}
