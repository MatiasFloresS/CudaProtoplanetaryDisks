#include "math_constants.h"

__global__ void substep1(float *press, float *rho, float *vradint, float *invdiffRmed, float *pot,
  float *Rinf, float *invRinf, float *vrad, float *vthetaint, float *Rmed, float *vtheta, float dt,
  int nrad, int nsec, float OmegaFrame, bool ZMPlus, float IMPOSEDDISKDRIFT, float SIGMASLOPE, float *powRmed);

__global__ void UpdateVelocities(float *vt, float *vr, float *invRmed, float *Rmed, float *Rsup,
  float *Rinf, float *invdiffRmed, float *invdiffRsup, float *rho, float *invRinf, float *Trr,
  float *Trp, float *Tpp, float DeltaT, int nrad, int nsec);

__global__ void InitComputeAccel(float *CellAbscissa, float *CellOrdinate, float *Rmed,
   float *cosns, float *sinns, int nsec, int nrad);

__global__ void ComputeSoundSpeed(float *SoundSpeed, float *dens, float *Rmed, float *energy, int nsec, int nrad,
   int Adiabaticc, float ADIABATICINDEX, float FLARINGINDEX, float *AspectRatioRmed);

__global__ void ComputePressureField(float *SoundSpeed, float *dens, float *press, int Adiabaticc, int nrad, int nsec,
   int ADIABATICINDEX, float *energy);
