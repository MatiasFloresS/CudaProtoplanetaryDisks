#include "math_constants.h"
#include <stdio.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


__global__ void substep1(float *press, float *rho, float *vradint, float *invdiffRmed, float *pot,
  float *Rinf, float *invRinf, float *vrad, float *vthetaint, float *Rmed, float *vtheta, float dt,
  int nrad, int nsec, float OmegaFrame, bool ZMPlus, float IMPOSEDDISKDRIFT, float SIGMASLOPE, float *powRmed);

__global__ void substep2(float *dens, float *vradint, float *vthetaint, float *temperatureint,
  int nrad, int nsec, float CVNR, float *invdiffRmed, float *invdiffRsup, float *densint, int Adiabaticc,
  float *Rmed, float dt,    float *vradnew, float *vthetanew, float *energy, float *energyint);

__global__ void substep3(float *dens, float *qplus, float *viscosity_array, float *Trr, float *Trp,float *Tpp,
  float *divergence, int nrad, int nsec, float *Rmed, int Cooling, float *energynew, float dt, float *EnergyMed,
  float *SigmaMed, float *CoolingTimeMed, float *energy, float ADIABATICINDEX, float *QplusMed);

__global__ void UpdateVelocities(float *vt, float *vr, float *invRmed, float *Rmed, float *Rsup,
  float *Rinf, float *invdiffRmed, float *invdiffRsup, float *rho, float *invRinf, float *Trr,
  float *Trp, float *Tpp, float DeltaT, int nrad, int nsec);

__global__ void InitComputeAccel(float *CellAbscissa, float *CellOrdinate, float *Rmed,
  float *cosns, float *sinns, int nsec, int nrad);

__global__ void ComputeSoundSpeed(float *SoundSpeed, float *dens, float *Rmed, float *energy, int nsec, int nrad,
  int Adiabaticc, float ADIABATICINDEX, float FLARINGINDEX, float *AspectRatioRmed);

__global__ void ComputePressureField(float *SoundSpeed, float *dens, float *press, int Adiabaticc, int nrad, int nsec,
  int ADIABATICINDEX, float *energy);

__global__ void ComputeTemperatureField(float *dens, float *temperature, float *press, float *energy, float MU, float R,
  float ADIABATICINDEX, int Adiabaticc, int nsec, int nrad);

__global__ void InitLabel (float *label, float xp, float yp, float rhill, float *Rmed, int nrad, int nsec);

__global__ void CircumPlanetaryMass (float *dens, float *Surf, float *CellAbscissa, float *CellOrdinate, float xpl, float ypl, int nrad,
  int nsec, float HillRadius, float *mdcp0);

__global__ void deviceReduceKernel(float *g_idata, float *g_odata, unsigned int n);

__host__ bool isPow2(unsigned int x);

__host__ long NearestPowerOf2(long n);

__host__ float deviceReduce(float *in, int N) ;

__global__ void MultiplyPolarGridbyConstant(float *dens_d, int nrad, int nsec, float ScalingFactor);

__global__ void ComputeForceKernel(float *CellAbscissa, float *CellOrdinate, float *Surf, float *dens, float x, float y, float rsmoothing,
  float *forcesxi, float *forcesyi, float *forcesxo, float *forcesyo, int nsec, int nrad, float G, float a, float *Rmed,
  int dimfxy, float rh);

__global__ void OpenBoundary(float *vrad, float *dens_d, float *energy_d, int nsec, float *SigmaMed);

__global__ void ReduceCs(float *SoundSpeed, float *cs0, float *cs1, float *csnrm1, float *csnrm2, int nsec, int nrad);

__global__ void ReduceMean(float *dens, float *energy, int nsec, float *mean_dens, float *mean_energy, float *mean_dens2,
  float *mean_energy2, int nrad);

__global__ void NonReflectingBoundary(float *dens, float *energy, int i_angle, int nsec, float *vrad, float *SoundSpeed,
  float SigmaMed, int nrad, float SigmaMed2, int i_angle2);

__global__ void MinusMean(float *dens, float *energy, float SigmaMed, float mean_dens_r, float mean_dens_r2, float mean_energy_r,
  float mean_energy_r2, float EnergyMed, int NSEC, int NRAD, float SigmaMed2, float EnergyMed2);

__global__ void make1Dprofile(float *device_out2, float *gridfield, float *GLOBAL_bufarray, int nsec, int nrad);

__global__ void InitGasVelocities(float *viscosity_array, int nsec, int nrad, int SelfGravity, float *Rmed, float G,
  float ASPECTRATIO, float FLARINGINDEX, float SIGMASLOPE, int CentrifugalBalance, float *vrad, float *vtheta,
  float ViscosityAlpha, float IMPOSEDDISKDRIFT, float SIGMA0, float *SigmaInf, float OmegaFrame, float *Rinf);

__host__ void make1Dprofilehost(float *gridfield);

__global__ void ViscousTerms(float *vrad, float *vtheta , float *Drr, float *Dpp, float *divergence, float *Drp,
  float *invdiffRsup, int invdphi, float *invRmed, float *Rsup, float *Rinf, float *invdiffRmed, int nrad, int nsec,
  float *Trr, float *Tpp, float *dens, float *viscosity_array, float onethird, float *Trp, float *invRinf);
