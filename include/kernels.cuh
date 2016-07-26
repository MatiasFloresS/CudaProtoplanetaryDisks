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
    int nrad, int nsec, float CVNR, float *invdiffRmed, float *invdiffRsup, float *densint, int Adiabaticc, float *Rmed, float dt,
    float *vradnew, float *vthetanew, float *energy, float *energyint);

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

__global__ void MultiplyPolarGridbyConstant(float *dens_d, float *fieldsrc_d, int nrad, int nsec, float ScalingFactor);

__global__ void ComputeForceKernel(float *CellAbscissa, float *CellOrdinate, float *Surf, float *dens, float x, float rsmoothing,
  int dimfxy, float mass, float a, float *fxi, float *fxo, float *fyi, float *fyo, float *Rmed);

__global__ void OpenBoundary(float *vrad, float *dens_d, float *energy_d, int nsec, float *SigmaMed);

__global__ void ReduceCs(float *SoundSpeed, float *cs0, float *cs1, int nsec);

__global__ void ReduceMean(float *dens, float *energy, int nsec, float *mean_dens, float *mean_energy);

__global__ void NonReflectingBoundary(float *dens, float *energy, int i_angle, int nsec, float *vrad, float *SoundSpeed,
  float SigmaMed);
