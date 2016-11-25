
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert (cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void Substep1Kernel (float *Pressure, float *Dens, float *VradInt, double *invdiffRmed, float *Potencial,
  double *Rinf, double *invRinf, float *Vrad, float *VthetaInt, float *Vtheta, double *Rmed, float dt,
  int nrad, int nsec, double OmegaFrame, int ZMPlus, float IMPOSEDDISKDRIFT, float SIGMASLOPE);

__global__ void Substep2Kernel (float *Dens, float *VradInt, float *VthetaInt, float *TemperInt,
  int nrad, int nsec, double *invdiffRmed, double *invdiffRsup, float *DensInt, int Adiabaticc,
  double *Rmed, float dt,  float *VradNew, float *VthetaNew, float *Energy, float *EnergyInt);

__global__ void Substep3Kernel (float *Dens, float *Qplus, float *viscosity_array, float *TAURR, float *TAURP,float *TAUPP,
  float *DivergenceVelocity, int nrad, int nsec, double *Rmed, int Cooling, float *EnergyNew, float dt, float *EnergyMed,
  float *SigmaMed, float *CoolingTimeMed, float *Energy, float ADIABATICINDEX, float *QplusMed);

__global__ void UpdateVelocitiesKernel (float *VthetaInt, float *VradInt, double *invRmed, double *Rmed, double *Rsup,
  double *Rinf, double *invdiffRmed, double *invdiffRsup, float *Dens, double *invRinf, float *TAURR,
  float *TAURP, float *TAUPP, float DeltaT, int nrad, int nsec);

__global__ void InitComputeAccelKernel (float *CellAbscissa, float *CellOrdinate, double *Rmed, int nsec, int nrad);

__global__ void ComputeSoundSpeedKernel (float *SoundSpeed, float *Dens, double *Rmed, float *Energy, int nsec, int nrad,
  int Adiabatic, float ADIABATICINDEX, float FLARINGINDEX, float ASPECTRATIO, float TRANSITIONWIDTH,
  float TRANSITIONRADIUS, float TRANSITIONRATIO, float PhysicalTime, float PhysicalTimeInitial, float LAMBDADOUBLING);

__global__ void ComputePressureFieldKernel (float *SoundSpeed, float *Dens, float *Pressure, int Adiabaticc, int nrad, int nsec,
  float ADIABATICINDEX, float *Energy);

__global__ void ComputeTemperatureFieldKernel (float *Dens, float *Temperature, float *Pressure, float *Energy,
  float ADIABATICINDEX, int Adiabaticc, int nsec, int nrad);

__global__ void InitLabelKernel (float *Label, float xp, float yp, float rhill, double *Rmed, int nrad, int nsec);

__global__ void CircumPlanetaryMassKernel (float *Dens, double *Surf, float *CellAbscissa, float *CellOrdinate, float xpl, float ypl, int nrad,
  int nsec, float HillRadius, float *mdcp0);

__global__ void DeviceReduceKernel (float *g_idata, float *g_odata, unsigned int n);

__host__ bool IsPow2 (unsigned int x);

__host__ long NearestPowerOf2 (long n);

__host__ float DeviceReduce (float *in, int N) ;

__global__ void MultiplyPolarGridbyConstantKernel (float *Dens, int nrad, int nsec, float ScalingFactor);

__global__ void ComputeForceKernel (float *CellAbscissa, float *CellOrdinate, double *Surf, float *Dens, float x,
  float y, float rsmoothing, int nsec, int nrad, float a, double *Rmed, int dimfxy, float rh, float *fxi,
  float *fxo, float *fyi, float *fyo, int k);

__global__ void OpenBoundaryKernel (float *Vrad, float *Dens, float *Energy, int nsec, float *SigmaMed);

__global__ void ReduceCsKernel (float *SoundSpeed, float *cs0, float *cs1, float *csnrm1, float *csnrm2, int nsec, int nrad);

__global__ void ReduceMeanKernel (float *Dens, float *Energy, int nsec, float *mean_dens, float *mean_energy, float *mean_dens2,
  float *mean_energy2, int nrad);

__global__ void NonReflectingBoundaryKernel (float *Dens, float *energy, int i_angle, int nsec, float *Vrad, float *SoundSpeed,
  float SigmaMed, int nrad, float SigmaMed2, int i_angle2);

__global__ void MinusMeanKernel (float *Dens, float *Energy, float SigmaMed, float mean_dens_r, float mean_dens_r2, float mean_energy_r,
  float mean_energy_r2, float EnergyMed, int NSEC, int NRAD, float SigmaMed2, float EnergyMed2);

__global__ void Make1DprofileKernel (float *device_out2, float *gridfield, float *axifield, int nsec, int nrad);

__global__ void InitGasVelocitiesKernel (float *viscosity_array, int nsec, int nrad, int SelfGravity, double *Rmed,
  float ASPECTRATIO, float FLARINGINDEX, float SIGMASLOPE, int CentrifugalBalance, float *Vrad, float *Vtheta,
  float ViscosityAlpha, float IMPOSEDDISKDRIFT, float SIGMA0, float *SigmaInf, double OmegaFrame, double *Rinf, float *vt_cent);

__host__ void Make1Dprofile (int option);

__global__ void ViscousTermsKernel (float *Vradial, float *Vazimutal , float *DRR, float *DPP, float *DivergenceVelocity, float *DRP,
  double *invdiffRsup, double *invRmed, double *Rsup, double *Rinf, double *invdiffRmed, int nrad, int nsec,
  float *TAURR, float *TAUPP, float *Dens, float *viscosity_array, float *TAURP, double *invRinf);

__global__ void LRMomentaKernel (float *RadMomP, float *RadMomM, float *ThetaMomP, float *ThetaMomM, float *Dens,
  float *Vrad, float *Vtheta, int nrad, int nsec, double *Rmed, double OmegaFrame);

__global__ void ExtQtyKernel (float *ExtLabel, float *Dens, float *Label, int nsec, int nrad);

__global__ void StarRadKernel (float *Qbase2, float *Vrad, float *QStar, float dt, int nrad, int nsec,
  double *invdiffRmed, double *Rmed, float *dq);


/* SG_ACCELERATION Kernels */
__global__ void ComputeFFTKernel (double *Radii, cufftComplex *SGP_Kr, cufftComplex *SGP_Kt, float SGP_eps, int nrad, int nsec,
  cufftComplex *SGP_Sr, cufftComplex *SGP_St, float *Dens, double *Rmed, float *Kr_aux, float *Kt_aux);

__global__ void ComputeConvolutionKernel (cufftComplex *Gr, cufftComplex *Gphi, cufftComplex *SGP_Kr, cufftComplex *SGP_Kt,
  cufftComplex *SGP_Sr, cufftComplex *SGP_St, int nsec, int nrad);

__global__ void ComputeSgAccKernel (float *SG_Accr, float *SG_Acct, float *Dens , float SGP_rstep, float SGP_tstep,
  float SGP_eps, int nrad, int nsec, double *Rmed, cufftComplex *Gr, cufftComplex *Gphi);

__global__ void Update_sgvelocityKernel (float *Vradial, float *Vazimutal, float *SG_Accr, float *SG_Acct, double *Rinf,
  double *Rmed, double *invdiffRmed, float dt, int nrad, int nsec);


__global__ void Azimutalvelocity_withSGKernel (float *Vtheta, double *Rmed, float FLARINGINDEX, float SIGMASLOPE,
  float ASPECTRATIO, float *GLOBAL_bufarray, int nrad, int nsec);

__global__ void CrashKernel (float *array, int NRAD, int NSEC, int Crash);

__global__ void EvanescentBoundaryKernel(double *Rmed, float *Vrad, float *Vtheta, float *Energy, float *Dens,
  float *viscosity_array, float DRMIN, float DRMAX, int nrad, int nsec, float Tin,
  float Tout, float step, float SIGMASLOPE, float FLARINGINDEX, float *GLOBAL_bufarray, double OmegaFrame,
  float *SigmaMed, float *EnergyMed, int Adiabatic, int SelfGravity, float ASPECTRATIO, float TRANSITIONWIDTH,
  float TRANSITIONRADIUS, float TRANSITIONRATIO, float PhysicalTime, float PhysicalTimeInitial, float LAMBDADOUBLING);

__global__ void DivisePolarGridKernel (float *Qbase, float *DensInt, float *Work, int nrad, int nsec);

__global__ void  VanLeerRadialKernel (double *Rinf, double *Rsup, float *QRStar, float *DensStar, float *Vrad,
  float *LostByDisk, int nsec, int nrad, float dt, int OpenInner,  float *Qbase, double *invSurf);

__global__ void ComputeAverageThetaVelocitiesKernel(float *Vtheta, float *VMed, int nsec, int nrad);

__global__ void ComputeResidualsKernel (float *VthetaRes, float *VMed, int nsec, int nrad, float *Vtheta);

__global__ void ComputeConstantResidualKernel (float *VMed, double *invRmed, long *Nshift, int *NoSplitAdvection,
  int nsec, int nrad, float dt, float *Vtheta, float *VthetaRes, double *Rmed, int FastTransport);

__global__ void StarThetaKernel (float *Qbase, double *Rmed, float *Vazimutal, float *QStar, int nrad, int nsec,
  float *dq, float dt);

__global__ void VanLeerThetaKernel (double *Rsup, double *Rinf, double *Surf, float dt, int nrad, int nsec,
  int UniformTransport, int *NoSplitAdvection, float *QRStar, float *DensStar, float *Vtheta, float *Qbase);

__global__ void AdvectSHIFTKernel(float *array, float *TempShift, int nsec, int nrad, long *Nshift);

__global__ void ComputeVelocitiesKernel(float *Vrad, float *Vtheta, float *Dens, double *Rmed, float *ThetaMomP,
  float *ThetaMomM, float *RadMomP, float *RadMomM, int nrad, int nsec, double OmegaFrame);

__global__ void ComputeSpeQtyKernel (float *Label, float *Dens, float *ExtLabel, int nrad, int nsec);

__global__ void FillForcesArraysKernel (double *Rmed, int nsec, int nrad, double xplanet, double yplanet, double smooth,
  double mplanet, int Indirect_Term, double InvPlanetDistance3, float *Potential, pair IndirectTerm, int k);

__global__ void CorrectVthetaKernel (float *Vtheta, float domega, double *Rmed, int nrad, int nsec);

__global__ void ConditionCFLKernel1D (double *Rsup, double *Rinf, double *Rmed, int nrad, int nsec,
  float *Vtheta, float *Vmoy);

__device__ float max2(float a, float b);

__device__ float min2(float a, float b);

__device__ float AspectRatioDevice(float r, float ASPECTRATIO, float TRANSITIONWIDTH, float TRANSITIONRADIUS,
  float TRANSITIONRATIO, float PhysicalTime, float PhysicalTimeInitial, float LAMBDADOUBLING);

__global__ void ConditionCFLKernel2D1 (double *Rsup, double *Rinf, double *Rmed, int nsec, int nrad,
  float *Vresidual, float *Vtheta, float *Vmoy, int FastTransport, float *SoundSpeed, float *Vrad,
  float *DT2D);

__global__ void ConditionCFLKernel2D2 (float *newDT, float *DT2D, float *DT1D, float *Vmoy, double *invRmed,
  int *CFL, int nsec, int nrad, float DeltaT);

__global__ void ApplySubKeplerianBoundaryKernel(float *VthetaInt, double *Rmed, double OmegaFrame, int nsec,
  int nrad, float VKepIn, float VKepOut);

__device__ inline void MyAtomicAdd (float *address, float value);

__global__ void kernel(float *Dens, float *VradInt, float *VthetaInt, float *TemperInt, int nrad,
  int nsec, double *invdiffRmed, double *invdiffRsup, float *DensInt, int Adiabatic, double *Rmed,
  float dt, float *VradNew, float *VthetaNew, float *Energy, float *EnergyInt);
