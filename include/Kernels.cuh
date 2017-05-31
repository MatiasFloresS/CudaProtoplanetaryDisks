
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert (cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void Substep1Kernel (double *Pressure, double *Dens, double *VradInt, double *invdiffRmed, double *Potenial,
  double *Rinf, double *invRinf, double *Vrad, double *VthetaInt, double *Vtheta, double *Rmed, double dt,
  int nrad, int nsec, double OmegaFrame, int ZMPlus, double IMPOSEDDISKDRIFT, double SIGMASLOPE);

__global__ void Substep2Kernel (double *Dens, double *VradInt, double *VthetaInt, double *TemperInt,
  int nrad, int nsec, double *invdiffRmed, double *invdiffRsup, double *DensInt, int Adiabaticc,
  double *Rmed, double dt,  double *VradNew, double *VthetaNew, double *Energy, double *EnergyInt);

__global__ void Substep3Kernel (double *Dens, double *Qplus, double *viscosity_array, double *TAURR, double *TAURP,double *TAUPP,
  double *DivergenceVelocity, int nrad, int nsec, double *Rmed, int Cooling, double *EnergyNew, double dt, double *EnergyMed,
  double *SigmaMed, double *CoolingTimeMed, double *EnergyInt, double ADIABATICINDEX, double *QplusMed);

__global__ void Substep3Kernel2 (double *Dens, double *Qplus, double *viscosity_array, double *TAURR, double *TAURP,double *TAUPP,
  double *DivergenceVelocity, int nrad, int nsec, double *Rmed, int Cooling, double *EnergyNew, double dt, double *EnergyMed,
  double *SigmaMed, double *CoolingTimeMed, double *EnergyInt, double ADIABATICINDEX, double *QplusMed);



__global__ void UpdateVelocitiesKernel (double *VthetaInt, double *VradInt, double *invRmed, double *Rmed, double *Rsup,
  double *Rinf, double *invdiffRmed, double *invdiffRsup, double *Dens, double *invRinf, double *TAURR,
  double *TAURP, double *TAUPP, double DeltaT, int nrad, int nsec);

__global__ void InitComputeAccelKernel (double *CellAbscissa, double *CellOrdinate, double *Rmed, int nsec, int nrad);

__global__ void ComputeSoundSpeedKernel (double *SoundSpeed, double *Dens, double *Rmed, double *Energy, int nsec, int nrad,
  int Adiabatic, double ADIABATICINDEX, double FLARINGINDEX, double ASPECTRATIO, double TRANSITIONWIDTH,
  double TRANSITIONRADIUS, double TRANSITIONRATIO, double PhysicalTime, double PhysicalTimeInitial, double LAMBDADOUBLING);

__global__ void ComputePressureFieldKernel (double *SoundSpeed, double *Dens, double *Pressure, int Adiabaticc, int nrad, int nsec,
  double ADIABATICINDEX, double *Energy);

__global__ void ComputeTemperatureFieldKernel (double *Dens, double *Temperature, double *Pressure, double *Energy,
  double ADIABATICINDEX, int Adiabaticc, int nsec, int nrad);

__global__ void InitLabelKernel (double *Label, double xp, double yp, double rhill, double *Rmed, int nrad, int nsec);

__global__ void CircumPlanetaryMassKernel (double *Dens, double *Surf, double *CellAbscissa, double *CellOrdinate, double xpl, double ypl, int nrad,
  int nsec, double HillRadius, double *mdcp0);

__global__ void DeviceReduceKernel (double *g_idata, double *g_odata, unsigned int n);

__host__ bool IsPow2 (unsigned int x);

__host__ long NearestPowerOf2 (long n);

__host__ double DeviceReduce (double *in, int N) ;

__global__ void MultiplyPolarGridbyConstantKernel (double *Dens, int nrad, int nsec, double ScalingFactor);

__global__ void ComputeForceKernel (double *CellAbscissa, double *CellOrdinate, double *Surf, double *Dens, double x,
  double y, double rsmoothing, int nsec, int nrad, double a, double *Rmed, int dimfxy, double rh, double *fxi,
  double *fxo, double *fyi, double *fyo, int k);

__global__ void OpenBoundaryKernel (double *Vrad, double *Dens, double *Energy, int nsec, double SigmaMed);

__global__ void ReduceCsKernel (double *SoundSpeed, double *cs0, double *cs1, double *csnrm1, double *csnrm2, int nsec, int nrad);

__global__ void ReduceMeanKernel (double *Dens, double *Energy, int nsec, double *mean_dens, double *mean_energy, double *mean_dens2,
  double *mean_energy2, int nrad);

__global__ void NonReflectingBoundaryKernel (double *Dens, double *energy, int i_angle, int nsec, double *Vrad, double *SoundSpeed,
  double SigmaMed, int nrad, double SigmaMed2, int i_angle2);

__global__ void NonReflectingBoundaryKernel2 (double *Dens, double *energy, int i_angle, int nsec, double *Vrad, double *SoundSpeed,
  double SigmaMed, int nrad, double SigmaMed2, int i_angle2);

__global__ void MinusMeanKernel (double *Dens, double *Energy, double SigmaMed, double mean_dens_r, double mean_dens_r2, double mean_energy_r,
  double mean_energy_r2, double EnergyMed, int NSEC, int NRAD, double SigmaMed2, double EnergyMed2);

__global__ void Make1DprofileKernel (double *device_out2, double *gridfield, double *axifield, int nsec, int nrad);

__global__ void InitGasVelocitiesKernel (int nsec, int nrad, int SelfGravity, double *Rmed,
  double ASPECTRATIO, double FLARINGINDEX, double SIGMASLOPE, double *Vrad, double *Vtheta,
  double IMPOSEDDISKDRIFT, double SIGMA0, double *SigmaInf, double OmegaFrame, double *Rinf, int ViscosityAlpha, double *viscosity_array);

__host__ void Make1Dprofile (int option);

__global__ void ViscousTermsKernel (double *Vradial, double *Vazimutal , double *DRR, double *DPP, double *DivergenceVelocity,
  double *DRP, double *invdiffRsup, double *invRmed, double *Rsup, double *Rinf, double *invdiffRmed, int nrad,
  int nsec, double *TAURR, double *TAUPP, double *dens, double *TAURP, double *invRinf, double *Rmed, double *viscosity_array_d);

__global__ void LRMomentaKernel (double *RadMomP, double *RadMomM, double *ThetaMomP, double *ThetaMomM, double *Dens,
  double *Vrad, double *Vtheta, int nrad, int nsec, double *Rmed, double OmegaFrame);

__global__ void ExtQtyKernel (double *ExtLabel, double *Dens, double *Label, int nsec, int nrad);

__global__ void StarRadKernel (double *Qbase2, double *Vrad, double *QStar, double dt, int nrad, int nsec,
  double *invdiffRmed, double *Rmed, double *dq);

__global__ void StarRadKernel2 (double *Qbase2, double *Vrad, double *QStar, double dt, int nrad, int nsec,
  double *invdiffRmed, double *Rmed, double *dq);

/* SG_ACCELERATION Kernels */
__global__ void ComputeFFTKernel (double *Radii, cufftDoubleComplex *SGP_Kr, cufftDoubleComplex *SGP_Kt, double SGP_eps, int nrad, int nsec,
  cufftDoubleComplex *SGP_Sr, cufftDoubleComplex *SGP_St, double *Dens, double *Rmed, double *Kr_aux, double *Kt_aux);

__global__ void ComputeConvolutionKernel (cufftDoubleComplex *Gr, cufftDoubleComplex *Gphi, cufftDoubleComplex *SGP_Kr, cufftDoubleComplex *SGP_Kt,
  cufftDoubleComplex *SGP_Sr, cufftDoubleComplex *SGP_St, int nsec, int nrad);

__global__ void ComputeSgAccKernel (double *SG_Accr, double *SG_Acct, double *Dens , double SGP_rstep, double SGP_tstep,
  double SGP_eps, int nrad, int nsec, double *Rmed, cufftDoubleComplex *Gr, cufftDoubleComplex *Gphi);

__global__ void Update_sgvelocityKernel (double *Vradial, double *Vazimutal, double *SG_Accr, double *SG_Acct, double *Rinf,
  double *Rmed, double *invdiffRmed, double dt, int nrad, int nsec);


__global__ void Azimutalvelocity_withSGKernel (double *Vtheta, double *Rmed, double FLARINGINDEX, double SIGMASLOPE,
  double ASPECTRATIO, double *axifield_d, int nrad, int nsec);

__global__ void CrashKernel (double *array, int NRAD, int NSEC, int Crash);

__global__ void EvanescentBoundaryKernel(double *Rmed, double *Vrad, double *Vtheta, double *Energy, double *Dens,
  double *viscosity_array, double DRMIN, double DRMAX, int nrad, int nsec, double Tin,
  double Tout, double step, double SIGMASLOPE, double FLARINGINDEX, double *GLOBAL_bufarray, double OmegaFrame,
  double *SigmaMed, double *EnergyMed, int Adiabatic, int SelfGravity, double ASPECTRATIO, double TRANSITIONWIDTH,
  double TRANSITIONRADIUS, double TRANSITIONRATIO, double PhysicalTime, double PhysicalTimeInitial, double LAMBDADOUBLING);

__global__ void DivisePolarGridKernel (double *Qbase, double *DensInt, double *Work, int nrad, int nsec);

__global__ void  VanLeerRadialKernel (double *Rinf, double *Rsup, double *QRStar, double *DensStar, double *Vrad,
  double *LostByDisk, int nsec, int nrad, double dt, int OpenInner,  double *Qbase, double *invSurf);

__global__ void ComputeAverageThetaVelocitiesKernel(double *Vtheta, double *VMed, int nsec, int nrad);

__global__ void ComputeResidualsKernel (double *VthetaRes, double *VMed, int nsec, int nrad, double *Vtheta);

__global__ void ComputeConstantResidualKernel (double *VMed, double *invRmed, int *Nshift, int *NoSplitAdvection,
  int nsec, int nrad, double dt, double *Vtheta, double *VthetaRes, double *Rmed, int FastTransport);

__global__ void StarThetaKernel (double *Qbase, double *Rmed, int nrad, int nsec, double *dq, double dt);

__global__ void StarThetaKernel2 (double *Qbase, double *Rmed, double *Vazimutal, double *QStar, int nrad, int nsec,
  double *dq, double dt);

__global__ void VanLeerThetaKernel (double *Rsup, double *Rinf, double *Surf, double dt, int nrad, int nsec,
  int UniformTransport, int *NoSplitAdvection, double *QRStar, double *DensStar, double *Vtheta, double *Qbase);

__global__ void AdvectSHIFTKernel(double *array, double *TempShift, int nsec, int nrad, int *Nshift);

__global__ void ComputeVelocitiesKernel(double *Vrad, double *Vtheta, double *Dens, double *Rmed, double *ThetaMomP,
  double *ThetaMomM, double *RadMomP, double *RadMomM, int nrad, int nsec, double OmegaFrame);

__global__ void ComputeSpeQtyKernel (double *Label, double *Dens, double *ExtLabel, int nrad, int nsec);

__global__ void FillForcesArraysKernel (double *Rmed, int nsec, int nrad, double xplanet, double yplanet, double smooth,
  double mplanet, int Indirect_Term, double InvPlanetDistance3, double *Potential, pair IndirectTerm, int k);

__global__ void CorrectVthetaKernel (double *Vtheta, double domega, double *Rmed, int nrad, int nsec);

__global__ void ConditionCFLKernel1D (double *Rsup, double *Rinf, double *Rmed, int nrad, int nsec,
  double *Vtheta, double *Vmoy);

__device__ double max2(double a, double b);

__device__ double min2(double a, double b);

__device__ double AspectRatioDevice(double r, double ASPECTRATIO, double TRANSITIONWIDTH, double TRANSITIONRADIUS,
  double TRANSITIONRATIO, double PhysicalTime, double PhysicalTimeInitial, double LAMBDADOUBLING);

__global__ void ConditionCFLKernel2D1 (double *Rsup, double *Rinf, double *Rmed, int nsec, int nrad,
  double *Vresidual, double *Vtheta, double *Vmoy, int FastTransport, double *SoundSpeed, double *Vrad,
  double *DT2D);

__global__ void ConditionCFLKernel2D2 (double *newDT, double *DT2D, double *DT1D, double *Vmoy, double *invRmed,
  int *CFL, int nsec, int nrad, double DeltaT);

__global__ void ConditionCFLKernel2D3 (double *newDT, double *DT2D, double *DT1D, double *Vmoy, double *invRmed,
    int *CFL, int nsec, int nrad, double DeltaT);

__global__ void ApplySubKeplerianBoundaryKernel(double *VthetaInt, double *Rmed, double OmegaFrame, int nsec,
  int nrad, double VKepIn, double VKepOut);

__device__ inline void MyAtomicAdd (double *address, double value);

__global__ void kernel(double *Dens, double *VradInt, double *VthetaInt, double *TemperInt, int nrad,
  int nsec, double *invdiffRmed, double *invdiffRsup, double *DensInt, int Adiabatic, double *Rmed,
  double dt, double *VradNew, double *VthetaNew, double *Energy, double *EnergyInt);

__device__ double FViscosityDevice(double r, double VISCOSITY, int ViscosityAlpha, double *Rmed, double ALPHAVISCOSITY,
  double CAVITYWIDTH, double CAVITYRADIUS, double CAVITYRATIO, double PhysicalTime, double PhysicalTimeInitial,
  double ASPECTRATIO, double LAMBDADOUBLING);
