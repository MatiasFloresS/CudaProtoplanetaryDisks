
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert (cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void Substep1Kernel (float *Pressure, float *Dens, float *VradInt, float *invdiffRmed, float *Potencial,
  float *Rinf, float *invRinf, float *Vrad, float *VthetaInt, float *Rmed, float *Vtheta, float dt,
  int nrad, int nsec, float OmegaFrame, bool ZMPlus, float IMPOSEDDISKDRIFT, float SIGMASLOPE, float *powRmed);

__global__ void Substep2Kernel (float *Dens, float *VradInt, float *VthetaInt, float *TemperInt,
  int nrad, int nsec, float CVNR, float *invdiffRmed, float *invdiffRsup, float *DensInt, int Adiabaticc,
  float *Rmed, float dt,  float *VradNew, float *VthetaNew, float *energy, float *energyInt);

__global__ void Substep3Kernel (float *Dens, float *Qplus, float *viscosity_array, float *TAURR, float *TAURP,float *TAUPP,
  float *DivergenceVelocity, int nrad, int nsec, float *Rmed, int Cooling, float *energyNew, float dt, float *EnergyMed,
  float *SigmaMed, float *CoolingTimeMed, float *energy, float ADIABATICINDEX, float *QplusMed);

__global__ void UpdateVelocitiesKernel (float *VthetaInt, float *VradInt, float *invRmed, float *Rmed, float *Rsup,
  float *Rinf, float *invdiffRmed, float *invdiffRsup, float *Dens, float *invRinf, float *TAURR,
  float *TAURP, float *TAUPP, float DeltaT, int nrad, int nsec);

__global__ void InitComputeAccelKernel (float *CellAbscissa, float *CellOrdinate, float *Rmed, int nsec, int nrad);

__global__ void ComputeSoundSpeedKernel (float *SoundSpeed, float *Dens, float *Rmed, float *energy, int nsec, int nrad,
  int Adiabaticc, float ADIABATICINDEX, float FLARINGINDEX, float *AspectRatioRmed);

__global__ void ComputePressureFieldKernel (float *SoundSpeed, float *Dens, float *Pressure, int Adiabaticc, int nrad, int nsec,
  int ADIABATICINDEX, float *energy);

__global__ void ComputeTemperatureFieldKernel (float *Dens, float *Temperature, float *Pressure, float *energy, float MU, float R,
  float ADIABATICINDEX, int Adiabaticc, int nsec, int nrad);

__global__ void InitLabelKernel (float *label, float xp, float yp, float rhill, float *Rmed, int nrad, int nsec);

__global__ void CircumPlanetaryMassKernel (float *Dens, float *Surf, float *CellAbscissa, float *CellOrdinate, float xpl, float ypl, int nrad,
  int nsec, float HillRadius, float *mdcp0);

__global__ void DeviceReduceKernel (float *g_idata, float *g_odata, unsigned int n);

__host__ bool IsPow2 (unsigned int x);

__host__ long NearestPowerOf2 (long n);

__host__ float DeviceReduce (float *in, int N) ;

__global__ void MultiplyPolarGridbyConstantKernel (float *Dens, int nrad, int nsec, float ScalingFactor);

__global__ void ComputeForceKernel (float *CellAbscissa, float *CellOrdinate, float *Surf, float *Dens, float x, float y, float rsmoothing,
  float *forcesxi, float *forcesyi, float *forcesxo, float *forcesyo, int nsec, int nrad, float G, float a, float *Rmed,
  int dimfxy, float rh);

__global__ void OpenBoundaryKernel (float *Vrad, float *Dens, float *energy, int nsec, float *SigmaMed);

__global__ void ReduceCsKernel (float *SoundSpeed, float *cs0, float *cs1, float *csnrm1, float *csnrm2, int nsec, int nrad);

__global__ void ReduceMeanKernel (float *Dens, float *energy, int nsec, float *mean_dens, float *mean_energy, float *mean_dens2,
  float *mean_energy2, int nrad);

__global__ void NonReflectingBoundaryKernel (float *Dens, float *energy, int i_angle, int nsec, float *Vrad, float *SoundSpeed,
  float SigmaMed, int nrad, float SigmaMed2, int i_angle2);

__global__ void MinusMeanKernel (float *Dens, float *energy, float SigmaMed, float mean_dens_r, float mean_dens_r2, float mean_energy_r,
  float mean_energy_r2, float EnergyMed, int NSEC, int NRAD, float SigmaMed2, float EnergyMed2);

__global__ void Make1DprofileKernel (float *device_out2, float *gridfield, float *GLOBAL_bufarray, int nsec, int nrad);

__global__ void InitGasVelocitiesKernel (float *viscosity_array, int nsec, int nrad, int SelfGravity, float *Rmed, float G,
  float ASPECTRATIO, float FLARINGINDEX, float SIGMASLOPE, int CentrifugalBalance, float *Vrad, float *Vtheta,
  float ViscosityAlpha, float IMPOSEDDISKDRIFT, float SIGMA0, float *SigmaInf, float OmegaFrame, float *Rinf, float *vt_cent);

__host__ void Make1Dprofile (float *gridfield);

__global__ void ViscousTermsKernel (float *Vradial, float *Vazimutal , float *DRR, float *DPP, float *DivergenceVelocity, float *DRP,
  float *invdiffRsup, int invdphi, float *invRmed, float *Rsup, float *Rinf, float *invdiffRmed, int nrad, int nsec,
  float *TAURR, float *TAUPP, float *Dens, float *viscosity_array, float onethird, float *TAURP, float *invRinf);

__global__ void LRMomentaKernel (float *RadMomP, float *RadMomM, float *ThetaMomP, float *ThetaMomM, float *Dens,
  float *Vrad, float *Vtheta, int nrad, int nsec, float *Rmed, float OmegaFrame);

__global__ void ExtQtyKernel (float *Extlabel, float *Dens, float *label, int nsec, int nrad);

__global__ void StarRadKernel (float *Qbase, float *Vrad, float *QStar, float dt, int nrad, int nsec,
  float *invdiffRmed, float *Rmed, float *dq);

__global__ void FftKernel (float *Radii, cufftComplex *SGP_Kr, cufftComplex *SGP_Kt, float SGP_eps, int nrad, int nsec,
  cufftComplex *SGP_Sr, cufftComplex *SGP_St, float *Dens, float *Rmed, float *Kr_aux, float *Kt_aux);

__global__ void FftmulKernel (cufftComplex *Gr, cufftComplex *Gphi, cufftComplex *SGP_Kr, cufftComplex *SGP_Kt,
  cufftComplex *SGP_Sr, cufftComplex *SGP_St, int nsec, float G, int nrad);

__global__ void Sg_AccKernel (float *SG_Accr, float *SG_Acct, float *Dens , float SGP_rstep, float SGP_tstep,
  float SGP_eps, int nrad, int nsec, float *Rmed, cufftComplex *Gr, cufftComplex *Gphi, float G);

__global__ void Update_sgvelocityKernel (float *Vrad, float *Vtheta, float *SG_Accr, float *SG_Acct, float *Rinf, float *Rmed,
  float *invdiffRmed, float dt, int nrad, int nsec);

__global__ void Azimutalvelocity_withSGKernel (float *Vtheta, float *Rmed, float FLARINGINDEX, float SIGMASLOPE,
  float ASPECTRATIO, float G, float *GLOBAL_bufarray, int nrad, int nsec);

__global__ void CrashKernel (float *array, int NRAD, int NSEC, bool Crash);

__global__ void EvanescentBoundaryKernel(float *Rmed, float *Vrad, float *Vtheta, float *energy, float *Dens,
  float *AspectRatioRmed, float *viscosity_array, float DRMIN, float DRMAX, int nrad, int nsec, float Tin,
  float Tout, float step, float G, float SIGMASLOPE, float FLARINGINDEX, float *GLOBAL_bufarray, float OmegaFrame,
  float *SigmaMed, float *EnergyMed, int Adiabaticc, int SelfGravity);

__global__ void DivisePolarGridKernel (float *res, float *num, float *denom, int nrad, int nsec);

__global__ void  VanLeerRadialKernel (float *Rinf, float *Rsup, float *QRStar, float *DensStar, float *Vrad,
  float *LostByDisk, int nsec, int nrad, float dt, int OpenInner);

__global__ void ComputeAverageThetaVelocitiesKernel(float *Vtheta, float *VMed, int nsec, int nrad);

__global__ void ComputeResidualsKernel (float *VthetaRes, float *VMed, int nsec, int nrad);

__global__ void ComputeConstantResidualKernel (float *VMed, float *invRmed, int *Nshift, int *NoSplitAdvection,
  int nsec, int nrad, float dt, int YES, int NO, float *Vtheta, float *VthetaRes, float *Rmed, int FastTransport);

__global__ void StarThetaKernel (float *Qbase, float *Rmed, float *Vtheta, float *QStar, int nrad, int nsec,
  float *dq, float dt);

__global__ void VanLeerThetaKernel (float *Rsup, float *Rinf, float *Surf, float dt, int nrad, int nsec,
  int UniformTransport, int *NoSplitAdvection, float *QRStar, float *DensStar, float *Vtheta, float *Qbase, int NO);

__global__ void AdvectSHIFTKernel(float *array, float *TempShift, int nsec, int nrad, int *Nshift);

__global__ void ComputeVelocitiesKernel(float *Vrad, float *Vtheta, float *Dens, float *Rmed, float *ThetaMomP,
  float *ThetaMomM, float *RadMomP, float *RadMomM, int nrad, int nsec, float OmegaFrame);

__global__ void ComputeSpeQtyKernel (float *label, float *Dens, float *Extlabel, int nrad, int nsec);

__global__ void FillForcesArraysKernel (float *Rmed, int nsec, int nrad, float xplanet, float yplanet, float smooth,
  float G, float mplanet, int Indirect_Term, float InvPlanetDistance3, float *Potential, pair IndirectTerm);

__global__ void CorrectVthetaKernel (float *Vtheta, float domega, float *Rmed, int nrad, int nsec);

__global__ void ConditionCFLKernel1D (float *Rsup, float *Rinf, float *Rmed, int nrad, int nsec,
  float *Vtheta, float *Vmoy);

__device__ float max2(float a, float b);

__device__ float min2(float a, float b);

__global__ void ConditionCFLKernel2D (float *Rsup, float *Rinf, float *Rmed, int nsec, int nrad,
  float *Vresidual, float *Vtheta, float *Vmoy, int FastTransport, float *SoundSpeed, float *Vrad,
  float DeltaT, float *DT1D, float CVNR, float *invRmed, float *DT2D, float CFLSECURITY, float *newDT);
