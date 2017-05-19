#include "Main.cuh"

extern int NSEC, size_grid, NRAD;

extern double TRANSITIONWIDTH, TRANSITIONRADIUS, TRANSITIONRATIO, ASPECTRATIO, LAMBDADOUBLING;
extern double VISCOSITY, CAVITYRATIO, CAVITYRADIUS, CAVITYWIDTH, ALPHAVISCOSITY;

extern double ViscosityAlpha;

extern double *SoundSpeed_d, *Vrad_d, *Vtheta_d,  *Dens_d;
extern double *viscosity_array_d,  *VthetaInt_d, *VradInt_d;
extern double *Vradial_d, *Vazimutal_d;

extern double *SoundSpeed,  *GLOBAL_bufarray,  *viscosity_array, *VradInt;
extern double *VthetaInt;

extern double *invdiffRmed_d, *Rinf_d, *invRinf_d, *invRmed_d, *Rmed_d, *invRmed, *Rmed;
extern double *invdiffRsup_d, *Rsup_d, *Rsup;

double *DivergenceVelocity, *DRP, *DRR, *DPP, *TAURR, *TAURP, *TAUPP;
double *DivergenceVelocity_d, *DRP_d, *DRR_d, *DPP_d, *TAURR_d, *TAURP_d, *TAUPP_d;
double PhysicalTime =0.0, PhysicalTimeInitial= 0.0;

extern dim3 dimGrid2, dimBlock2;

__host__ void UpdateVelocitiesWithViscosity(double *VradInt, double *VthetaInt, double *Dens, double DeltaT)
{
  UpdateVelocitiesKernel<<<dimGrid2, dimBlock2>>>(VthetaInt_d, VradInt_d, invRmed_d, Rmed_d, Rsup_d, Rinf_d,
    invdiffRmed_d, invdiffRsup_d,  Dens_d, invRinf_d, TAURR_d, TAURP_d, TAUPP_d, DeltaT, NRAD, NSEC);
    gpuErrchk(cudaDeviceSynchronize());
}


__host__ double FViscosity(double r)
{
  double viscosity, rmin, rmax, scale;
  int i = 0;
  viscosity = VISCOSITY;
  if (ViscosityAlpha){
    while (Rmed[i] < r) i++;
    viscosity = ALPHAVISCOSITY*GLOBAL_bufarray[i] * GLOBAL_bufarray[i] * pow(r, 1.5);
  }
  rmin = CAVITYRADIUS-CAVITYWIDTH*ASPECTRATIO;
  rmax = CAVITYRADIUS+CAVITYWIDTH*ASPECTRATIO;
  scale = 1.0+(PhysicalTime-PhysicalTimeInitial)*LAMBDADOUBLING;
  rmin *= scale;
  rmax *= scale;
  if (r < rmin) viscosity *= CAVITYRATIO;
  if ((r >= rmin) && (r <= rmax)) viscosity *= exp((rmax-r)/(rmax-rmin)*log(CAVITYRATIO));
  return viscosity;
}

__host__ void ComputeViscousTerms (double *Vradial_d, double *Vazimutal_d, double *Dens)
{

  if (ViscosityAlpha){
    gpuErrchk(cudaMemcpy(SoundSpeed, SoundSpeed_d, size_grid*sizeof(double), cudaMemcpyDeviceToHost));
    Make1Dprofile (1);
  }

  for (int i = 0; i < NRAD; i++) viscosity_array[i] = FViscosity(Rmed[i]);
  gpuErrchk(cudaMemcpy(viscosity_array_d, viscosity_array, (NRAD+1)*sizeof(double), cudaMemcpyHostToDevice));

  ViscousTermsKernel<<<dimGrid2, dimBlock2>>>(Vradial_d, Vazimutal_d, DRR_d, DPP_d, DivergenceVelocity_d,
    DRP_d, invdiffRsup_d, invRmed_d, Rsup_d, Rinf_d, invdiffRmed_d, NRAD, NSEC, TAURR_d, TAUPP_d, Dens_d,
    TAURP_d, invRinf_d, Rmed_d, viscosity_array_d);
  gpuErrchk(cudaDeviceSynchronize());
}

__host__ void InitViscosity ()
{
  DivergenceVelocity  = (double *)malloc(size_grid*sizeof(double));
  DRR                 = (double *)malloc(size_grid*sizeof(double));
  DRP                 = (double *)malloc(size_grid*sizeof(double));
  DPP                 = (double *)malloc(size_grid*sizeof(double));
  TAURR               = (double *)malloc(size_grid*sizeof(double));
  TAURP               = (double *)malloc(size_grid*sizeof(double));
  TAUPP               = (double *)malloc(size_grid*sizeof(double));
  InitViscosityDevice ();
}

__host__ void InitViscosityDevice ()
{
  gpuErrchk(cudaMalloc((void**)&DivergenceVelocity_d, size_grid*sizeof(double)));
  gpuErrchk(cudaMalloc((void**)&DRR_d,                size_grid*sizeof(double)));
  gpuErrchk(cudaMalloc((void**)&DRP_d,                size_grid*sizeof(double)));
  gpuErrchk(cudaMalloc((void**)&DPP_d,                size_grid*sizeof(double)));
  gpuErrchk(cudaMalloc((void**)&TAURR_d,              size_grid*sizeof(double)));
  gpuErrchk(cudaMalloc((void**)&TAURP_d,              size_grid*sizeof(double)));
  gpuErrchk(cudaMalloc((void**)&TAUPP_d,              size_grid*sizeof(double)));
}


__host__ double AspectRatioHost(double r)
{
  double aspectratio, rmin, rmax, scale;
  aspectratio = ASPECTRATIO;
  rmin = TRANSITIONRADIUS-TRANSITIONWIDTH*ASPECTRATIO;
  rmax = TRANSITIONRADIUS+TRANSITIONWIDTH*ASPECTRATIO;
  scale = 1.0+(PhysicalTime-PhysicalTimeInitial)*LAMBDADOUBLING;
  rmin *= scale;
  rmax *= scale;
  if (r < rmin) aspectratio *= TRANSITIONRATIO;
  if ((r >= rmin) && (r <= rmax)){
    aspectratio *= exp((rmax-r)/(rmax-rmin)*log(TRANSITIONRATIO));
  }
  return aspectratio;
}

__host__ void ComputeViscosity()
{

}
