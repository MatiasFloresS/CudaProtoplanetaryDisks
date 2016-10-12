#include "Main.cuh"

extern int NSEC, size_grid, nsec2pot, nrad2pot, blocksize, NRAD;

extern float TRANSITIONWIDTH, TRANSITIONRADIUS, TRANSITIONRATIO, ASPECTRATIO, LAMBDADOUBLING, *SoundSpeed_d, \
*SoundSpeed, VISCOSITY, ViscosityAlpha, *Rmed, CAVITYRATIO, CAVITYRADIUS, CAVITYWIDTH, *GLOBAL_bufarray, \
ALPHAVISCOSITY, *vrad_d, *vtheta_d, *invdiffRsup_d, *Rinf_d, *invdiffRmed_d, *dens_d, *viscosity_array_d,  \
*invRinf_d, *Rsup, *invRmed, *vthetaint_d, *VradInt_d, *viscosity_array, *Rsup_d, *invRmed_d, \
dphi, invdphi, onethird, *VradInt, *vthetaint, *Rmed_d, *invdiffRsup_d;

float PhysicalTime =0.0, PhysicalTimeInitial= 0.0, *DivergenceVelocity, *DRP, *DRR, *DPP, *TAURR, *TAURP, *TAUPP, \
*DivergenceVelocity_d, *DRP_d, *DRR_d, *DPP_d, *TAURR_d, *TAURP_d, *TAUPP_d;

extern dim3 dimGrid2, dimBlock2;

__host__ void UpdateVelocitiesWithViscosity(float *VradInt, float *vthetaint, float *dens, float DeltaT)
{

  UpdateVelocitiesKernel<<<dimGrid2, dimBlock2>>>(vthetaint_d, VradInt_d, invRmed_d, Rmed_d, Rsup_d, Rinf_d,
    invdiffRmed_d, invdiffRsup_d,  dens_d, invRinf_d, TAURR_d, TAURP_d, TAUPP_d, DeltaT, NRAD, NSEC);
    gpuErrchk(cudaDeviceSynchronize());

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
  if ((r >= rmin) && (r <= rmax))
  {
    aspectratio *= exp((rmax-r)/(rmax-rmin)*log(TRANSITIONRATIO));
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

__host__ void ComputeViscousTerms (float *vradial, float *vazimutal, float *dens, int option)
{

  if (ViscosityAlpha)
  {
    gpuErrchk(cudaMemcpy(SoundSpeed, SoundSpeed_d, size_grid*sizeof(float), cudaMemcpyDeviceToHost));
    Make1Dprofile (SoundSpeed);
  }

  for (int i = 0; i < NRAD; i++) viscosity_array[i] = FViscosity(Rmed[i]);
  gpuErrchk(cudaMemcpy(viscosity_array_d, viscosity_array, (NRAD+1)*sizeof(float), cudaMemcpyHostToDevice));

  if (option == 1)
  {

    ViscousTermsKernel<<<dimGrid2, dimBlock2>>>(vrad_d, vtheta_d, DRR_d, DPP_d, DivergenceVelocity_d, DRP_d, invdiffRsup_d,
      invdphi, invRmed_d, Rsup_d, Rinf_d, invdiffRmed_d, NRAD, NSEC, TAURR_d, TAUPP_d, dens_d, viscosity_array_d,
      onethird, TAURP_d, invRinf_d);
  }
  else
  {
    ViscousTermsKernel<<<dimGrid2, dimBlock2>>>(VradInt_d, vthetaint_d, DRR_d, DPP_d, DivergenceVelocity_d, DRP_d, invdiffRsup_d,
      invdphi, invRmed_d, Rsup_d, Rinf_d, invdiffRmed_d, NRAD, NSEC, TAURR_d, TAUPP_d, dens_d, viscosity_array_d,
      onethird, TAURP_d, invRinf_d);
  }

  gpuErrchk(cudaDeviceSynchronize());
}

__host__ void InitViscosity ()
{
  DivergenceVelocity  = (float *)malloc(size_grid*sizeof(float));
  DRR                 = (float *)malloc(size_grid*sizeof(float));
  DRP                 = (float *)malloc(size_grid*sizeof(float));
  DPP                 = (float *)malloc(size_grid*sizeof(float));
  TAURR               = (float *)malloc(size_grid*sizeof(float));
  TAURP               = (float *)malloc(size_grid*sizeof(float));
  TAUPP               = (float *)malloc(size_grid*sizeof(float));
  InitViscosityDevice ();
}

__host__ void InitViscosityDevice ()
{
  gpuErrchk(cudaMalloc((void**)&DivergenceVelocity_d, size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&DRR_d,                size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&DRP_d,                size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&DPP_d,                size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&TAURR_d,              size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&TAURP_d,              size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&TAUPP_d,              size_grid*sizeof(float)));
}
