#include "main.cuh"
#include "kernels.cuh"

using namespace std;

extern int NSEC, size_grid, nsec2pot, nrad2pot, blocksize, NRAD;
extern float TRANSITIONWIDTH, TRANSITIONRADIUS, TRANSITIONRATIO, ASPECTRATIO, LAMBDADOUBLING, *SoundSpeed_d, *SoundSpeed;
extern float VISCOSITY, ViscosityAlpha, *Rmed, CAVITYRATIO, CAVITYRADIUS, CAVITYWIDTH, *GLOBAL_bufarray, ALPHAVISCOSITY;
extern float *vrad_d, *vtheta_d, *Drr_d, *Dpp_d, *divergence_d, *Drp_d, *invdiffRsup_d, *Rinf_d, *Dpp;
extern float *invdiffRmed_d, *Trr_d, *Tpp_d, *dens_d, *viscosity_array_d, *Trp_d, *divergence, *Drr, *Drp, *Trr, *Trp, *Tpp;
extern float *invRinf_d, *Rsup, *invRmed;
float *Rsup_d, *invRmed_d, PhysicalTime =0.0, PhysicalTimeInitial= 0.0;

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

__host__ void ComputeViscousTerms (float *vrad, float *vtheta, float *dens, int i)
{

  float dphi, invdphi, onethird;

  dphi = 2.0*CUDART_PI_F/NSEC;
  invdphi = 1.0/dphi;
  onethird = 1.0/3.0;

  if (ViscosityAlpha)
  {
    gpuErrchk(cudaMemcpy(SoundSpeed, SoundSpeed_d, size_grid*sizeof(float), cudaMemcpyDeviceToHost));
    make1Dprofilehost (SoundSpeed);
  }

  dim3 dimGrid( nsec2pot/blocksize, nrad2pot/blocksize );
  dim3 dimBlock( blocksize, blocksize );

  if (i == 0) Viscouscudamalloc();

  ViscousTerms<<<dimGrid, dimBlock>>>(vrad_d, vtheta_d, Drr_d, Dpp_d, divergence_d, Drp_d, invdiffRsup_d,
   invdphi, invRmed_d, Rsup_d, Rinf_d, invdiffRmed_d, NRAD, NSEC, Trr_d, Tpp_d, dens_d, viscosity_array_d,
   onethird, Trp_d, invRinf_d);
  gpuErrchk(cudaDeviceSynchronize());

}

__host__ void Viscouscudamalloc()
{
  gpuErrchk(cudaMalloc((void**)&Drr_d,size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&Dpp_d,size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&divergence_d,size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&Drp_d,size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&Trr_d,size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&Tpp_d,size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&Trp_d,size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&Trp_d,size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&Rsup_d,NRAD*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&invRmed_d,NRAD*sizeof(float)));

  gpuErrchk(cudaMemcpy(invRmed_d, invRmed, NRAD*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(Rsup_d, Rsup, NRAD*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(Drr_d, Drr, size_grid*sizeof(float), cudaMemcpyHostToDevice ));
  gpuErrchk(cudaMemcpy(Dpp_d, Dpp, size_grid*sizeof(float), cudaMemcpyHostToDevice ));
  gpuErrchk(cudaMemcpy(divergence_d, divergence, size_grid*sizeof(float), cudaMemcpyHostToDevice ));
  gpuErrchk(cudaMemcpy(Drp_d, Drp, size_grid*sizeof(float), cudaMemcpyHostToDevice ));
  gpuErrchk(cudaMemcpy(Trr_d, Trr, size_grid*sizeof(float), cudaMemcpyHostToDevice ));
  gpuErrchk(cudaMemcpy(Tpp_d, Tpp, size_grid*sizeof(float), cudaMemcpyHostToDevice ));
  gpuErrchk(cudaMemcpy(Trp_d, Trp, size_grid*sizeof(float), cudaMemcpyHostToDevice ));

}
