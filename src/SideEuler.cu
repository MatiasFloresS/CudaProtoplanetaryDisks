#include "SideEuler.cuh"
#include "kernels.cuh"
#include "SourceEuler.cuh"

extern int OpenInner, YES, blocksize, NSEC, size_grid, NonReflecting, Adiabaticc, NRAD, nsec2pot, nrad2pot, Evanescent;
extern float *SigmaMed, *vrad, *energy, *Rmed, *SoundSpeed, *AspectRatioRmed, *Rinf, *EnergyMed, ADIABATICINDEX, FLARINGINDEX;
extern float *vrad_d, *dens_d, *energy_d, *SoundSpeed_d, *AspectRatioRmed_d;
float *mean_dens, *mean_energy, mean_dens_r, mean_energy_r, *mean_dens2, *mean_energy2, mean_dens_r2, mean_energy_r2;
float *cs0, *cs1, cs0_r, cs1_r, csnrm1_r, csnrm2_r, *csnrm1, *csnrm2, *mean_dens_d, *mean_energy_d;
float *cs0_d, *cs1_d, *csnrm1_d, *csnrm2_d, *mean_dens_d2, *mean_energy_d2;


__host__ void ApplyBoundaryCondition (float *dens, float *energy, float *vrad, float *vtheta, float step, int i)
{

  if(OpenInner == YES) OpenBoundaryhost(dens, energy, vrad);

  if (NonReflecting == YES)
  {
    if (Adiabaticc) ComputeSoundSpeedhost(dens, energy);
    NonReflectingBoundaryhost(dens, energy, vrad, i);

  }
  if (Evanescent == YES) EvanescentBoundary (vrad, vtheta, step);
}

__host__ void OpenBoundaryhost(float *dens, float *energy, float *vrad)
{

  dim3 dimGrid( nsec2pot/blocksize, 1);
  dim3 dimBlock( blocksize, 1);

  OpenBoundary<<<dimGrid, dimBlock>>> (vrad_d, dens_d, energy_d, NSEC, SigmaMed);
  gpuErrchk(cudaDeviceSynchronize());

}

__host__ void NonReflectingBoundaryhost(float *dens, float *energy, float *vrad, int a)
{
  dim3 dimGrid( nsec2pot/blocksize, 1);
  dim3 dimBlock( blocksize, 1);

  ReduceCshost(a);
  int i;
  float dangle, i_angle, dangle2, i_angle2;

  i = 1;
  dangle = (pow(Rinf[i],-1.5)-1.0)/(.5*(cs0_r+cs1_r));
  dangle *= (Rmed[i] - Rmed[i-1]);
  i_angle = (int)(dangle/2.0/CUDART_PI_F*(float)NSEC+.5);

  i = NRAD-1;
  dangle2 = (pow(Rinf[i-1],-1.5)-1.0)/(.5*(csnrm1_r+csnrm2_r));
  dangle2 *= (Rmed[i]-Rmed[i-1]);
  i_angle2 = (int)(dangle/2.0/CUDART_PI_F*(float)NSEC+.5);

  NonReflectingBoundary<<<dimGrid, dimBlock>>>(dens_d, energy_d, i_angle, NSEC, vrad_d, SoundSpeed_d, SigmaMed[1], NRAD,
  SigmaMed[i-1], i_angle2);
  gpuErrchk(cudaDeviceSynchronize());

  //gpuErrchk(cudaMemcpy(dens, dens_d, size_grid*sizeof(float), cudaMemcpyDeviceToHost));
  //gpuErrchk(cudaMemcpy(energy, energy_d, size_grid*sizeof(float), cudaMemcpyDeviceToHost));
  //gpuErrchk(cudaMemcpy(vrad, vrad_d, size_grid*sizeof(float), cudaMemcpyDeviceToHost));

  ReduceMeanHost(dens,energy, a);
  MinusMeanHost(dens, energy);

}

__host__ void ReduceCshost(int i)
{

  dim3 dimGrid( nsec2pot/blocksize, 1);
  dim3 dimBlock( blocksize, 1);

  cs0 = (float *)malloc(sizeof(float)*NSEC);
  cs1 = (float *)malloc(sizeof(float)*NSEC);
  csnrm1 = (float *)malloc(sizeof(float)*NSEC);
  csnrm2 = (float *)malloc(sizeof(float)*NSEC);

  if (i == 0) cscudamalloc();

  ReduceCs<<<dimGrid, dimBlock>>> (SoundSpeed_d, cs0_d, cs1_d, csnrm1_d, csnrm2_d, NSEC, NRAD);
  gpuErrchk(cudaDeviceSynchronize());


  cs0_r = deviceReduce(cs0_d, NSEC);
  cs0_r /= NSEC;
  cs1_r = deviceReduce(cs1_d, NSEC);
  cs1_r /= NSEC;

  csnrm1_r = deviceReduce(csnrm1_d, NSEC);
  csnrm1_r /= NSEC;

  csnrm2_r = deviceReduce(csnrm2_d, NSEC);
  csnrm2_r /= NSEC;

  free(cs0);
  free(cs1);
  free(csnrm1);
  free(csnrm2);
}

__host__ void ReduceMeanHost(float *dens, float *energy, int i)
{

  dim3 dimGrid( nsec2pot/blocksize, 1);
  dim3 dimBlock( blocksize, 1);

  mean_dens = (float *)malloc(sizeof(float)*NSEC);
  mean_dens2 = (float *)malloc(sizeof(float)*NSEC);
  mean_energy = (float *)malloc(sizeof(float)*NSEC);
  mean_energy2 = (float *)malloc(sizeof(float)*NSEC);

  if (i == 0) meancudamalloc();

  ReduceMean<<<dimGrid, dimBlock>>>(dens_d, energy_d, NSEC, mean_dens_d, mean_energy_d, mean_dens_d2, mean_energy_d2, NRAD);
  gpuErrchk(cudaDeviceSynchronize());

  mean_dens_r = deviceReduce(mean_dens_d, NSEC);
  mean_energy_r = deviceReduce(mean_energy_d, NSEC);
  mean_dens_r2 = deviceReduce(mean_dens_d2, NSEC);
  mean_energy_r2 = deviceReduce(mean_energy_d2, NSEC);

  //cudaFree(mean_dens_d);
  //cudaFree(mean_energy_d);
  //cudaFree(mean_dens_d2);
  //cudaFree(mean_energy_d2);

  free(mean_dens);
  free(mean_dens2);
  free(mean_energy);
  free(mean_energy2);
}

__host__ void MinusMeanHost(float *dens, float *energy)
{
  dim3 dimGrid( nsec2pot/blocksize, 1);
  dim3 dimBlock( blocksize, 1);

  MinusMean<<<dimGrid, dimBlock>>>(dens_d, energy_d, SigmaMed[0], mean_dens_r, mean_dens_r2, mean_energy_r, mean_energy_r2,
  EnergyMed[0], NSEC, NRAD, SigmaMed[NRAD-1], EnergyMed[NRAD-1]);
  gpuErrchk(cudaDeviceSynchronize());

}

__host__ void EvanescentBoundary (float *vrad, float *vtheta, float step)
{
  float Tin, Tout, DRMIN, DRMAX;
  /* Orbital period at inner and outer boundary */
  Tin = 2.0*CUDART_PI_F*pow(Rmed[0],3./2);;
  Tout = 2.0*CUDART_PI_F*pow(Rmed[NRAD-1],3./2);
  /* DRMIN AND DRMAX are global Radii boundaries of killing wave zones */
  DRMIN = Rmed[0]*1.25;
  DRMAX = Rmed[NRAD-1]*0.84;

  //viscosity = Rmed[];

}

__host__ void cscudamalloc()
{
  gpuErrchk(cudaMalloc((void**)&cs0_d, NSEC*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&cs1_d, NSEC*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&csnrm1_d, NSEC*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&csnrm2_d, NSEC*sizeof(float)));

  gpuErrchk(cudaMemcpy(cs0_d, cs0, NSEC*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(cs1_d, cs1, NSEC*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(csnrm1_d, csnrm1, NSEC*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(csnrm2_d, csnrm2, NSEC*sizeof(float), cudaMemcpyHostToDevice));

}

__host__ void meancudamalloc()
{

  gpuErrchk(cudaMalloc((void**)&mean_dens_d, NSEC*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&mean_energy_d, NSEC*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&mean_dens_d2, NSEC*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&mean_energy_d2, NSEC*sizeof(float)));

  gpuErrchk(cudaMemcpy(mean_dens_d, mean_dens, NSEC*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(mean_energy_d, mean_energy, NSEC*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(mean_dens_d2, mean_dens2, NSEC*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(mean_energy_d2, mean_energy2, NSEC*sizeof(float), cudaMemcpyHostToDevice));

}
