#include "Main.cuh"

extern int OpenInner, YES, blocksize2, NSEC, size_grid, NonReflecting, Adiabaticc, NRAD, nsec2pot, nrad2pot, \
Evanescent, SelfGravity, NO, ExcludeHill, dimfxy;

extern float *SigmaMed, *Rmed, *SoundSpeed, *AspectRatioRmed, *Rinf, *EnergyMed, ADIABATICINDEX, \
FLARINGINDEX, *vrad_d, *dens_d, *energy_d, *SoundSpeed_d, *AspectRatioRmed_d, *Rmed_d, *mean_dens, *mean_energy, \
*cs0, *cs1, *csnrm1, *csnrm2, *mean_dens2, *mean_energy2, *mean_dens_d, *mean_energy_d, *cs0_d, *cs1_d, \
*csnrm1_d, *csnrm2_d, *mean_dens_d2, *mean_energy_d2, *viscosity_array, *viscosity_array_d, *vtheta_d, \
G, SIGMASLOPE, OmegaFrame1, *SigmaMed_d, *EnergyMed_d, *GLOBAL_bufarray_d;

float  mean_dens_r, mean_energy_r, mean_dens_r2, mean_energy_r2, cs0_r, cs1_r, csnrm1_r, csnrm2_r, \
*CellAbscissa, *CellAbscissa_d, *CellOrdinate, *CellOrdinate_d;

extern dim3 dimGrid, dimBlock, dimBlock2, dimGrid2;

__host__ void ApplyBoundaryCondition (float *dens, float *energy, float *vrad, float *vtheta, float step)
{
  if(OpenInner == YES) OpenBoundary ();

  if (NonReflecting == YES)
  {
    if (Adiabaticc) ComputeSoundSpeed ();

    NonReflectingBoundary (dens, energy, vrad);
    gpuErrchk(cudaMemcpy(energy, energy_d, size_grid*sizeof(float), cudaMemcpyDeviceToHost));
  }
  if (Evanescent == YES) EvanescentBoundary (vrad, vtheta, step);
}

__host__ void NonReflectingBoundary (float *dens, float *energy, float *vrad)
{
  int i,i_angle, i_angle2;
  double dangle, dangle2;

  ReduceCs();

  i = 1;
  dangle = (pow(Rinf[i],-1.5)-1.0)/(.5*(cs0_r+cs1_r));
  dangle *= (Rmed[i] - Rmed[i-1]);
  i_angle = (int)(dangle/2.0/M_PI*(float)NSEC+.5);


  i = NRAD-1;
  dangle2 = (pow(Rinf[i-1],-1.5)-1.0)/(.5*(csnrm1_r+csnrm2_r));
  dangle2 *= (Rmed[i]-Rmed[i-1]);
  i_angle2 = (int)(dangle2/2.0/M_PI*(float)NSEC+.5);

  // printf("angle %d\n", i_angle);
  // printf("angle %d\n", i_angle2);

  // gpuErrchk(cudaMemcpy(vrad, vrad_d, size_grid*sizeof(float), cudaMemcpyDeviceToHost));
  // gpuErrchk(cudaMemcpy(SoundSpeed, SoundSpeed_d, size_grid*sizeof(float), cudaMemcpyDeviceToHost));
  // gpuErrchk(cudaMemcpy(energy, energy_d, size_grid*sizeof(float), cudaMemcpyDeviceToHost));
  // gpuErrchk(cudaMemcpy(dens, dens_d, size_grid*sizeof(float), cudaMemcpyDeviceToHost));

  NonReflectingBoundaryKernel<<<dimGrid, dimBlock>>>(dens_d, energy_d, i_angle, NSEC, vrad_d, SoundSpeed_d, SigmaMed[1], NRAD,
  SigmaMed[NRAD-2], i_angle2);
  gpuErrchk(cudaDeviceSynchronize());

  ReduceMean(dens,energy);

  MinusMeanKernel<<<dimGrid, dimBlock>>>(dens_d, energy_d, SigmaMed[0], mean_dens_r, mean_dens_r2, mean_energy_r,
  mean_energy_r2, EnergyMed[0], NSEC, NRAD, SigmaMed[NRAD-1], EnergyMed[NRAD-1]);
  gpuErrchk(cudaDeviceSynchronize());

}

__host__ void ReduceCs ()
{
  ReduceCsKernel<<<dimGrid, dimBlock>>> (SoundSpeed_d, cs0_d, cs1_d, csnrm1_d, csnrm2_d, NSEC, NRAD);
  gpuErrchk(cudaDeviceSynchronize());

  cs0_r    = DeviceReduce(cs0_d, NSEC) / NSEC;
  cs1_r    = DeviceReduce(cs1_d, NSEC) / NSEC;
  csnrm1_r = DeviceReduce(csnrm1_d, NSEC) / NSEC;
  csnrm2_r = DeviceReduce(csnrm2_d, NSEC) / NSEC;
}

__host__ void ReduceMean (float *dens, float *energy)
{
  ReduceMeanKernel<<<dimGrid, dimBlock>>>(dens_d, energy_d, NSEC, mean_dens_d, mean_energy_d, mean_dens_d2, mean_energy_d2, NRAD);
  gpuErrchk(cudaDeviceSynchronize());

  mean_dens_r    = DeviceReduce(mean_dens_d, NSEC)    / NSEC;
  mean_dens_r2   = DeviceReduce(mean_dens_d2, NSEC)   / NSEC;
  mean_energy_r  = DeviceReduce(mean_energy_d, NSEC)  / NSEC;
  mean_energy_r2 = DeviceReduce(mean_energy_d2, NSEC) / NSEC;
}

__host__ void EvanescentBoundary (float *vrad, float *vtheta, float step)
{
  float Tin, Tout, DRMIN, DRMAX;
  /* Orbital period at inner and outer boundary */
  Tin = 2.0*M_PI*pow(Rmed[0],3./2);;
  Tout = 2.0*M_PI*pow(Rmed[NRAD-1],3./2);
  /* DRMIN AND DRMAX are global Radii boundaries of killing wave zones */
  DRMIN = Rmed[0]*1.25;
  DRMAX = Rmed[NRAD-1]*0.84;

  for (int i = 0; i < NRAD; i++) viscosity_array[i] = FViscosity(Rmed[i]);
  for (int i = 0; i < NRAD; i++) AspectRatioRmed[i] = AspectRatio(Rmed[i]);

  gpuErrchk(cudaMemcpy(viscosity_array_d,viscosity_array, (NRAD+1)*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(AspectRatioRmed_d,AspectRatioRmed, NRAD*sizeof(float), cudaMemcpyHostToDevice));

  EvanescentBoundaryKernel<<<dimGrid2, dimBlock2>>>(Rmed_d, vrad_d, vtheta_d, energy_d, dens_d, AspectRatioRmed_d, viscosity_array_d,
    DRMIN, DRMAX, NRAD, NSEC, Tin, Tout, step, G, SIGMASLOPE, FLARINGINDEX,  GLOBAL_bufarray_d, OmegaFrame1, SigmaMed_d, EnergyMed_d,
    Adiabaticc, SelfGravity);
  gpuErrchk(cudaDeviceSynchronize());
}

__host__ void OpenBoundary ()
{
  OpenBoundaryKernel<<<dimGrid, dimBlock>>> (vrad_d, dens_d, energy_d, NSEC, SigmaMed);
  gpuErrchk(cudaDeviceSynchronize());
}

__host__ Pair ComputeAccel (Force *force, float *dens, float x, float y, float rsmoothing, float mass)
{
  Pair acceleration;
  ComputeForce (force, dens, x, y, rsmoothing, mass, dimfxy, 0.0, 0.0);
  if (ExcludeHill)
  {
    acceleration.x = force->fx_ex_inner+force->fx_ex_outer;
    acceleration.y = force->fy_ex_inner+force->fy_ex_outer;
  }
  else
  {
    acceleration.x = force->fx_inner+force->fx_outer;
    acceleration.y = force->fx_inner+force->fy_outer;
  }
  return acceleration;
}

__host__ void DivisePolarGrid (float *Num, float *Denom, float *Res)
{
  //DivisePolarGridKernel<<<dimGrid2, dimBlock2>>> (res_d, num_d, denom_d, NRAD, NSEC);
}

__host__ void InitComputeAccel ()
{
  CellAbscissa    = (float *)malloc(size_grid*sizeof(float));
  CellOrdinate    = (float *)malloc(size_grid*sizeof(float));

  InitComputeAccelDevice();

  InitComputeAccelKernel<<<dimGrid2, dimBlock2>>>(CellAbscissa_d, CellOrdinate_d, Rmed_d, NSEC, NRAD);
  gpuErrchk(cudaDeviceSynchronize());

}

__host__ void InitComputeAccelDevice()
{
  gpuErrchk(cudaMalloc((void**)&CellAbscissa_d, size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&CellOrdinate_d, size_grid*sizeof(float)));
  gpuErrchk(cudaMemcpy(CellAbscissa_d, CellAbscissa, size_grid*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(CellOrdinate_d, CellOrdinate, size_grid*sizeof(float), cudaMemcpyHostToDevice));
}
