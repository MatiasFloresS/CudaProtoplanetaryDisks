#include "Main.cuh"

extern int OpenInner, NSEC, size_grid, NonReflecting, Adiabatic, NRAD;
extern int Evanescent, SelfGravity, ExcludeHill, dimfxy, OuterSourceMass;

extern float *SigmaMed, *EnergyMed, *mean_dens, *mean_energy;
extern float *cs0, *cs1, *csnrm1, *csnrm2, *mean_dens2, *mean_energy2, *viscosity_array;

extern float ADIABATICINDEX, FLARINGINDEX, SIGMASLOPE, ASPECTRATIO;
extern float TRANSITIONWIDTH, TRANSITIONRATIO, TRANSITIONRADIUS;
extern float LAMBDADOUBLING;
extern float  PhysicalTime, PhysicalTimeInitial;

extern float *Vrad_d, *Dens_d, *Energy_d, *Vtheta_d, *VthetaInt_d, *DensInt_d, *Work_d, *SoundSpeed, *SoundSpeed_d;

extern float *mean_dens_d, *mean_energy_d;
extern float *cs0_d, *cs1_d, *csnrm1_d, *csnrm2_d, *mean_dens_d2, *mean_energy_d2, *viscosity_array_d;
extern float *SigmaMed_d, *EnergyMed_d, *GLOBAL_bufarray_d, *Qbase_d;

extern float *GLOBAL_AxiSGAccr, *axifield_d;

float  mean_dens_r, mean_energy_r, mean_dens_r2, mean_energy_r2, cs0_r, cs1_r, csnrm1_r, *CellAbscissa,        \
csnrm2_r, *CellAbscissa_d, *CellOrdinate, *CellOrdinate_d, *Vmoy_d;

extern dim3 dimGrid, dimBlock, dimBlock2, dimGrid2;

extern float OmegaFrame;
extern float *Rinf, *Rmed, *Rmed_d;


__host__ void ApplyBoundaryCondition (float *Dens, float *Energy, float *Vrad, float *Vtheta, float step)
{
  if(OpenInner == YES) OpenBoundary ();

  if (NonReflecting == YES){
    if (Adiabatic) ComputeSoundSpeed ();
    NonReflectingBoundary (Dens, Energy, Vrad);
  }
  if (Evanescent == YES) EvanescentBoundary (Vrad, Vtheta, Dens, Energy, step);
  if (OuterSourceMass == YES) printf("hola\n" );
}



__host__ void NonReflectingBoundary (float *Dens, float *Energy, float *Vrad)
{
  int i,i_angle, i_angle2;
  float dangle, dangle2;

  ReduceCs();

  i = 1;
  dangle = (powf(Rinf[i],-1.5)-1.0)/(.5*(cs0_r+cs1_r));
  dangle *= (Rmed[i] - Rmed[i-1]);
  i_angle = (int)(dangle/2.0/PI*(float)NSEC+.5);

  i = NRAD-1;
  dangle2 = (powf(Rinf[i-1],-1.5)-1.0)/(.5*(csnrm1_r+csnrm2_r));
  dangle2 *= (Rmed[i]-Rmed[i-1]);
  i_angle2 = (int)(dangle2/2.0/PI*(float)NSEC+.5);

  NonReflectingBoundaryKernel<<<dimGrid, dimBlock>>>(Dens_d, Energy_d, i_angle, NSEC, Vrad_d, SoundSpeed_d, SigmaMed[1], NRAD,
  SigmaMed[NRAD-2], i_angle2);
  gpuErrchk(cudaDeviceSynchronize());

  NonReflectingBoundaryKernel2<<<dimGrid, dimBlock>>>(Dens_d, Energy_d, i_angle, NSEC, Vrad_d, SoundSpeed_d, SigmaMed[1], NRAD,
  SigmaMed[NRAD-2], i_angle2);
  gpuErrchk(cudaDeviceSynchronize());

  ReduceMean(Dens, Energy);

  MinusMeanKernel<<<dimGrid, dimBlock>>>(Dens_d, Energy_d, SigmaMed[0], mean_dens_r, mean_dens_r2, mean_energy_r,
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


__host__ void ReduceMean (float *Dens, float *Energy)
{
  ReduceMeanKernel<<<dimGrid, dimBlock>>>(Dens_d, Energy_d, NSEC, mean_dens_d, mean_energy_d, mean_dens_d2, mean_energy_d2, NRAD);
  gpuErrchk(cudaDeviceSynchronize());

  mean_dens_r    = DeviceReduce(mean_dens_d, NSEC)    / NSEC;
  mean_dens_r2   = DeviceReduce(mean_dens_d2, NSEC)   / NSEC;
  mean_energy_r  = DeviceReduce(mean_energy_d, NSEC)  / NSEC;
  mean_energy_r2 = DeviceReduce(mean_energy_d2, NSEC) / NSEC;
}



__host__ void EvanescentBoundary (float *Vrad, float *Vtheta, float *Dens, float *Energy, float step)
{

  float Tin, Tout, DRMIN, DRMAX;
  /* Orbital period at inner and outer boundary */
  Tin = 2.0*PI*powf(Rmed[0],3./2);;
  Tout = 2.0*PI*powf(Rmed[NRAD-1],3./2);
  /* DRMIN AND DRMAX are global Radii boundaries of killing wave zones */
  DRMIN = Rmed[0]*1.25;
  DRMAX = Rmed[NRAD-1]*0.84;

  for (int i = 0; i < NRAD; i++) viscosity_array[i] = FViscosity(Rmed[i]);
  gpuErrchk(cudaMemcpy(viscosity_array_d,viscosity_array, (NRAD+1)*sizeof(float), cudaMemcpyHostToDevice));

  EvanescentBoundaryKernel<<<dimGrid2, dimBlock2>>>(Rmed_d, Vrad_d, Vtheta_d, Energy_d, Dens_d,
    viscosity_array_d, DRMIN, DRMAX, NRAD, NSEC, Tin, Tout, step, SIGMASLOPE, FLARINGINDEX,  GLOBAL_bufarray_d,
    OmegaFrame, SigmaMed_d, EnergyMed_d, Adiabatic, SelfGravity, ASPECTRATIO, TRANSITIONWIDTH, TRANSITIONRADIUS,
    TRANSITIONRATIO, PhysicalTime, PhysicalTimeInitial, LAMBDADOUBLING);
  gpuErrchk(cudaDeviceSynchronize());
}



__host__ void OpenBoundary ()
{
  OpenBoundaryKernel<<<dimGrid, dimBlock>>> (Vrad_d, Dens_d, Energy_d, NSEC, SigmaMed[0]);
  gpuErrchk(cudaDeviceSynchronize());
}



__host__ Pair ComputeAccel (Force *force, float *Dens, float x, float y, float rsmoothing, float mass)
{
  Pair acceleration;
  ComputeForce (force, Dens, x, y, rsmoothing, mass, dimfxy, 0.0, 0.0);
  if (ExcludeHill){
    acceleration.x = force->fx_ex_inner+force->fx_ex_outer;
    acceleration.y = force->fy_ex_inner+force->fy_ex_outer;
  }
  else{
    acceleration.x = force->fx_inner+force->fx_outer;
    acceleration.y = force->fy_inner+force->fy_outer;
  }
  return acceleration;
}



__host__ void DivisePolarGrid (float *Qbase_d, float *DensInt_d, float *Work_d)
{
  DivisePolarGridKernel<<<dimGrid2, dimBlock2>>> (Qbase_d, DensInt_d, Work_d, NRAD, NSEC);
  gpuErrchk(cudaDeviceSynchronize());
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
}



__host__ void CorrectVtheta (float *Vtheta, float domega)
{
  CorrectVthetaKernel<<<dimGrid2, dimBlock2>>>(Vtheta_d, domega, Rmed_d, NRAD, NSEC);
  gpuErrchk(cudaDeviceSynchronize());
}



__host__ void ApplySubKeplerianBoundary(float *VthetaInt)
{
  double VKepIn, VKepOut;

  if (!SelfGravity){
    VKepIn = sqrt(G*1.0/Rmed[0] * (1.0 - (1.0+SIGMASLOPE-2.0*FLARINGINDEX) * \
      pow(AspectRatioHost(Rmed[0]), 2.0)*pow(Rmed[0], 2.0*FLARINGINDEX)));
    VKepOut = sqrt(G*1.0/Rmed[NRAD-1] * (1.0 - (1.0+SIGMASLOPE-2.0*FLARINGINDEX)* \
      pow(AspectRatioHost(Rmed[NRAD-1]), 2.0)*pow(Rmed[NRAD-1], 2.0*FLARINGINDEX)));
  }
  else{
    Make1Dprofile (1);
    gpuErrchk(cudaMemcpy(GLOBAL_AxiSGAccr, axifield_d, NRAD*sizeof(float), cudaMemcpyDeviceToHost));

    VKepIn = sqrt(G*1.0/Rmed[0] * (1.0 - (1.0+SIGMASLOPE-2.0*FLARINGINDEX) * \
      pow(AspectRatioHost(Rmed[0]), 2.0)*pow(Rmed[0], 2.0*FLARINGINDEX)) - Rmed[0]*GLOBAL_AxiSGAccr[0]);
    VKepOut = sqrt(G*1.0/Rmed[NRAD-1] * (1.0 - (1.0+SIGMASLOPE-2.0*FLARINGINDEX)* \
      pow(AspectRatioHost(Rmed[NRAD-1]), 2.0)*pow(Rmed[NRAD-1], 2.0*FLARINGINDEX)) - Rmed[NRAD-1]*GLOBAL_AxiSGAccr[NRAD-1]);

  }

  ApplySubKeplerianBoundaryKernel<<<dimGrid, dimBlock>>>(VthetaInt_d, Rmed_d, OmegaFrame, NSEC, NRAD,
    VKepIn, VKepOut);
  gpuErrchk(cudaDeviceSynchronize());
}
