#include "Main.cuh"

extern int OpenInner, NSEC, size_grid, NonReflecting, Adiabatic, NRAD;
extern int Evanescent, SelfGravity, ExcludeHill, dimfxy, OuterSourceMass;

extern double *SigmaMed, *SoundSpeed, *EnergyMed, *mean_dens, *mean_energy;
extern double *cs0, *cs1, *csnrm1, *csnrm2, *mean_dens2, *mean_energy2, *viscosity_array;

extern double ADIABATICINDEX, FLARINGINDEX, SIGMASLOPE, ASPECTRATIO;
extern double TRANSITIONWIDTH, TRANSITIONRATIO, TRANSITIONRADIUS, PhysicalTime, PhysicalTimeInitial;
extern double LAMBDADOUBLING;

extern double *Vrad_d, *Dens_d, *Energy_d, *SoundSpeed_d, *mean_dens_d, *mean_energy_d;
extern double *cs0_d, *cs1_d, *csnrm1_d, *csnrm2_d, *mean_dens_d2, *mean_energy_d2, *viscosity_array_d;
extern double *Vtheta_d, *SigmaMed_d, *EnergyMed_d, *GLOBAL_bufarray_d, *VthetaInt_d, *Work_d, *DensInt_d, *Qbase_d;

double  mean_dens_r, mean_energy_r, mean_dens_r2, mean_energy_r2, cs0_r, cs1_r, csnrm1_r, *CellAbscissa,        \
csnrm2_r, *CellAbscissa_d, *CellOrdinate, *CellOrdinate_d, *Vmoy_d;

extern dim3 dimGrid, dimBlock, dimBlock2, dimGrid2;

extern double OmegaFrame, *Rinf, *Rmed, *Rmed_d;


__host__ void ApplyBoundaryCondition (double *Dens, double *Energy, double *Vrad, double *Vtheta, double step)
{
  if(OpenInner == YES) OpenBoundary ();

  if (NonReflecting == YES){
    if (Adiabatic) ComputeSoundSpeed ();
    NonReflectingBoundary (Dens, Energy, Vrad);
  }
  if (Evanescent == YES) EvanescentBoundary (Vrad, Vtheta, Dens, Energy, step);
  if (OuterSourceMass == YES) printf("hola\n" );
}



__host__ void NonReflectingBoundary (double *Dens, double *Energy, double *Vrad)
{
  int i,i_angle, i_angle2;
  double dangle, dangle2;

  ReduceCs();

  i = 1;
  dangle = (pow(Rinf[i],-1.5)-1.0)/(.5*(cs0_r+cs1_r));
  dangle *= (Rmed[i] - Rmed[i-1]);
  i_angle = (int)(dangle/2.0/M_PI*(double)NSEC+.5);

  i = NRAD-1;
  dangle2 = (pow(Rinf[i-1],-1.5)-1.0)/(.5*(csnrm1_r+csnrm2_r));
  dangle2 *= (Rmed[i]-Rmed[i-1]);
  i_angle2 = (int)(dangle2/2.0/M_PI*(double)NSEC+.5);

  NonReflectingBoundaryKernel<<<dimGrid, dimBlock>>>(Dens_d, Energy_d, i_angle, NSEC, Vrad_d, SoundSpeed_d, SigmaMed[1], NRAD,
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


__host__ void ReduceMean (double *Dens, double *Energy)
{
  ReduceMeanKernel<<<dimGrid, dimBlock>>>(Dens_d, Energy_d, NSEC, mean_dens_d, mean_energy_d, mean_dens_d2, mean_energy_d2, NRAD);
  gpuErrchk(cudaDeviceSynchronize());

  mean_dens_r    = DeviceReduce(mean_dens_d, NSEC)    / NSEC;
  mean_dens_r2   = DeviceReduce(mean_dens_d2, NSEC)   / NSEC;
  mean_energy_r  = DeviceReduce(mean_energy_d, NSEC)  / NSEC;
  mean_energy_r2 = DeviceReduce(mean_energy_d2, NSEC) / NSEC;
}



__host__ void EvanescentBoundary (double *Vrad, double *Vtheta, double *Dens, double *Energy, double step)
{
  double Tin, Tout, DRMIN, DRMAX;
  /* Orbital period at inner and outer boundary */
  Tin = 2.0*M_PI*pow(Rmed[0],3./2);;
  Tout = 2.0*M_PI*pow(Rmed[NRAD-1],3./2);
  /* DRMIN AND DRMAX are global Radii boundaries of killing wave zones */
  DRMIN = Rmed[0]*1.25;
  DRMAX = Rmed[NRAD-1]*0.84;

  for (int i = 0; i < NRAD; i++) viscosity_array[i] = FViscosity(Rmed[i]);
  gpuErrchk(cudaMemcpy(viscosity_array_d,viscosity_array, (NRAD+1)*sizeof(double), cudaMemcpyHostToDevice));

  EvanescentBoundaryKernel<<<dimGrid2, dimBlock2>>>(Rmed_d, Vrad_d, Vtheta_d, Energy_d, Dens_d,
    viscosity_array_d, DRMIN, DRMAX, NRAD, NSEC, Tin, Tout, step, SIGMASLOPE, FLARINGINDEX,  GLOBAL_bufarray_d,
    OmegaFrame, SigmaMed_d, EnergyMed_d, Adiabatic, SelfGravity, ASPECTRATIO, TRANSITIONWIDTH, TRANSITIONRADIUS,
    TRANSITIONRATIO, PhysicalTime, PhysicalTimeInitial, LAMBDADOUBLING);
  gpuErrchk(cudaDeviceSynchronize());
}



__host__ void OpenBoundary ()
{
  OpenBoundaryKernel<<<dimGrid, dimBlock>>> (Vrad_d, Dens_d, Energy_d, NSEC, SigmaMed);
  gpuErrchk(cudaDeviceSynchronize());
}



__host__ Pair ComputeAccel (Force *force, double *Dens, double x, double y, double rsmoothing, double mass)
{
  Pair acceleration;
  ComputeForce (force, Dens, x, y, rsmoothing, mass, dimfxy, 0.0, 0.0);
  if (ExcludeHill){
    acceleration.x = force->fx_ex_inner+force->fx_ex_outer;
    acceleration.y = force->fy_ex_inner+force->fy_ex_outer;
  }
  else{
    acceleration.x = force->fx_inner+force->fx_outer;
    acceleration.y = force->fx_inner+force->fy_outer;
  }
  return acceleration;
}



__host__ void DivisePolarGrid (double *Qbase_d, double *DensInt_d, double *Work_d)
{
  DivisePolarGridKernel<<<dimGrid2, dimBlock2>>> (Qbase_d, DensInt_d, Work_d, NRAD, NSEC);
  gpuErrchk(cudaDeviceSynchronize());
}



__host__ void InitComputeAccel ()
{
  CellAbscissa    = (double *)malloc(size_grid*sizeof(double));
  CellOrdinate    = (double *)malloc(size_grid*sizeof(double));

  InitComputeAccelDevice();

  InitComputeAccelKernel<<<dimGrid2, dimBlock2>>>(CellAbscissa_d, CellOrdinate_d, Rmed_d, NSEC, NRAD);
  gpuErrchk(cudaDeviceSynchronize());
}



__host__ void InitComputeAccelDevice()
{
  gpuErrchk(cudaMalloc((void**)&CellAbscissa_d, size_grid*sizeof(double)));
  gpuErrchk(cudaMalloc((void**)&CellOrdinate_d, size_grid*sizeof(double)));
  //gpuErrchk(cudaMemcpy(CellAbscissa_d, CellAbscissa, size_grid*sizeof(double), cudaMemcpyHostToDevice));
  //gpuErrchk(cudaMemcpy(CellOrdinate_d, CellOrdinate, size_grid*sizeof(double), cudaMemcpyHostToDevice));
}



__host__ void CorrectVtheta (double *Vtheta, double domega)
{
  CorrectVthetaKernel<<<dimGrid2, dimBlock2>>>(Vtheta_d, domega, Rmed_d, NRAD, NSEC);
  gpuErrchk(cudaDeviceSynchronize());
}



__host__ void ApplySubKeplerianBoundary(double *VthetaInt)
{
  double VKepIn, VKepOut;

  if (!SelfGravity){
    VKepIn = sqrt(G*1.0/Rmed[0] * (1.0 - (1.0+SIGMASLOPE-2.0*FLARINGINDEX) * \
      pow(AspectRatioHost(Rmed[0]), 2.0)*pow(Rmed[0], 2.0*FLARINGINDEX)));
    VKepOut = sqrt(G*1.0/Rmed[NRAD-1] * (1.0 - (1.0+SIGMASLOPE-2.0*FLARINGINDEX)* \
      pow(AspectRatioHost(Rmed[NRAD-1]), 2.0)*pow(Rmed[NRAD-1], 2.0*FLARINGINDEX)));
  }


  ApplySubKeplerianBoundaryKernel<<<dimGrid, dimBlock>>>(VthetaInt_d, Rmed_d, OmegaFrame, NSEC, NRAD,
    VKepIn, VKepOut);
  gpuErrchk(cudaDeviceSynchronize());
}
