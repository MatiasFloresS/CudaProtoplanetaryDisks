#include "SideEuler.cuh"
#include "Kernels.cuh"
#include "SourceEuler.cuh"

extern int OpenInner, YES, blocksize2, NSEC, size_grid, NonReflecting, Adiabaticc, NRAD, nsec2pot, nrad2pot, \
Evanescent;

extern float *SigmaMed, *Rmed, *SoundSpeed, *AspectRatioRmed, *Rinf, *EnergyMed, ADIABATICINDEX, \
FLARINGINDEX, *vrad_d, *dens_d, *energy_d, *SoundSpeed_d, *AspectRatioRmed_d, *Rmed_d, *mean_dens, *mean_energy, \
*cs0, *cs1, *csnrm1, *csnrm2, *mean_dens2, *mean_energy2, *mean_dens_d, *mean_energy_d, *cs0_d, *cs1_d, \
*csnrm1_d, *csnrm2_d, *mean_dens_d2, *mean_energy_d2;

float  mean_dens_r, mean_energy_r, mean_dens_r2, mean_energy_r2, cs0_r, cs1_r, csnrm1_r, csnrm2_r;

extern dim3 dimGrid, dimBlock, dimBlock2, dimGrid2;

__host__ void ApplyBoundaryCondition (float *dens, float *energy, float *vrad, float *vtheta, float step)
{

  if(OpenInner == YES)
  {
    OpenBoundary<<<dimGrid, dimBlock>>> (vrad_d, dens_d, energy_d, NSEC, SigmaMed);
    gpuErrchk(cudaDeviceSynchronize());
  }

  if (NonReflecting == YES)
  {
    if (Adiabaticc)
    {
      ComputeSoundSpeed<<<dimGrid2, dimBlock2>>>(SoundSpeed_d, dens_d, Rmed_d, energy_d, NSEC, NRAD,
         Adiabaticc, ADIABATICINDEX, FLARINGINDEX, AspectRatioRmed_d);
      gpuErrchk(cudaDeviceSynchronize());

    }
    NonReflectingBoundaryhost(dens, energy, vrad);
    gpuErrchk(cudaMemcpy(energy, energy_d, size_grid*sizeof(float), cudaMemcpyDeviceToHost ));

  }
  if (Evanescent == YES) EvanescentBoundary (vrad, vtheta, step);
}

__host__ void NonReflectingBoundaryhost(float *dens, float *energy, float *vrad)
{

  ReduceCshost();

  int i,i_angle, i_angle2;
  float dangle, dangle2 ;

  i = 1;
  dangle = (pow(Rinf[i],-1.5)-1.0)/(.5*(cs0_r+cs1_r));
  dangle *= (Rmed[i] - Rmed[i-1]);

  i_angle = (int)(dangle/2.0/CUDART_PI_F*(float)NSEC+.5);
  i = NRAD-1;
  dangle2 = (pow(Rinf[i-1],-1.5)-1.0)/(.5*(csnrm1_r+csnrm2_r));
  dangle2 *= (Rmed[i]-Rmed[i-1]);
  i_angle2 = (int)(dangle2/2.0/CUDART_PI_F*(float)NSEC+.5);

  // printf("%d %d\n",i_angle , i_angle2);
  // printf("%f %f\n",dangle, dangle2 );
  // printf("%f %f %f %f\n", cs0_r, cs1_r, csnrm1_r, csnrm2_r);

  NonReflectingBoundary<<<dimGrid, dimBlock>>>(dens_d, energy_d, i_angle, NSEC, vrad_d, SoundSpeed_d, SigmaMed[1], NRAD,
  SigmaMed[NRAD-2], i_angle2);
  gpuErrchk(cudaDeviceSynchronize());

  ReduceMeanHost(dens,energy);

  MinusMean<<<dimGrid, dimBlock>>>(dens_d, energy_d, SigmaMed[0], mean_dens_r, mean_dens_r2, mean_energy_r,
  mean_energy_r2, EnergyMed[0], NSEC, NRAD, SigmaMed[NRAD-1], EnergyMed[NRAD-1]);
  gpuErrchk(cudaDeviceSynchronize());

}

__host__ void ReduceCshost()
{

  ReduceCs<<<dimGrid, dimBlock>>> (SoundSpeed_d, cs0_d, cs1_d, csnrm1_d, csnrm2_d, NSEC, NRAD);
  gpuErrchk(cudaDeviceSynchronize());

  cs0_r = DeviceReduce(cs0_d, NSEC) / NSEC;
  cs1_r = DeviceReduce(cs1_d, NSEC) / NSEC;
  csnrm1_r = DeviceReduce(csnrm1_d, NSEC) / NSEC;
  csnrm2_r = DeviceReduce(csnrm2_d, NSEC) / NSEC;
}

__host__ void ReduceMeanHost(float *dens, float *energy)
{

  ReduceMean<<<dimGrid, dimBlock>>>(dens_d, energy_d, NSEC, mean_dens_d, mean_energy_d, mean_dens_d2, mean_energy_d2, NRAD);
  gpuErrchk(cudaDeviceSynchronize());

  mean_dens_r = DeviceReduce(mean_dens_d, NSEC) / NSEC;
  mean_dens_r2 = DeviceReduce(mean_dens_d2, NSEC) / NSEC;
  mean_energy_r = DeviceReduce(mean_energy_d, NSEC) / NSEC;
  mean_energy_r2 = DeviceReduce(mean_energy_d2, NSEC) / NSEC;

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
