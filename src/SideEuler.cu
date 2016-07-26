#include "SideEuler.cuh"
#include "kernels.cuh"
#include "SourceEuler.cuh"

extern int OpenInner, YES, blocksize, NSEC, size_grid, NonReflecting, Adiabaticc, NRAD, nsec2pot, nrad2pot;
extern float *SigmaMed, *vrad, *dens, *energy, *Rmed, *SoundSpeed, *AspectRatioRmed, *Rinf;
extern float ADIABATICINDEX, FLARINGINDEX;
float *mean_dens, *mean_energy, mean_dens_r, mean_energy_r;
float *cs0, *cs1, cs0_r, cs1_r;

__host__ void ApplyBoundaryCondition (float *vrad, float *vtheta, float *dens, float *energy, float step)
{
  float *vrad_d, *dens_d, *energy_d, *Rmed_d, *SoundSpeed_d, *AspectRatioRmed_d;

  if(OpenInner == YES) OpenBoundaryhost(vrad, dens, energy);

  if (NonReflecting == YES)
  {
    if (Adiabaticc) ComputeSoundSpeedhost(dens, energy);
    NonReflectingBoundaryhost(vrad, dens, energy);

  }
}

__host__ void OpenBoundaryhost(float *vrad, float *dens, float *energy)
{
  float *vrad_d, *dens_d, *energy_d;

  dim3 dimGrid( nsec2pot/blocksize, 1);
  dim3 dimBlock( blocksize, 1);

  gpuErrchk(cudaMalloc((void**)&vrad_d,size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&dens_d, size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&energy_d,size_grid*sizeof(float)));

  gpuErrchk(cudaMemcpy(vrad_d, vrad, size_grid*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(dens_d, dens, size_grid*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(energy_d, energy, size_grid*sizeof(float), cudaMemcpyHostToDevice));

  OpenBoundary<<<dimGrid, dimBlock>>> (vrad_d, dens_d, energy_d, NSEC, SigmaMed);
  gpuErrchk(cudaDeviceSynchronize());
  gpuErrchk(cudaMemcpy(vrad, vrad_d, size_grid*sizeof(float), cudaMemcpyDeviceToHost));

  cudaFree(vrad_d);
  cudaFree(dens_d);
  cudaFree(energy_d);

}

__host__ void NonReflectingBoundaryhost(float *vrad, float *dens, float *energy)
{
  dim3 dimGrid( nsec2pot/blocksize, 1);
  dim3 dimBlock( blocksize, 1);

  ReduceCshost();

  float dangle, i_angle;
  float *dens_d, *energy_d, *SoundSpeed_d, *vrad_d;
  dangle = (pow(Rinf[1],-1.5)-1.0)/(.5*(cs0_r+cs1_r));
  dangle *= (Rmed[1] - Rmed[0]);
  i_angle = (int)(dangle/2.0/CUDART_PI_F*(float)NSEC+.5);


  gpuErrchk(cudaMalloc((void**)&energy_d, size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&dens_d, size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&SoundSpeed_d, size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&vrad_d, size_grid*sizeof(float)));

  gpuErrchk(cudaMemcpy(energy_d, energy, size_grid*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(dens_d, dens, size_grid*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(SoundSpeed_d, SoundSpeed, size_grid*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(vrad_d, vrad, size_grid*sizeof(float), cudaMemcpyHostToDevice));

  NonReflectingBoundary<<<dimGrid, dimBlock>>>(dens_d, energy_d, i_angle, NSEC, vrad_d, SoundSpeed_d, SigmaMed[1]);
  gpuErrchk(cudaDeviceSynchronize());

  gpuErrchk(cudaMemcpy(dens, dens_d, size_grid*sizeof(float), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(energy, energy_d, size_grid*sizeof(float), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(vrad, vrad_d, size_grid*sizeof(float), cudaMemcpyDeviceToHost));

  cudaFree(dens_d);
  cudaFree(energy_d);
  cudaFree(vrad_d);
  cudaFree(SoundSpeed_d);

  ReduceMeanHost();

  printf("dens R %f\n", mean_dens_r);
  printf("energy R %f\n", mean_energy_r);
  printf("%f\n", SigmaMed[1]);

}

__host__ void ReduceCshost()
{
  float *cs0_d, *cs1_d, *SoundSpeed_d;

  dim3 dimGrid( nsec2pot/blocksize, 1);
  dim3 dimBlock( blocksize, 1);

  cs0 = (float *)malloc(sizeof(float)*NSEC);
  cs1 = (float *)malloc(sizeof(float)*NSEC);

  gpuErrchk(cudaMalloc((void**)&cs0_d, NSEC*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&cs1_d, NSEC*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&SoundSpeed_d, size_grid*sizeof(float)));

  gpuErrchk(cudaMemcpy(cs0_d, cs0, NSEC*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(cs1_d, cs1, NSEC*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(SoundSpeed_d, SoundSpeed, size_grid*sizeof(float), cudaMemcpyHostToDevice));

  ReduceCs<<<dimGrid, dimBlock>>> (SoundSpeed_d, cs0_d, cs1_d, NSEC);

  gpuErrchk(cudaDeviceSynchronize());


  cs0_r = deviceReduce(cs0_d, NSEC);
  cs0_r /= NSEC;
  cs1_r = deviceReduce(cs1_d, NSEC);
  cs1_r /= NSEC;

  cudaFree(cs0_d);
  cudaFree(cs1_d);
  cudaFree(SoundSpeed_d);
}

__host__ void ReduceMeanHost()
{
  float *dens_d, *energy_d, *mean_dens_d, *mean_energy_d;

  dim3 dimGrid( nsec2pot/blocksize, 1);
  dim3 dimBlock( blocksize, 1);

  mean_dens = (float *)malloc(sizeof(float)*NSEC);
  mean_energy = (float *)malloc(sizeof(float)*NSEC);

  gpuErrchk(cudaMalloc((void**)&dens_d, size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&energy_d, size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&mean_dens_d, NSEC*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&mean_energy_d, NSEC*sizeof(float)));

  gpuErrchk(cudaMemcpy(dens_d, dens, size_grid*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(energy_d, energy, size_grid*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(mean_dens_d, mean_dens, NSEC*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(mean_dens_d, mean_energy, NSEC*sizeof(float), cudaMemcpyHostToDevice));

  ReduceMean<<<dimGrid, dimBlock>>>(dens_d, energy_d, NSEC, mean_dens_d, mean_energy_d);
  gpuErrchk(cudaDeviceSynchronize());

  mean_dens_r = deviceReduce(mean_dens_d, NSEC);
  mean_energy_r = deviceReduce(mean_energy_d, NSEC);

  cudaFree(mean_dens_d);
  cudaFree(mean_energy_d);
  cudaFree(dens_d);
  cudaFree(energy_d);
}
