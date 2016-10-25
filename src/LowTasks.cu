#include "Main.cuh"

extern float *Dens_d, ScalingFactor;
extern int NRAD, NSEC;

extern dim3 dimGrid2, dimBlock2;

__host__ void MultiplyPolarGridbyConstant (float *Dens)
{
  MultiplyPolarGridbyConstantKernel<<<dimGrid2, dimBlock2>>>(Dens_d, NRAD, NSEC, ScalingFactor);
  gpuErrchk(cudaDeviceSynchronize());
}
