#include "Main.cuh"

extern float *Dens_d;
extern float ScalingFactor;
extern int NRAD, NSEC, size_grid;

extern dim3 dimGrid2, dimBlock2;

__host__ void MultiplyPolarGridbyConstant (float *Dens)
{
  MultiplyPolarGridbyConstantKernel<<<dimGrid2, dimBlock2>>>(Dens_d, NRAD, NSEC, ScalingFactor);
  gpuErrchk(cudaDeviceSynchronize());
}
