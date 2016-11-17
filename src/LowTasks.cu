#include "Main.cuh"

extern double *Dens_d, ScalingFactor;
extern int NRAD, NSEC, size_grid;

extern dim3 dimGrid2, dimBlock2;

__host__ void MultiplyPolarGridbyConstant (double *Dens)
{
  MultiplyPolarGridbyConstantKernel<<<dimGrid2, dimBlock2>>>(Dens_d, NRAD, NSEC, ScalingFactor);
  gpuErrchk(cudaDeviceSynchronize());

/*
  gpuErrchk(cudaMemcpy(Dens, Dens_d,     size_grid*sizeof(double), cudaMemcpyDeviceToHost));

  FILE *f;

  f = fopen("newdens.txt","w");
  for (int i = 0; i < (NRAD+1)*NSEC; i++) {
    fprintf(f, "%.10f\n", Dens[i]);
  }
  fclose(f);*/
}
