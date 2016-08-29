#include "main.cuh"
#include "types.cuh"
#include "SourceEuler.cuh"
#include "kernels.cuh"
#include "Output.cuh"
#include "Init.cuh"

extern int NRAD, NSEC, nsec2pot, blocksize, nrad2pot, size_grid;
extern float *Rmed, *Rmed_d;
float *gas_label_d;

__host__ void Initialization (float *dens, float *energy, float *vrad, float * vtheta, float * gas_label, PlanetarySystem *sys)
{
  float rp, rhill, xp, yp;
  xp = sys->x[0];
  yp = sys->y[0];
  rp = sqrt(xp*xp+yp*yp);
  rhill = rp * pow(sys->mass[0]/3., 1./3);

  InitEuler (dens, energy, vrad, vtheta);
  InitLabelhost(xp, yp, rhill, gas_label);
  WriteDim();

}

__host__ void InitLabelhost(float xp, float yp, float rhill, float *gas_label)
{
  dim3 dimGrid( nsec2pot/blocksize, nrad2pot/blocksize );
  dim3 dimBlock( blocksize, blocksize );

  gpuErrchk(cudaMalloc((void**)&gas_label_d, size_grid*sizeof(float)));
  gpuErrchk(cudaMemcpy(gas_label_d, gas_label, size_grid*sizeof(float), cudaMemcpyHostToDevice ));

  InitLabel<<<dimGrid, dimBlock>>>(gas_label_d, xp, yp, rhill, Rmed_d, NRAD, NSEC);
  gpuErrchk(cudaDeviceSynchronize());
  //gpuErrchk(cudaMemcpy(gas_label, gas_label_d, size_grid*sizeof(float), cudaMemcpyDeviceToHost));

  //cudaFree(gas_label_d);
}
