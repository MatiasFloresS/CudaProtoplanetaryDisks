#include "SideEuler.cuh"
#include "kernels.cuh"

extern int OpenInner, YES, blocksize, NSEC, size_grid;
extern float *SigmaMed, *vrad, *dens, *energy;

__host__ void ApplyBoundaryCondition (float *vrad, float *vtheta, float *dens, float *energy, float step)
{
  int nsec2pot;
  float *vrad_d, *dens_d, *energy_d;

  if(!isPow2(NSEC)) nsec2pot = NearestPowerOf2(NSEC);
  dim3 dimGrid( nsec2pot/blocksize, 1);
  dim3 dimBlock( blocksize, 1);

  if(~OpenInner == YES){
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
}
