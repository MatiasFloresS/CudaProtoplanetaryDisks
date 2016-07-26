#include "SideEuler.cuh"
#include "kernels.cuh"
#include "SourceEuler.cuh"

extern int OpenInner, YES, blocksize, NSEC, size_grid, NonReflecting, Adiabaticc, NRAD, nsec2pot, nrad2pot;
extern float *SigmaMed, *vrad, *dens, *energy, *Rmed, *SoundSpeed, *AspectRatioRmed;
extern float ADIABATICINDEX, FLARINGINDEX;

__host__ void ApplyBoundaryCondition (float *vrad, float *vtheta, float *dens, float *energy, float step)
{
  float *vrad_d, *dens_d, *energy_d, *Rmed_d, *SoundSpeed_d, *AspectRatioRmed_d;

  if(OpenInner == YES) OpenBoundaryhost(vrad, dens, energy);

  if (NonReflecting == YES)
  {
    if (Adiabaticc) ComputeSoundSpeedhost(dens, energy);


    /* <---------     NonReflectingBoundary()        --------> */

    float *cs1;
    cs1 = (float *)malloc(sizeof(float)*NSEC);
    //cs1 = SoundSpeed[]
    //cs0 = deviceReduce(mdcp0_d, size_grid);
    //cs1 =
    /* <---------     NonReflectingBoundary()        --------> */
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
