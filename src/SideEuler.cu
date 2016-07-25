#include "SideEuler.cuh"
#include "kernels.cuh"

extern int OpenInner, YES, blocksize, NSEC, size_grid, NonReflecting, Adiabaticc, NRAD;
extern float *SigmaMed, *vrad, *dens, *energy, *Rmed, *SoundSpeed, *AspectRatioRmed;
extern float ADIABATICINDEX, FLARINGINDEX;

__host__ void ApplyBoundaryCondition (float *vrad, float *vtheta, float *dens, float *energy, float step)
{
  int nsec2pot, nrad2pot;
  float *vrad_d, *dens_d, *energy_d, *Rmed_d, *SoundSpeed_d, *AspectRatioRmed_d;

  if(!isPow2(NSEC)) nsec2pot = NearestPowerOf2(NSEC);
  if(!isPow2(NRAD)) nrad2pot = NearestPowerOf2(NRAD);
  dim3 dimGrid( nsec2pot/blocksize, 1);
  dim3 dimBlock( blocksize, 1);

  if(OpenInner == YES){
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
    //cudaFree(dens_d);
    //cudaFree(energy_d);
  }
  if (NonReflecting == YES)
  {
    if (Adiabaticc)
    {
      /* <---------     ComputeSoundSpeed()        --------> */

      dim3 dimGrid( nsec2pot/blocksize, nrad2pot/blocksize);
      dim3 dimBlock( blocksize, blocksize);

      gpuErrchk(cudaMalloc((void**)&SoundSpeed_d, size_grid*sizeof(float)));
      gpuErrchk(cudaMalloc((void**)&Rmed_d, NRAD*sizeof(float)));
      gpuErrchk(cudaMalloc((void**)&AspectRatioRmed_d, NRAD*sizeof(float)));


      gpuErrchk(cudaMemcpy(SoundSpeed_d, SoundSpeed, size_grid*sizeof(float), cudaMemcpyHostToDevice ));
      gpuErrchk(cudaMemcpy(Rmed_d, Rmed, NRAD*sizeof(float), cudaMemcpyHostToDevice));
      gpuErrchk(cudaMemcpy(AspectRatioRmed_d, AspectRatioRmed, NRAD*sizeof(float), cudaMemcpyHostToDevice));

      ComputeSoundSpeed<<<dimGrid, dimBlock>>>(SoundSpeed_d, dens_d, Rmed_d, energy_d, NSEC, NRAD,
         Adiabaticc, ADIABATICINDEX, FLARINGINDEX, AspectRatioRmed_d);
      gpuErrchk(cudaDeviceSynchronize());
      gpuErrchk(cudaMemcpy(SoundSpeed, SoundSpeed_d, size_grid*sizeof(float), cudaMemcpyDeviceToHost));

      cudaFree(SoundSpeed_d );
      cudaFree(dens_d );
      cudaFree(Rmed_d);
      cudaFree(energy_d);
      cudaFree(AspectRatioRmed_d);


      /* <---------     ComputeSoundSpeed()        --------> */


    }

    /* <---------     NonReflectingBoundary()        --------> */

    float *cs1;
    cs1 = (float *)malloc(sizeof(float)*NSEC);
    //cs1 = SoundSpeed[]
    //cs0 = deviceReduce(mdcp0_d, size_grid);
    //cs1 =
    /* <---------     NonReflectingBoundary()        --------> */
  }
}
