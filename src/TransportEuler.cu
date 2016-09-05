#include "Main.cuh"
#include "TransportEuler.cuh"
#include "Kernels.cuh"

extern int NRAD, NSEC, size_grid, nsec2pot, blocksize, nrad2pot;
float *RadMomP_d, *RadMomM_d, *ThetaMomP_d, *ThetaMomM_d;
extern float OmegaFrame1, *RadMomP, *RadMomM, *ThetaMomP, *ThetaMomM, *dens_d, *vrad_d, *Rmed_d, *vtheta_d;

__host__ void Transport (float *dens, float *vrad, float *vtheta, float *energy, float *label, float dt, int i)
{
  ComputeLRMomenta (dens, vrad, vtheta, i);
}

__host__ void ComputeLRMomenta (float *dens, float *vrad, float *vtheta, int i)
{

  dim3 dimGrid( nsec2pot/blocksize, nrad2pot/blocksize );
  dim3 dimBlock( blocksize, blocksize );

  if(i == 0) Momentacudamalloc();

  LRMomenta<<<dimGrid, dimBlock>>>(RadMomP_d, RadMomM_d, ThetaMomP_d, ThetaMomM_d, dens_d, vrad_d, vtheta_d,
    NRAD, NSEC, Rmed_d, OmegaFrame1);
  gpuErrchk(cudaDeviceSynchronize());

}

__host__ void Momentacudamalloc()
{
  gpuErrchk(cudaMalloc((void**)&RadMomP_d,size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&RadMomM_d,size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&ThetaMomP_d,size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&ThetaMomM_d,size_grid*sizeof(float)));

  gpuErrchk(cudaMemcpy(RadMomP_d, RadMomP, size_grid*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(RadMomM_d, RadMomM, size_grid*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(ThetaMomP_d, ThetaMomP, size_grid*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(ThetaMomM_d, ThetaMomM, size_grid*sizeof(float), cudaMemcpyHostToDevice));

}
