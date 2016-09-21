#include "Main.cuh"
#include "TransportEuler.cuh"
#include "Kernels.cuh"

extern int NRAD, NSEC, size_grid, nsec2pot, blocksize, nrad2pot, AdvecteLabel, YES;

extern float OmegaFrame1, *RadMomP, *RadMomM, *ThetaMomP, *ThetaMomM, *dens_d, *vrad_d, *Rmed_d, *vtheta_d, \
*label_d, *extlabel, *densStar, *invdiffRmed_d, *QStar, *Qbase, *RadMomP_d, *RadMomM_d, *ThetaMomP_d, *ThetaMomM_d, \
*extlabel_d, *QStar_d, *Qbase_d;

extern dim3 dimGrid2, dimBlock2;

__host__ void Transport (float *dens, float *vrad, float *vtheta, float *energy, float *label, float dt)
{

  LRMomenta<<<dimGrid2, dimBlock2>>>(RadMomP_d, RadMomM_d, ThetaMomP_d, ThetaMomM_d, dens_d, vrad_d, vtheta_d,
    NRAD, NSEC, Rmed_d, OmegaFrame1);
  gpuErrchk(cudaDeviceSynchronize());

  if (AdvecteLabel == YES)
  {
    ExtQty<<<dimGrid2, dimBlock2>>>(extlabel_d, dens_d, label_d, NSEC, NRAD);
    gpuErrchk(cudaDeviceSynchronize());
  }
  /* No-Alternate Directionnal Splitting */
  OneWindRad (dens, vrad, energy, dt);

}

__host__ void OneWindRad (float *dens, float *vrad, float *energy, float dt)
{
  StarRad<<<dimGrid2, dimBlock2>>> (Qbase_d, vrad_d, QStar_d, dt, NRAD, NSEC, invdiffRmed_d, Rmed_d);
  gpuErrchk(cudaDeviceSynchronize());
}
