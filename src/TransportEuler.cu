#include "Main.cuh"

extern int NRAD, NSEC, size_grid, nsec2pot, blocksize, nrad2pot, AdvecteLabel, YES;

extern float OmegaFrame1, *RadMomP, *RadMomM, *ThetaMomP, *ThetaMomM, *dens_d, *vrad_d, *Rmed_d, *vtheta_d, \
*label_d, *extlabel, *densStar, *invdiffRmed_d, *QStar, *Qbase, *RadMomP_d, *RadMomM_d, *ThetaMomP_d, *ThetaMomM_d, \
*extlabel_d, *QStar_d, *Qbase_d, *densint, *densint_d;

extern dim3 dimGrid2, dimBlock2;

__host__ void Transport (float *dens, float *vrad, float *vtheta, float *energy, float *label, float dt)
{

  ComputeLRMomenta();

  if (AdvecteLabel == YES) ComputeExtQty();

  /* No-Alternate Directionnal Splitting */
  OneWindRad (dens, vrad, energy, dt);
  // OneWindTheta (dens, vtheta, energy, dt);
}


__host__ void ComputeLRMomenta()
{
  LRMomentaKernel<<<dimGrid2, dimBlock2>>>(RadMomP_d, RadMomM_d, ThetaMomP_d, ThetaMomM_d, dens_d, vrad_d, vtheta_d,
    NRAD, NSEC, Rmed_d, OmegaFrame1);
  gpuErrchk(cudaDeviceSynchronize());
}

__host__ void ComputeExtQty()
{
  ExtQtyKernel<<<dimGrid2, dimBlock2>>>(extlabel_d, dens_d, label_d, NSEC, NRAD);
  gpuErrchk(cudaDeviceSynchronize());
}

__host__ void OneWindRad (float *dens, float *vrad, float *energy, float dt)
{
  StarRadKernel<<<dimGrid2, dimBlock2>>> (Qbase_d, vrad_d, QStar_d, dt, NRAD, NSEC, invdiffRmed_d, Rmed_d);
  gpuErrchk(cudaDeviceSynchronize());
  ActualiseGasDens (densint, dens);

}

__host__ void ActualiseGasDens(float *densint, float *dens)
{
  gpuErrchk(cudaMemcpy(densint_d, dens_d, size_grid*sizeof(float), cudaMemcpyDeviceToDevice));
  gpuErrchk(cudaDeviceSynchronize());
}
