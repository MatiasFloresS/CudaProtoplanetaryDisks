#include "Main.cuh"

extern int NRAD, NSEC, size_grid, nsec2pot, blocksize, nrad2pot, AdvecteLabel, YES;

extern float OmegaFrame1, *Dens_d, *Vrad_d, *Rmed_d, *Vtheta_d, *label_d, *densStar, *invdiffRmed_d, \
*QStar, *Qbase, *QStar_d, *Qbase_d, *DensInt, *DensInt_d, *RhoStar, *RhoStar_d;

extern dim3 dimGrid2, dimBlock2;

float *RadMomP, *RadMomM, *ThetaMomP, *ThetaMomM, *Work, *QRStar, *Extlabel, *RadMomP_d, *RadMomM_d,  \
*ThetaMomP_d, *ThetaMomM_d, *Work_d, *QRStar_d, *Extlabel_d, *dq, *dq_d;

__host__ void Transport (float *Dens, float *Vrad, float *Vtheta, float *energy, float *label, float dt)
{

  ComputeLRMomenta();

  if (AdvecteLabel == YES) ComputeExtQty();

  /* No-Alternate Directionnal Splitting */
  OneWindRad (Dens, Vrad, energy, dt);
  // OneWindTheta (dens, vtheta, energy, dt);
}


__host__ void ComputeLRMomenta()
{
  LRMomentaKernel<<<dimGrid2, dimBlock2>>>(RadMomP_d, RadMomM_d, ThetaMomP_d, ThetaMomM_d, Dens_d, Vrad_d, Vtheta_d,
    NRAD, NSEC, Rmed_d, OmegaFrame1);
  gpuErrchk(cudaDeviceSynchronize());
}

__host__ void ComputeExtQty()
{
  ExtQtyKernel<<<dimGrid2, dimBlock2>>>(Extlabel_d, Dens_d, label_d, NSEC, NRAD);
  gpuErrchk(cudaDeviceSynchronize());
}

__host__ void OneWindRad (float *Dens, float *Vrad, float *energy, float dt)
{
  gpuErrchk(cudaMemcpy(Qbase_d, Dens_d, size_grid*sizeof(float), cudaMemcpyDeviceToDevice)); // dens_d -> Qbase_d
  ComputeStarRad(Dens, Vrad, RhoStar, dt);

  ActualiseGasDens (DensInt, Dens);
  VanLeerRadial (Vrad, RadMomP, dt);
}

__host__ void ActualiseGasDens(float *DensInt, float *Dens)
{
  gpuErrchk(cudaMemcpy(DensInt_d, Qbase_d, size_grid*sizeof(float), cudaMemcpyDeviceToDevice));
  gpuErrchk(cudaDeviceSynchronize());
}

__host__ void ComputeStarRad(float *Qbase, float *Vrad, float *QStar, float dt)
{

  StarRadKernel<<<dimGrid2, dimBlock2>>> (Qbase_d, Vrad_d, QStar_d, dt, NRAD, NSEC, invdiffRmed_d, Rmed_d, dq_d);
  gpuErrchk(cudaDeviceSynchronize());
}

__host__ void VanLeerRadial (float *Vrad, float *Qbase, float dt)
{
  DivisePolarGrid (Qbase, DensInt, Work);
  gpuErrchk(cudaMemcpy(Qbase_d, Work_d, size_grid*sizeof(float), cudaMemcpyDeviceToDevice)); // Work_d -> Qbase_d
  ComputeStarRad (Work, Vrad, QRStar, dt);

  //VanLeerRadialKernel<<<>>>();
  gpuErrchk(cudaDeviceSynchronize());
}

__host__ void InitTransport ()
{
  RadMomP         = (float *)malloc(size_grid*sizeof(float));
  RadMomM         = (float *)malloc(size_grid*sizeof(float));
  ThetaMomP       = (float *)malloc(size_grid*sizeof(float));
  ThetaMomM       = (float *)malloc(size_grid*sizeof(float));
  Work            = (float *)malloc(size_grid*sizeof(float));
  QRStar          = (float *)malloc(size_grid*sizeof(float));
  Extlabel        = (float *)malloc(size_grid*sizeof(float));
  //VthetaRes
  //Elongations
  //tempshift
  dq              = (float *)malloc(size_grid*sizeof(float));
  InitTransportDevice();

}

__host__ void InitTransportDevice()
{
  gpuErrchk(cudaMalloc((void**)&RadMomP_d,        size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&RadMomM_d,        size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&ThetaMomP_d,      size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&ThetaMomM_d,      size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&Work_d,           size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&QRStar_d,         size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&Extlabel_d,       size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&dq_d,             size_grid*sizeof(float)));
}
