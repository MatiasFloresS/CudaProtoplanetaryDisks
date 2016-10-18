#include "Main.cuh"

extern int NRAD, NSEC, size_grid, nsec2pot, blocksize, nrad2pot, AdvecteLabel, YES, OpenInner,         \
Adiabaticc, FastTransport, NO;

extern float OmegaFrame1, *Dens_d, *Vrad_d, *Rmed_d, *Vtheta_d, *label_d, *DensStar, *invdiffRmed_d,   \
*QStar, *Qbase, *QStar_d, *Qbase_d, *DensInt, *DensInt_d, *DensStar_d, *Rinf_d, *Rsup_d, *invRmed_d,   \
*Vtheta_d, *Surf_d, *array_d;

extern dim3 dimGrid2, dimBlock2, dimBlock, dimGrid4;

float *RadMomP, *RadMomM, *ThetaMomP, *ThetaMomM, *Work, *QRStar, *Extlabel, *RadMomP_d, *RadMomM_d,    \
*ThetaMomP_d, *ThetaMomM_d, *Work_d, *QRStar_d, *Extlabel_d, *dq, *dq_d, *LostByDisk_d, LostMass = 0.0, \
*VMed_d, *VthetaRes_d, *VthetaRes, *TempShift, *TempShift_d;

static bool UniformTransport;

int *NoSplitAdvection_d, *Nshift_d;

__host__ void Transport (float *Dens, float *Vrad, float *Vtheta, float *energy, float *label, float dt)
{

  ComputeLRMomenta();

  if (AdvecteLabel == YES) ComputeExtQty();

  /* No-Alternate Directionnal Splitting */
  OneWindRad (Dens, Vrad, energy, dt);
  OneWindTheta (Dens, Vtheta, energy, dt);
  ComputeVelocities (Dens, Vrad, Vtheta);
  if (AdvecteLabel) ComputeSpeQty (Dens, label, Extlabel);
}



__host__ void ComputeSpeQty (float *Dens, float *labe, float *Extlabel)
{
  ComputeSpeQtyKernel<<<dimGrid2, dimBlock2>>>(label_d, Dens_d, Extlabel_d, NRAD, NSEC);
  gpuErrchk(cudaDeviceSynchronize());
}



__host__ void ComputeVelocities(float *Dens, float *label, float *Extlabel)
{
  ComputeVelocitiesKernel<<<dimGrid2, dimBlock2>>>(Vrad_d, Vtheta_d, Dens_d, Rmed_d, ThetaMomP_d,
    ThetaMomM_d, RadMomP_d, RadMomM_d,  NRAD,  NSEC, OmegaFrame1);
  gpuErrchk(cudaDeviceSynchronize());
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
  ComputeStarRad(Dens, Vrad, DensStar, dt);

  ActualiseGasDens (DensInt, Dens);
  VanLeerRadial (Vrad, RadMomP, dt, 0);
  VanLeerRadial (Vrad, RadMomM, dt, 0);
  VanLeerRadial (Vrad, ThetaMomP, dt, 0);
  VanLeerRadial (Vrad, ThetaMomP, dt, 0);
  if (Adiabaticc)
    VanLeerRadial (Vrad, energy, dt, 0);
  if (AdvecteLabel)
    VanLeerRadial (Vrad, Extlabel, dt, 0);
  LostMass += VanLeerRadial (Vrad, Dens, dt, 1);

}



__host__ void OneWindTheta (float *Dens, float *Vtheta, float *energy, float dt)
{
  ComputeAverageThetaVelocities (Vtheta, dt);
  ComputeResiduals (Vtheta, dt);
  ComputeConstantResidual (Vtheta, dt); /* Constant residual is in Vtheta from now on */
  UniformTransport = NO;
  QuantitiesAdvection (Dens, VthetaRes, energy, dt);
  UniformTransport = YES;
  QuantitiesAdvection (Dens, Vtheta, energy, dt);
  AdvectSHIFT (RadMomP);
  AdvectSHIFT (RadMomM);
  AdvectSHIFT (ThetaMomP);
  AdvectSHIFT (ThetaMomM);
  if (Adiabaticc) AdvectSHIFT (energy);
  if (AdvecteLabel) AdvectSHIFT (Extlabel);
  AdvectSHIFT (Dens);
}



__host__ void AdvectSHIFT (float *array)
{
  AdvectSHIFTKernel<<<dimGrid2, dimBlock2>>> (array_d, TempShift_d, NSEC, NRAD, Nshift_d);
  gpuErrchk(cudaDeviceSynchronize());
}



__host__ void ComputeConstantResidual (float *Vtheta, float dt)
{
  ComputeConstantResidualKernel<<<dimGrid2, dimBlock2>>>(VMed_d, invRmed_d, Nshift_d, NoSplitAdvection_d,
    NSEC, NRAD, dt, YES, NO, Vtheta_d, VthetaRes_d, Rmed_d, FastTransport);
  gpuErrchk(cudaDeviceSynchronize());
}



__host__ void ComputeAverageThetaVelocities (float *Vtheta, float dt)
{
  ComputeAverageThetaVelocitiesKernel<<<dimGrid4, dimBlock>>>(Vtheta_d, VMed_d, NSEC, NRAD);
  gpuErrchk(cudaDeviceSynchronize());
}



__host__ void ComputeResiduals (float *Vtheta, float dt)
{
  ComputeResidualsKernel<<<dimGrid2, dimBlock2>>>(VthetaRes_d, VMed_d, NSEC, NRAD);
  gpuErrchk(cudaDeviceSynchronize());
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



__host__ float VanLeerRadial (float *Vrad, float *Qbase, float dt, int ReturnLost)
{
  float Lost = 0.0;
  DivisePolarGrid (Qbase, DensInt, Work);
  gpuErrchk(cudaMemcpy(Qbase_d, Work_d, size_grid*sizeof(float), cudaMemcpyDeviceToDevice)); // Work_d -> Qbase_d
  ComputeStarRad (Work, Vrad, QRStar, dt);

  VanLeerRadialKernel<<<dimGrid2, dimBlock2>>>(Rinf_d, Rsup_d, QRStar_d, DensStar_d, Vrad_d,
    LostByDisk_d, NSEC, NRAD, dt, OpenInner);
  gpuErrchk(cudaDeviceSynchronize());

  if (ReturnLost) Lost = DeviceReduce(LostByDisk_d, NSEC);

  return Lost;
}



__host__ void VanLeerTheta (float *Vazimutal, float *Qbase, float dt)
{
  DivisePolarGrid (Qbase, DensInt, Work);
  ComputeStarTheta (Work, Vazimutal, QRStar, dt);

  VanLeerThetaKernel<<<dimGrid2, dimBlock2>>>(Rsup_d, Rinf_d, Surf_d, dt, NRAD, NSEC, UniformTransport,
    NoSplitAdvection_d, QRStar_d, DensStar_d, Vtheta_d, Qbase_d, NO);
  gpuErrchk(cudaDeviceSynchronize());
}



__host__ void QuantitiesAdvection (float *Dens, float *Vazimutal, float *energy, float dt)
{
  ComputeStarTheta (Dens, Vazimutal, DensStar, dt);
  ActualiseGasDens (DensInt, Dens);
  VanLeerTheta (Vazimutal, RadMomP, dt);
  VanLeerTheta (Vazimutal, RadMomM, dt);
  VanLeerTheta (Vazimutal, ThetaMomP, dt);
  VanLeerTheta (Vazimutal, ThetaMomM, dt);
  if (Adiabaticc) VanLeerTheta (Vazimutal, energy, dt);
  if (AdvecteLabel) VanLeerTheta (Vazimutal, Extlabel, dt);
  VanLeerTheta (Vazimutal, Dens, dt); /* MUST be the last line */
}



__host__ void ComputeStarTheta (float *Qbase, float *Vtheta, float *QStar, float dt)
{
    StarThetaKernel<<<dimGrid2, dimBlock2>>> (Qbase_d, Rmed_d, Vtheta_d, QStar_d, NRAD, NSEC, dq_d, dt);
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
  VthetaRes       = (float *)malloc(size_grid*sizeof(float));
  TempShift       = (float *)malloc(size_grid*sizeof(float));
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
  gpuErrchk(cudaMalloc((void**)&LostByDisk_d,     size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&VthetaRes_d,      size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&TempShift_d,      size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&VMed_d,             NRAD*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&Nshift_d,           NRAD*sizeof(int)));
  gpuErrchk(cudaMalloc((void**)&NoSplitAdvection_d, NRAD*sizeof(int)));
}
