#include "Main.cuh"

extern int NRAD, NSEC, size_grid, AdvecteLabel, OpenInner, Adiabatic, FastTransport;

extern float OmegaFrame;

extern float *Label_d, *QStar_d, *Qbase_d, *Qbase2_d;
extern float *DensStar_d, *array_d;
extern float *DensStar, *QStar, *Qbase;
extern float *invdiffRmed_d, *Rinf_d, *Rmed_d, *invRmed_d, *Rsup_d, *invSurf_d,  *Surf_d;

extern float *Dens_d, *Vrad_d, *Vtheta_d, *Energy_d,  *Vazimutal_d, *DensInt_d, *DensInt;

extern dim3 dimGrid2, dimBlock2, dimBlock, dimGrid4;

float *RadMomP, *RadMomM, *ThetaMomP, *ThetaMomM, *Work;
float *VthetaRes, *RadMomP_d, *RadMomM_d, *ThetaMomP_d, *ThetaMomM_d, *VthetaRes_d, *Work_d, *TempShift_d;

float *QRStar, *ExtLabel, *dq;
float *TempShift;
float *QRStar_d, *ExtLabel_d;
float *dq_d, *LostByDisk_d, *VMed_d;
float LostMass = 0.0;

static int UniformTransport;

int *NoSplitAdvection_d;
int *Nshift_d;

__host__ void Transport (float *Dens, float *Vrad, float *Vtheta, float *Energy, float *Label, float dt)
{

  ComputeLRMomenta(); // bien hasta aca
  //if (AdvecteLabel == YES) ComputeExtQty();

  /* No-Alternate Directionnal Splitting */
  OneWindRad (Dens, Vrad, Energy, dt);
  OneWindTheta (Dens, Vtheta, Energy, dt);
  ComputeVelocities (Dens, Vrad, Vtheta);
  //if (AdvecteLabel) ComputeSpeQty (Dens, Label, ExtLabel);
}


__host__ void ComputeLRMomenta()
{
  LRMomentaKernel<<<dimGrid2, dimBlock2>>>(RadMomP_d, RadMomM_d, ThetaMomP_d, ThetaMomM_d, Dens_d, Vrad_d, Vtheta_d,
    NRAD, NSEC, Rmed_d, OmegaFrame);
  gpuErrchk(cudaDeviceSynchronize());
}


__host__ void ComputeExtQty()
{
  ExtQtyKernel<<<dimGrid2, dimBlock2>>>(ExtLabel_d, Dens_d, Label_d, NSEC, NRAD);
  gpuErrchk(cudaDeviceSynchronize());
}



__host__ void OneWindRad (float *Dens, float *Vrad, float *Energy, float dt)
{

  ComputeStarRad(Vrad, dt, 0);
  ActualiseGasDens (DensInt, Dens);

  VanLeerRadial (Vrad, dt, 0, 0);
  VanLeerRadial (Vrad, dt, 0, 1);
  VanLeerRadial (Vrad, dt, 0, 2);
  VanLeerRadial (Vrad, dt, 0, 3);

  if (Adiabatic)
    VanLeerRadial (Vrad, dt, 0, 4);

  if (AdvecteLabel == YES)
    VanLeerRadial (Vrad, dt, 0, 5);

  LostMass += VanLeerRadial (Vrad, dt, 1, 6);

}

__host__ void ComputeStarRad(float *Vrad, float dt, int option)
{
  if(option == 0){
    StarRadKernel<<<dimGrid2, dimBlock2>>> (Dens_d, Vrad_d, DensStar_d, dt, NRAD, NSEC, invdiffRmed_d, Rmed_d, dq_d);
    gpuErrchk(cudaDeviceSynchronize());
    StarRadKernel2<<<dimGrid2, dimBlock2>>> (Dens_d, Vrad_d, DensStar_d, dt, NRAD, NSEC, invdiffRmed_d, Rmed_d, dq_d);
    gpuErrchk(cudaDeviceSynchronize());
  }
  else{
    StarRadKernel<<<dimGrid2, dimBlock2>>> (Work_d, Vrad_d, QRStar_d, dt, NRAD, NSEC, invdiffRmed_d, Rmed_d, dq_d);
    gpuErrchk(cudaDeviceSynchronize());
    StarRadKernel2<<<dimGrid2, dimBlock2>>> (Work_d, Vrad_d, QRStar_d, dt, NRAD, NSEC, invdiffRmed_d, Rmed_d, dq_d);
    gpuErrchk(cudaDeviceSynchronize());
  }
}



__host__ void ActualiseGasDens(float *DensInt, float *Dens)
{
  gpuErrchk(cudaMemcpy(DensInt_d, Dens_d, size_grid*sizeof(float), cudaMemcpyDeviceToDevice));
}



__host__ float VanLeerRadial (float *Vrad, float dt, int ReturnLost, int option)
{
  float Lost = 0.0;
  if(option == 0) DivisePolarGrid (RadMomP_d, DensInt_d, Work_d);
  if(option == 1) DivisePolarGrid (RadMomM_d, DensInt_d, Work_d);
  if(option == 2) DivisePolarGrid (ThetaMomP_d, DensInt_d, Work_d);
  if(option == 3) DivisePolarGrid (ThetaMomM_d, DensInt_d, Work_d);
  if(option == 4) DivisePolarGrid (Energy_d, DensInt_d, Work_d);
  if(option == 6) DivisePolarGrid (Dens_d, DensInt_d, Work_d);
  ComputeStarRad (Vrad, dt, 1);

  if (option == 0) {
    VanLeerRadialKernel<<<dimGrid2, dimBlock2>>>(Rinf_d, Rsup_d, QRStar_d, DensStar_d, Vrad_d,
      LostByDisk_d, NSEC, NRAD, dt, OpenInner, RadMomP_d, invSurf_d);
    gpuErrchk(cudaDeviceSynchronize());
  }

  if (option == 1) {
    VanLeerRadialKernel<<<dimGrid2, dimBlock2>>>(Rinf_d, Rsup_d, QRStar_d, DensStar_d, Vrad_d,
      LostByDisk_d, NSEC, NRAD, dt, OpenInner, RadMomM_d, invSurf_d);
    gpuErrchk(cudaDeviceSynchronize());
  }

  if (option == 2) {
    VanLeerRadialKernel<<<dimGrid2, dimBlock2>>>(Rinf_d, Rsup_d, QRStar_d, DensStar_d, Vrad_d,
      LostByDisk_d, NSEC, NRAD, dt, OpenInner, ThetaMomP_d, invSurf_d);
    gpuErrchk(cudaDeviceSynchronize());
  }

  if (option == 3) {
    VanLeerRadialKernel<<<dimGrid2, dimBlock2>>>(Rinf_d, Rsup_d, QRStar_d, DensStar_d, Vrad_d,
      LostByDisk_d, NSEC, NRAD, dt, OpenInner, ThetaMomM_d, invSurf_d);
    gpuErrchk(cudaDeviceSynchronize());
  }

  if (option == 4) {
    VanLeerRadialKernel<<<dimGrid2, dimBlock2>>>(Rinf_d, Rsup_d, QRStar_d, DensStar_d, Vrad_d,
      LostByDisk_d, NSEC, NRAD, dt, OpenInner, Energy_d, invSurf_d);
    gpuErrchk(cudaDeviceSynchronize());
  }

  if (option == 6) {
    VanLeerRadialKernel<<<dimGrid2, dimBlock2>>>(Rinf_d, Rsup_d, QRStar_d, DensStar_d, Vrad_d,
      LostByDisk_d, NSEC, NRAD, dt, OpenInner, Dens_d, invSurf_d);
    gpuErrchk(cudaDeviceSynchronize());
  }

  if (ReturnLost) Lost = DeviceReduce(LostByDisk_d, NSEC);

  return Lost;
}



__host__ void ComputeSpeQty (float *Dens, float *label, float *ExtLabel)
{
  ComputeSpeQtyKernel<<<dimGrid2, dimBlock2>>>(Label_d, Dens_d, ExtLabel_d, NRAD, NSEC);
  gpuErrchk(cudaDeviceSynchronize());
}



__host__ void ComputeVelocities(float *Dens, float *Vrad, float *Vtheta)
{
  ComputeVelocitiesKernel<<<dimGrid2, dimBlock2>>>(Vrad_d, Vtheta_d, Dens_d, Rmed_d, ThetaMomP_d,
    ThetaMomM_d, RadMomP_d, RadMomM_d,  NRAD,  NSEC, OmegaFrame);
  gpuErrchk(cudaDeviceSynchronize());
}



__host__ void OneWindTheta (float *Dens, float *Vtheta, float *Energy, float dt)
{
  ComputeAverageThetaVelocities (Vtheta, dt);
  ComputeResiduals (Vtheta, dt);
  ComputeConstantResidual (Vtheta, dt); /* Constant residual is in Vtheta from now on */
  UniformTransport = NO;
  QuantitiesAdvection (Dens, VthetaRes_d, Energy, dt, 0);
  UniformTransport = YES;
  QuantitiesAdvection (Dens, Vtheta_d, Energy, dt, 1);

  AdvectSHIFT (RadMomP_d);
  AdvectSHIFT (RadMomM_d);
  AdvectSHIFT (ThetaMomP_d);
  AdvectSHIFT (ThetaMomM_d);
  if (Adiabatic) AdvectSHIFT (Energy_d);
  //if (AdvecteLabel) AdvectSHIFT (ExtLabel_d);
  AdvectSHIFT (Dens_d);
}

__host__ void ComputeAverageThetaVelocities (float *Vtheta, float dt)
{
  ComputeAverageThetaVelocitiesKernel<<<dimGrid4, dimBlock>>>(Vtheta_d, VMed_d, NSEC, NRAD);
  gpuErrchk(cudaDeviceSynchronize());
}


__host__ void ComputeResiduals (float *Vtheta, float dt)
{
  ComputeResidualsKernel<<<dimGrid2, dimBlock2>>>(VthetaRes_d, VMed_d, NSEC, NRAD, Vtheta_d);
  gpuErrchk(cudaDeviceSynchronize());
}


__host__ void AdvectSHIFT (float *array_d)
{
  AdvectSHIFTKernel<<<dimGrid2, dimBlock2>>> (array_d, TempShift_d, NSEC, NRAD, Nshift_d);
  gpuErrchk(cudaDeviceSynchronize());
  gpuErrchk(cudaMemcpy(array_d, TempShift_d, size_grid*sizeof(float), cudaMemcpyDeviceToDevice));

}



__host__ void ComputeConstantResidual (float *Vtheta, float dt)
{
  ComputeConstantResidualKernel<<<dimGrid2, dimBlock2>>>(VMed_d, invRmed_d, Nshift_d, NoSplitAdvection_d,
    NSEC, NRAD, dt, Vtheta_d, VthetaRes_d, Rmed_d, FastTransport);
  gpuErrchk(cudaDeviceSynchronize());
}



__host__ void QuantitiesAdvection (float *Dens, float *Vazimutal_d, float *Energy, float dt, int option)
{

  ComputeStarTheta (Dens_d, Vazimutal_d, DensStar_d, dt);
  ActualiseGasDens (DensInt, Dens);
  VanLeerTheta (Vazimutal_d, RadMomP_d, dt);
  VanLeerTheta (Vazimutal_d, RadMomM_d, dt);
  VanLeerTheta (Vazimutal_d, ThetaMomP_d, dt);
  VanLeerTheta (Vazimutal_d, ThetaMomM_d, dt);

  if (Adiabatic)
    VanLeerTheta (Vazimutal_d, Energy_d, dt);
  //if (AdvecteLabel)
    //VanLeerTheta (Vazimutal_d, ExtLabel_d, dt);
  VanLeerTheta (Vazimutal_d, Dens_d, dt); /* MUST be the last line */

}



__host__ void VanLeerTheta (float *Vazimutal_d, float *Qbase_d, float dt)
{
  DivisePolarGrid (Qbase_d, DensInt_d, Work_d);
  ComputeStarTheta (Work_d, Vazimutal_d, QRStar_d, dt);

  VanLeerThetaKernel<<<dimGrid2, dimBlock2>>>(Rsup_d, Rinf_d, Surf_d, dt, NRAD, NSEC, 0,
    NoSplitAdvection_d, QRStar_d, DensStar_d, Vazimutal_d, Qbase_d);
    gpuErrchk(cudaDeviceSynchronize());
  }



__host__ void ComputeStarTheta (float *Qbase_d, float *Vazimutal_d, float *QStar_d, float dt)
{
    //gpuErrchk(cudaMemset(dq_d, 0, size_grid*sizeof(float)));
    StarThetaKernel<<<dimGrid2, dimBlock2>>> (Qbase_d, Rmed_d, NRAD, NSEC, dq_d, dt);
    gpuErrchk(cudaDeviceSynchronize());

    StarThetaKernel2<<<dimGrid2, dimBlock2>>>(Qbase_d, Rmed_d, Vazimutal_d, QStar_d, NRAD, NSEC, dq_d, dt);
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
  ExtLabel        = (float *)malloc(size_grid*sizeof(float));
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
  gpuErrchk(cudaMalloc((void**)&ExtLabel_d,       size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&dq_d,             size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&LostByDisk_d,     NSEC*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&VthetaRes_d,      size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&TempShift_d,      size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&VMed_d,             NRAD*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&Nshift_d,           NRAD*sizeof(int)));
  gpuErrchk(cudaMalloc((void**)&NoSplitAdvection_d, NRAD*sizeof(int)));


  gpuErrchk(cudaMemset(RadMomP_d, 0, size_grid*sizeof(float)));
  gpuErrchk(cudaMemset(RadMomM_d, 0, size_grid*sizeof(float)));
  gpuErrchk(cudaMemset(ThetaMomP_d, 0, size_grid*sizeof(float)));
  gpuErrchk(cudaMemset(ThetaMomM_d, 0, size_grid*sizeof(float)));
  gpuErrchk(cudaMemset(Work_d, 0, size_grid*sizeof(float)));
  gpuErrchk(cudaMemset(QRStar_d, 0, size_grid*sizeof(float)));
  gpuErrchk(cudaMemset(ExtLabel_d, 0, size_grid*sizeof(float)));
  gpuErrchk(cudaMemset(dq_d, 0, size_grid*sizeof(float)));
  gpuErrchk(cudaMemset(LostByDisk_d, 0, NSEC*sizeof(float)));
  gpuErrchk(cudaMemset(VthetaRes_d, 0, size_grid*sizeof(float)));
  gpuErrchk(cudaMemset(TempShift_d, 0, size_grid*sizeof(float)));
  gpuErrchk(cudaMemset(VMed_d, 0, NRAD*sizeof(float)));
  gpuErrchk(cudaMemset(Nshift_d, 0, NRAD*sizeof(int)));
  gpuErrchk(cudaMemset(NoSplitAdvection_d, 0, NRAD*sizeof(int)));

}
