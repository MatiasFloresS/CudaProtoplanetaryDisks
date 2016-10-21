#include "Main.cuh"

extern dim3 dimGrid2, dimBlock2, dimGrid3;

extern cufftComplex *SGP_Kt_dc, *SGP_Kr_dc, *SGP_St_dc, *SGP_Sr_dc, *Gr_dc, *Gphi_dc;
extern cufftComplex *Gr_d, *Gphi_d, *SGP_Kt_d, *SGP_Kr_d, *SGP_Sr_d, *SGP_St_d;

extern int NSEC, NRAD, size_grid;

extern float *Vrad_d, *Vtheta_d, *Rmed_d, *invdiffRmed_d, *Rinf_d, *Dens_d, *Radii_d;
extern float *Kr_aux_d, *Kt_aux_d, *SG_Acct_d, *SG_Accr_d, *Vradial_d, *Vazimutal_d;
extern float *VthetaInt_d, *VradInt_d, *Rmed, *SG_Accr, *GLOBAL_AxiSGAccr, *axifield_d;
extern float *GLOBAL_AxiSGAccr;

extern float G, SGP_eps, SGP_rstep, SGP_tstep, ECCENTRICITY;

extern cufftHandle planf, planb;

float *Kr_aux, *Kt_aux;
extern double *Radii2;

__host__ void compute_selfgravity (float *Dens, float DeltaT, boolean SGUpdate, int option, int initialization)
{
  /* We compute Kernel */
  if (initialization)
    compute_kernel();
  /* We compute Density */
  compute_FFT ();
  /* Execute plan2d forward Complex-to-Complex */
  ExecuteExeC2Cforward();
  /* Convolution Kr*Sr and Kt*St in Fourier space */
  compute_Convolution();
  /* Execute plan2d backward Complex-to-Complex */
  ExecuteExeC2Cbackward();
  /* Here we compute radial and azimutal components of sg acceleration
     as a convolution product of reduced density and kernel arrays */
  compute_sgacc();

  if (SGUpdate) {
    /* Computes polar components of acceleration and
      updates values of vrad, vtheta at each step */
      if (option == 1)
      {
        // Vrad and Vtheta case
        gpuErrchk(cudaMemcpy(Vradial_d, Vrad_d, size_grid*sizeof(float), cudaMemcpyDeviceToDevice));
        gpuErrchk(cudaMemcpy(Vazimutal_d, Vtheta_d, size_grid*sizeof(float), cudaMemcpyDeviceToDevice));
        update_sgvelocity(DeltaT);
        gpuErrchk(cudaMemcpy(Vrad_d, Vradial_d, size_grid*sizeof(float), cudaMemcpyDeviceToDevice));
        gpuErrchk(cudaMemcpy(Vtheta_d, Vazimutal_d, size_grid*sizeof(float), cudaMemcpyDeviceToDevice));
      }
      if (option == 2)
      {
        // VradInt and VthetaInt case
        gpuErrchk(cudaMemcpy(Vradial_d, VradInt_d, size_grid*sizeof(float), cudaMemcpyDeviceToDevice));
        gpuErrchk(cudaMemcpy(Vazimutal_d, VthetaInt_d, size_grid*sizeof(float), cudaMemcpyDeviceToDevice));
        update_sgvelocity(DeltaT);
        gpuErrchk(cudaMemcpy(VradInt_d, Vradial_d, size_grid*sizeof(float), cudaMemcpyDeviceToDevice));
        gpuErrchk(cudaMemcpy(VthetaInt_d, Vazimutal_d, size_grid*sizeof(float), cudaMemcpyDeviceToDevice));
      }
  }
}



__host__ void compute_kernel ()
{
  double u, theta, Kr, Kt, base;
  int i,j;
  /* Si se elige la opcion SelfGravity se crean los arreglos para calcular
     los Kernels Kr, Kt */
  Kr_aux      = (float *)malloc(2*size_grid*sizeof(float));
  Kt_aux      = (float *)malloc(2*size_grid*sizeof(float));

  /* Aca calculo los kernels Kr y Kt en CPU ya que son constantes */
  for (i = 0; i < 2*NRAD; i++){
    if(i < NRAD)
      u = log(Radii2[i]/Radii2[0]);
    else
      u = -log(Radii2[2*NRAD-i]/Radii2[0]);

    for (j = 0; j < NSEC; j++){
      theta = 2.0*M_PI*(double)j  / (double)NSEC;
      Kr = 1.0 + SGP_eps*SGP_eps - cos(theta)*exp(-u);
      base = SGP_eps*SGP_eps*exp(u) + 2.0*(cosh(u) - cos(theta)) ;
      Kr *= pow(base , -1.5);

      Kt = sin(theta);
      Kt *= pow(base, -1.5);

      Kr_aux[i*NSEC+j] = (float) Kr;
      Kt_aux[i*NSEC+j] = (float) Kt;
    }
  }

  gpuErrchk(cudaMemcpy(Kr_aux_d, Kr_aux, 2*size_grid*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(Kt_aux_d, Kt_aux, 2*size_grid*sizeof(float), cudaMemcpyHostToDevice));
}


__host__ void compute_FFT ()
{
  gpuErrchk(cudaMemcpy(Kr_aux_d, Kr_aux, 2*size_grid*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(Kt_aux_d, Kt_aux, 2*size_grid*sizeof(float), cudaMemcpyHostToDevice));
  ComputeFFTKernel<<<dimGrid3, dimBlock2>>>(Radii_d, SGP_Kr_d, SGP_Kt_d, SGP_eps, NRAD, NSEC, SGP_Sr_d,
    SGP_St_d, Dens_d, Rmed_d, Kr_aux_d, Kt_aux_d);
    gpuErrchk(cudaDeviceSynchronize());
}



__host__ void compute_Convolution ()
{
  ComputeConvolutionKernel<<<dimGrid3, dimBlock2>>>(Gr_dc, Gphi_dc, SGP_Kr_dc, SGP_Kt_dc, SGP_Sr_dc, SGP_St_dc,
    NSEC, G, NRAD);
  gpuErrchk(cudaDeviceSynchronize());
}


__host__ void compute_sgacc ()
{
  ComputeSgAccKernel<<<dimGrid2, dimBlock2>>>(SG_Accr_d, SG_Acct_d, Dens_d, SGP_rstep, SGP_tstep, SGP_eps,
    NRAD, NSEC, Rmed_d, Gr_d, Gphi_d, G);
  gpuErrchk(cudaDeviceSynchronize());
}

__host__ void update_sgvelocity (float DeltaT)
{
  Update_sgvelocityKernel <<<dimGrid2, dimBlock2>>>(Vradial_d, Vazimutal_d, SG_Accr_d, SG_Acct_d, Rinf_d, Rmed_d,
    invdiffRmed_d, DeltaT , NRAD,  NSEC);
  gpuErrchk(cudaDeviceSynchronize());
}

__host__ void ExecuteExeC2Cforward ()
{
  if ((cufftExecC2C(planf, (cufftComplex *)SGP_St_d, (cufftComplex *)SGP_St_dc, CUFFT_FORWARD))!= \
    CUFFT_SUCCESS){
      printf("cufft execC2C error\n");
      exit(-1);
  }
  gpuErrchk(cudaDeviceSynchronize());

  if ((cufftExecC2C(planf,(cufftComplex *)SGP_Kt_d, (cufftComplex *)SGP_Kt_dc, CUFFT_FORWARD))!= \
    CUFFT_SUCCESS){
      printf("cufft execC2C error\n");
      exit(-1);
  }
  gpuErrchk(cudaDeviceSynchronize());

  if ((cufftExecC2C(planf, (cufftComplex *)SGP_Sr_d, (cufftComplex *)SGP_Sr_dc, CUFFT_FORWARD))!= \
    CUFFT_SUCCESS){
      printf("cufft execC2C error\n");
      exit(-1);
  }
  gpuErrchk(cudaDeviceSynchronize());

  if ((cufftExecC2C(planf, (cufftComplex *)SGP_Kr_d, (cufftComplex *)SGP_Kr_dc, CUFFT_FORWARD))!= \
    CUFFT_SUCCESS){
      printf("cufft execC2C error\n");
      exit(-1);
  }
  gpuErrchk(cudaDeviceSynchronize());

}

__host__ void ExecuteExeC2Cbackward ()
{
  if ((cufftExecC2C(planb, (cufftComplex *)Gphi_dc, (cufftComplex *)Gphi_d, CUFFT_INVERSE))!= \
    CUFFT_SUCCESS){
      printf("cufft execC2C error\n");
      exit(-1);
  }
  gpuErrchk(cudaDeviceSynchronize());

  if ((cufftExecC2C(planb, (cufftComplex *)Gr_dc, (cufftComplex *)Gr_d, CUFFT_INVERSE))!= \
    CUFFT_SUCCESS){
      printf("cufft execC2C error\n");
      exit(-1);
  }
  gpuErrchk(cudaDeviceSynchronize());
}




__host__ void Init_planetarysys_withSG (PlanetarySystem *sys)
{
  /*  !SGZeroMode case */
  Make1Dprofile (1);
  gpuErrchk(cudaMemcpy(GLOBAL_AxiSGAccr, axifield_d, NRAD*sizeof(float), cudaMemcpyDeviceToHost));

  float r, dist, ri, rip1, dr, sgacc;
  int ipl, k;

  /* Planetary system initialization in self-gravity cases:
     planets are put in a fixed circular orbit, we need to know radial sg acceleration
     felt by planets.*/

  for (k = 0; k < sys->nb; k++){
    r = sys->x[k];
    /* dist denotes the planet's semi-major axis */
    dist = (float) (r / (1. + ECCENTRICITY));
    ipl = 0;
    while (Rmed[ipl] <= dist) ipl++;
    ri = Rmed[ipl];
    rip1 = Rmed[ipl+1];
    dr = rip1 - ri;
    sgacc = (dist - ri)*GLOBAL_AxiSGAccr[ipl+1] + (rip1 - dist)*GLOBAL_AxiSGAccr[ipl];
    sgacc /= dr;

    /* sgacc is the radial sg acc. at the planet's semi-major axis */
    sys->vy[k] *= (float) sqrt(1. - dist*dist*sgacc);
  }
}
