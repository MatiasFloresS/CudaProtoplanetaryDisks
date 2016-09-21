#include "Main.cuh"
#include "Kernels.cuh"
#include "Fondam.cuh"
#include "SourceEuler.cuh"
#include "Psys.cuh"
#include "Pframeforce.cuh"
#include "Init.cuh"
#include "cuda_runtime.h"
#include "Output.cuh"
#include "Force.cuh"
#include "SideEuler.cuh"
#include "TransportEuler.cuh"

using namespace std;

extern float *SigmaInf_d, *AspectRatioRmed_d, *vrad_d,  OMEGAFRAME, OmegaFrame1, *press, *CellAbscissa, \
*CellOrdinate, HillRadius, PhysicalTimeInitial, PhysicalTime, CVNR, *CellAbscissa_d, *CellOrdinate_d, \
*temperature_d, *energy_d, *temperature, *vtheta_d, *press_d, *Rinf_d, *cosns_d, *sinns_d, *Qbase_d, \
*SoundSpeed_d, *viscosity_array_d, *ThetaMomP_d, *ThetaMomM_d, *RadMomP_d, *RadMomM_d, *extlabel_d, *QStar_d, \
*vt_cent, THICKNESSSMOOTHING, G, *SoundSpeed, *AspectRatioRmed;

float *Rinf, *Rmed, *Rsup, *Surf, *invRinf, *invSurf, *invdiffSurf, *invdiffRsup, *invdiffRmed, *invRmed, \
*Radii, *SigmaMed, *SigmaInf, *EnergyMed, *cosns, *sinns, mdcp, exces_mdcp = 0.0, ScalingFactor = 1.0, \
*forcesxi, *forcesyi, *forcesxo, *forcesyo, *vradint, *pot, *vthetaint, *powRmed, *densint_d, \
*temperatureint, *densint, *vradnew, *vthetanew, *energyint, *fieldsrc, *vt_int, *GLOBAL_bufarray, *Surf_d, \
*vradint_d, *pot_d, *vthetaint_d, *invdiffRmed_d, *invRinf_d, *powRmed_d, *vthetanew_d, *vradnew_d, \
*temperatureint_d, *energyint_d, *invdiffRsup_d, *CoolingTimeMed, *QplusMed , *viscosity_array, *Drr, \
*Drr_d, *Dpp, *Dpp_d, *Drp, *Drp_d, *divergence, *divergence_d, *Trr, *Trr_d, *Tpp, *Tpp_d, *Trp , *Trp_d, \
*qplus_d, *energynew_d, *EnergyMed_d, *SigmaMed_d, *CoolingTimeMed_d, *QplusMed_d, *energynew, *qplus, \
*RadMomP, *RadMomM, *ThetaMomP, *ThetaMomM, *extlabel, *densStar, *QStar, *Qbase, *gridfield_d, *GLOBAL_bufarray_d, \
*Rsup_d, *invRmed_d, *label_d, *RadMomP_d, *RadMomM_d, *ThetaMomP_d, *ThetaMomM_d, *extlabel_d, *QStar_d, *Qbase_d, \
*cs0, *cs1, *csnrm1, *csnrm2, *mean_dens, *mean_dens2, *mean_energy, *mean_energy2, *cs0_d, *cs1_d, *csnrm1_d, \
*csnrm2_d, *mean_dens_d, *mean_dens_d2, *mean_energy_d, *mean_energy_d2, *forcesxi_d, *forcesyi_d, *forcesxo_d, \
*forcesyo_d, dphi, invdphi, onethird, *SGP_Kr, *SGP_Kt, *Radii_d, *SGP_St, *SGP_Sr, *Rmed_d, *dens_d, *Gr, \
*Gphi, *SG_Accr_d, *SG_Acct_d, *SG_Accr, *SG_Acct;

extern int NRAD, NSEC, SelfGravity, Corotating, FREQUENCY, Adiabaticc, IsDisk, Cooling;
static int StillWriteOneOutput, InnerOutputCounter=0;
int nrad2pot, nsec2pot, blocksize = 16, size_grid, dimfxy=11, TimeStep = 0, NbRestart = 0, blocksize2 = 256, \
nrad2potSG, nsec2potSG, size_grid2;

bool ZMPlus = false, verbose = false, Restart = false, TimeToWrite;

dim3 dimGrid2, dimBlock2, dimGrid, dimBlock, dimGrid3;
cufftHandle planf, planb;
cufftComplex *SGP_Kt_dc, *SGP_Kr_dc, *SGP_St_dc, *SGP_Sr_dc, *Gr_dc, *Gphi_dc;
cufftReal *Gr_d, *Gphi_d, *SGP_Kt_d, *SGP_Kr_d, *SGP_Sr_d, *SGP_St_d;

__host__ int main(int argc, char *argv[])
{

  bool disable = false, TimeInfo = false, Profiling = false, Stockholm = false;
  char ParameterFile[256];

  PlanetarySystem *sys;
  Force *force;

  float *label, *dens, *energy, *vrad, *vtheta, xpl, ypl;

  if (argc == 1) PrintUsage (argv[0]);

  strcpy (ParameterFile, "");
  for (int i = 1; i < argc; i++)
  {
    if (*(argv[i]) == '-')
    {
      if (strspn (argv[i], "-secndovtpfamzib0123456789") != strlen (argv[i]))
	    PrintUsage (argv[0]);
      if (strchr (argv[i], 'n')) disable = true;
      if (strchr (argv[i], 'e')) Stockholm = true;
      if (strchr (argv[i], 'v')) verbose = true;
      if (strchr (argv[i], 't')) TimeInfo = true;
      if (strchr (argv[i], 'c')) SloppyCFL = true;
      if (strchr (argv[i], 'p')) Profiling = true;
      if (strchr (argv[i], 'd')) debug = true;
      if (strchr (argv[i], 'b')) CentrifugalBalance = true;
      if (strchr (argv[i], 'm')) Merge = true;
      if (strchr (argv[i], 'a')) MonitorIntegral = true;
      if (strchr (argv[i], 'z')) FakeSequential = true;
      if (strchr (argv[i], 'i'))
      {
        StoreSigma = true;
	      if (Adiabaticc) StoreEnergy = true;
      }
      if (strchr (argv[i], '0')) OnlyInit = true;
      if ((argv[i][1] >= '1') && (argv[i][1] <= '9'))
      {
	       GotoNextOutput = true;
	       StillWriteOneOutput = (int)(argv[i][1]-'0');
      }
      if (strchr (argv[i], 's'))
      {
        Restart = true;
	      i++;
	      NbRestart = atoi(argv[i]);
	      if ((NbRestart < 0))
        {
          printf ("Incorrect restart number\n");
	        PrintUsage (argv[0]);
        }
      }
      if (strchr (argv[i], 'o'))
      {
        OverridesOutputdir = true;
	      i++;
	      sprintf (NewOutputdir, "%s", argv[i]);
      }
      else
      {
        if (strchr (argv[i], 'f'))
        {
	        i++;
	        ScalingFactor = atof(argv[i]);
	        printf ("Scaling factor = %g\n", ScalingFactor);
	        if ((ScalingFactor <= 0))
          {
	          printf ("Incorrect scaling factor\n");
	          PrintUsage (argv[0]);
          }
        }
      }
    }
    else strcpy (ParameterFile, argv[i]);
  }

  if ( (StoreSigma || StoreEnergy) && !(Restart))
  {
    printf ("You cannot use tabulated surface density\n");
    printf ("or surface internal energy in a non-restart run.\n");
    printf ("Aborted\n");
    exit (0);
  }
  if (ParameterFile[0] == 0) PrintUsage (argv[0]);

  ReadFile(ParameterFile);
  size_grid = NRAD*NSEC;
  size_grid2 = (NRAD+1)*NSEC;

  if(!IsPow2(NRAD+1)) nrad2pot = NearestPowerOf2(NRAD+1);
  if(!IsPow2(NSEC)) nsec2pot = NearestPowerOf2(NSEC);

  if(!IsPow2(2*NRAD)) nrad2potSG = NearestPowerOf2(2*NRAD);

  if(!IsPow2(2*(NSEC/2+1))) nsec2potSG = NearestPowerOf2(2*(NSEC/2+1));

  if ((cufftPlan2d(&planf, 2*NRAD, NSEC, CUFFT_R2C)) != CUFFT_SUCCESS)
  {
    printf("cufft plan error\n");
    exit(-1);
  }

  if ((cufftPlan2d(&planb, 2*NRAD, (NSEC/2 + 1), CUFFT_C2R)) != CUFFT_SUCCESS)
  {
    printf("cufft plan error\n");
    exit(-1);
  }

  dim3 dimG2( nsec2pot/blocksize, nrad2pot/blocksize );
  dim3 dimB2( blocksize, blocksize );
  dimGrid2 = dimG2;
  dimBlock2 = dimB2;

  dim3 dimG( nsec2pot/blocksize2, 1);
  dim3 dimB( blocksize2, 1);
  dimGrid = dimG;
  dimBlock = dimB;

  dim3 dimG3(nsec2pot/blocksize, nrad2potSG/blocksize);
  dimGrid3 = dimG3;

  dphi = 2.0*CUDART_PI_F/NSEC;
  invdphi = 1.0/dphi;
  onethird = 1.0/3.0;

  if (verbose == YES) TellEverything();
  if (disable == YES) exit(0);
  printf("Allocating arrays...\n");

  vtheta = (float *) malloc(sizeof(float)*size_grid2);
  vrad = (float *) malloc(sizeof(float)*size_grid2);
  dens = (float *) malloc(sizeof(float)*size_grid2);
  energy = (float *) malloc(sizeof(float)*size_grid);
  label = (float *) malloc(sizeof(float)*size_grid);
  CreateArrays();

  printf("done.\n");

  FillPolar1DArray();

  force = AllocateForce (dimfxy);

  char configplanet[100];
  strncpy(configplanet, PLANETCONFIG.c_str(), sizeof(configplanet));
  configplanet[sizeof(configplanet)-1]=0;

  /* Here planets are initialized feeling star potential but they do
     not feel disk potential  */

  sys = InitPlanetarySystem(configplanet);

  /* Gas density initialization */
  InitGasDensity (dens);


  /* If energy equation is taken into account, we initialize the gas
     thermal energy  */
  if ( Adiabaticc ) InitGasEnergy (energy);

  cudamalloc(label, dens);

  if ( SelfGravity )
  {

    /* If SelfGravity = YES or Z, planets are initialized feeling disk
       potential. Only the surface density is required to calculate
       the radial self-gravity acceleration. The disk radial and
       azimutal velocities are not updated */

    float SGP_eps, SGP_rstep, SGP_tstep;

    SGP_eps = THICKNESSSMOOTHING * ASPECTRATIO;
    SGP_rstep = log(Radii[NRAD]/Radii[0])/(float)NRAD;
    SGP_tstep = 2.0*PI/(float)NSEC;

    fftkernel<<<dimGrid3, dimBlock2>>>(Radii_d, SGP_Kr_d, SGP_Kt_d, SGP_eps, NRAD, NSEC, SGP_Sr_d,
    SGP_St_d, dens_d, Rmed_d, nrad2potSG);
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaMemcpy(SG_Accr, SGP_Kr_d, (NRAD*NSEC)*sizeof(float), cudaMemcpyDeviceToHost));

    FILE *f;
    f = fopen("SGP_Kr_d.txt","w");

    for (int i = 0; i < NRAD*NSEC; i++) fprintf(f, "%f\n",SG_Accr[i] );

    fclose(f);


    if ((cufftExecR2C(planf, (cufftReal *)SGP_Sr_d, (cufftComplex *)SGP_Sr_dc))!= \
      CUFFT_SUCCESS)
    {
        printf("cufft execR2C error\n");
        exit(-1);
    }
    gpuErrchk(cudaDeviceSynchronize());

    if ((cufftExecR2C(planf, (cufftReal *)SGP_St_d, (cufftComplex *)SGP_St_dc))!= \
      CUFFT_SUCCESS)
      {
        printf("cufft execR2C error\n");
        exit(-1);
    }
    gpuErrchk(cudaDeviceSynchronize());

    if ((cufftExecR2C(planf, (cufftReal *)SGP_Kt_d, (cufftComplex *)SGP_Kt_dc))!= \
      CUFFT_SUCCESS)
    {
        printf("cufft execR2C error\n");
        exit(-1);
    }
    gpuErrchk(cudaDeviceSynchronize());

    if ((cufftExecR2C(planf, (cufftReal *)SGP_Kr_d, (cufftComplex *)SGP_Kr_dc))!= \
      CUFFT_SUCCESS)
    {
        printf("cufft execR2C error\n");
        exit(-1);
    }
    gpuErrchk(cudaDeviceSynchronize());

    fftkernelmul<<<dimGrid3, dimBlock2>>>(Gr_dc, Gphi_dc, SGP_Kr_dc, SGP_Kt_dc, \
      SGP_Sr_dc, SGP_St_dc, NSEC, G, NRAD);
    gpuErrchk(cudaDeviceSynchronize());


    if ((cufftExecC2R(planb, (cufftComplex *)Gr_dc, (cufftReal *)Gr_d))!= \
      CUFFT_SUCCESS)
    {
        printf("cufft execC2R error\n");
        exit(-1);
    }
    gpuErrchk(cudaDeviceSynchronize());

    if ((cufftExecC2R(planb, (cufftComplex *)Gphi_dc, (cufftReal *)Gphi_d))!= \
      CUFFT_SUCCESS)
    {
        printf("cufft execC2R error\n");
        exit(-1);
    }
    gpuErrchk(cudaDeviceSynchronize());

    kernelSg_Acc <<<dimGrid2, dimBlock2>>>(SG_Accr_d, SG_Acct_d, dens_d, SGP_rstep, SGP_tstep, SGP_eps,
      NRAD, NSEC, Rmed_d, Gr_d, Gphi_d, G);
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaMemcpy(SG_Accr, SG_Accr_d, (NRAD*NSEC)*sizeof(float), cudaMemcpyDeviceToHost));


    f = fopen("SG_Accr.txt","w");

    for (int i = 0; i < NRAD*NSEC; i++) fprintf(f, "%f\n",SG_Accr[i] );

    fclose(f);

    //init_planetarysys_withSG (sys);
  }

  ListPlanets (sys);
  OmegaFrame1 = OMEGAFRAME;

  if (Corotating) OmegaFrame1 = GetPsysInfo (sys, FREQUENCY);
  /* Only gas velocities remain to be initialized */


  Initialization (dens, energy, vrad, vtheta, label, sys);

  xpl = sys->x[0];
  ypl = sys->y[0];

  mdcp = CircumPlanetaryMasshost(dens, xpl, ypl);
  float dt = 0.999;

  EmptyPlanetSystemFile (sys);
  PhysicalTimeInitial = PhysicalTime;

  MultiplyPolarGridbyConstant<<<dimGrid2, dimBlock2>>>(dens_d, NRAD, NSEC, ScalingFactor);
  gpuErrchk(cudaDeviceSynchronize());

  for (int i  = 0; i < size_grid; i++) pot[i] = 0.00001*i;

  for (int i = 0; i < NRAD; i++) powRmed[i] = powf(Rmed[i],-2.5+SIGMASLOPE);

  for (int i = 0; i <= NTOT; i++)
  {
    InnerOutputCounter++;

    if (InnerOutputCounter == 1)
    {
      InnerOutputCounter = 0;
      WriteBigPlanetSystemFile (sys, TimeStep);
      UpdateLog(force, dens, sys, TimeStep, PhysicalTime, dimfxy, i);
    }

    if (NINTERM * (TimeStep = (i / NINTERM)) == i)
    {
      /* Outputs are done here */
      TimeToWrite = YES;

      DeviceToHostcudaMemcpy(dens, energy, label, temperature, vrad, vtheta); // Traigo los valores desde la GPU
      SendOutput (TimeStep, dens, vrad, vtheta, energy, label);
      WritePlanetSystemFile (sys, TimeStep);

    }
    else TimeToWrite = NO;

    // algogas

    if (Adiabaticc)
    {
      for (int i = 0; i < NRAD; i++) AspectRatioRmed[i] = AspectRatio(Rmed[i]);
      gpuErrchk(cudaMemcpy(AspectRatioRmed_d, AspectRatioRmed, NRAD*sizeof(float), cudaMemcpyHostToDevice));

      ComputeSoundSpeed<<<dimGrid2, dimBlock2>>>(SoundSpeed_d, dens_d, Rmed_d, energy_d, NSEC, NRAD,
         Adiabaticc, ADIABATICINDEX, FLARINGINDEX, AspectRatioRmed_d);
      gpuErrchk(cudaDeviceSynchronize());

      /* it is necesary to update computation of soundspeed if one uses
        alphaviscosity in Fviscosity. It is not necesary in locally
        isothermal runs since cs is constant. It is computed here for
        the needs of ConditionCFL. */
    }
    if (IsDisk == YES)
    {

    }

    Substep1host(dens, vrad, vtheta, dt, i); // aca voy

    Substep2<<<dimGrid2, dimBlock2>>>(dens_d, vradint_d, vthetaint_d, temperatureint_d, NRAD, NSEC, CVNR, invdiffRmed_d,
    invdiffRsup_d, densint_d, Adiabaticc, Rmed_d, dt, vradnew_d, vthetanew_d, energy_d, energyint_d);
    gpuErrchk(cudaDeviceSynchronize());

    ActualiseGasVrad(vrad, vradnew);
    ActualiseGasVtheta(vtheta, vthetanew);
    ApplyBoundaryCondition (dens, energy, vrad, vtheta, dt);

    if (Adiabaticc)
    {
      ComputeViscousTerms (vrad, vtheta, dens, 1);
      Substep3host(dens, dt);
      ActualiseGasEnergy (energy, energynew);
    }
    Transport (dens, vrad, vtheta, energy, label, dt);

    if (NINTERM * TimeStep == i) printf("step = %d\n",TimeStep );

  }

  FreePlanetary (sys);
  FreeForce (force);

  FreeArrays(dens, energy, label, vrad, vtheta);
  FreeCuda();

	return EXIT_SUCCESS;
}

__host__ void FreeCuda()
{
  cudaFree(Rmed_d);
  cudaFree(CellAbscissa_d);
  cudaFree(CellOrdinate_d);
  cudaFree(cosns_d);
  cudaFree(sinns_d);
  cudaFree(SoundSpeed_d);
  cudaFree(energy_d);
  cudaFree(AspectRatioRmed_d);
  cudaFree(dens_d);
  cudaFree(press_d);
  cudaFree(temperature_d);
  cudaFree(viscosity_array_d);
  cudaFree(SigmaInf_d);
  cudaFree(Rinf_d);
  cudaFree(label_d);
  cudaFree(Surf_d);
  cudaFree(vradint_d);
  cudaFree(pot_d);
  cudaFree(vrad_d);
  cudaFree(vthetaint_d);
  cudaFree(vtheta_d);
  cudaFree(invdiffRmed_d);
  cudaFree(invRinf_d);
  cudaFree(powRmed_d);
  cudaFree(densint_d);
  cudaFree(temperatureint_d);
  cudaFree(invdiffRsup_d);
  cudaFree(vradnew_d);
  cudaFree(vthetanew_d);
  cudaFree(energyint_d);
  cudaFree(cs0_d);
  cudaFree(cs1_d);
  cudaFree(csnrm1_d);
  cudaFree(csnrm2_d);
  cudaFree(mean_dens_d);
  cudaFree(mean_energy_d);
  cudaFree(mean_dens_d2);
  cudaFree(mean_energy_d2);
  cudaFree(forcesxi_d);
  cudaFree(forcesyi_d);
  cudaFree(forcesxo_d);
  cudaFree(forcesyo_d);
  cudaFree(qplus_d);
  cudaFree(energynew_d);
  cudaFree(EnergyMed_d);
  cudaFree(SigmaMed_d);
  cudaFree(CoolingTimeMed_d);
  cudaFree(QplusMed_d);
  cudaFree(qplus_d);
  cudaFree(ThetaMomP_d);
  cudaFree(ThetaMomM_d);
  cudaFree(RadMomP_d);
  cudaFree(RadMomM_d);
  cudaFree(Drr_d);
  cudaFree(Dpp_d);
  cudaFree(divergence_d);
  cudaFree(Drp_d);
  cudaFree(Trr_d);
  cudaFree(Tpp_d);
  cudaFree(Trp_d);
  cudaFree(extlabel_d);
  cudaFree(QStar_d);
  cudaFree(Qbase_d);
  cudaFree(GLOBAL_bufarray_d);
  cudaFree(gridfield_d);
  cudaFree(Radii_d);
  cudaFree(SGP_Kt_d);
  cudaFree(SGP_Kr_d);
  cudaFree(SGP_St_d);
  cudaFree(SGP_Sr_d);
  cudaFree(SGP_Kt_dc);
  cudaFree(SGP_Kr_dc);
  cudaFree(SGP_St_dc);
  cudaFree(SGP_Sr_dc);
  cudaFree(Gr_dc);
  cudaFree(Gphi_dc);
  cudaFree(Gr_d);
  cudaFree(Gphi_d);
  cudaFree(SG_Accr_d);
  cudaFree(SG_Acct_d);
}

__host__ void FreeArrays(float *dens, float *energy, float *label, float *vrad, float *vtheta)
{
  free(vt_cent);
  free(Rinf);
  free(Rmed);
  free(Rsup);
  free(Surf);
  free(invRinf);
  free(invSurf);
  free(invRmed);
  free(invdiffSurf);
  free(invdiffRsup);
  free(invdiffRmed);
  free(dens);
  free(energy);
  free(label);
  free(EnergyMed);
  free(SigmaMed);
  free(SigmaInf);
  free(vt_int);
  free(vrad);
  free(vtheta);
  free(GLOBAL_bufarray);
  free(vradint);
  free(pot);
  free(vthetaint);
  free(powRmed);
  free(temperatureint);
  free(densint);
  free(vradnew);
  free(vthetanew);
  free(energyint);
  free(energynew);
  free(forcesxi);
  free(forcesyi);
  free(forcesxo);
  free(forcesyo);
  free(QplusMed);
  free(CoolingTimeMed);
  free(viscosity_array);
  free(Drr);
  free(Dpp);
  free(divergence);
  free(Drp);
  free(Trr);
  free(Tpp);
  free(Trp);
  free(cs0);
  free(cs1);
  free(csnrm1);
  free(csnrm2);
  free(mean_dens);
  free(mean_dens2);
  free(mean_energy);
  free(mean_energy2);
  free(qplus);
  free(ThetaMomP);
  free(ThetaMomM);
  free(RadMomP);
  free(RadMomM);
  free(extlabel);
  free(QStar);
  free(Qbase);
  free(densStar);
  free(Radii);
  free(SG_Accr);
  free(SG_Acct);
  free(Gr);
  free(Gphi);

}

__host__ void DeviceToHostcudaMemcpy(float *dens, float *energy, float *label, float *temperature, float *vrad, float *vtheta)
{
  gpuErrchk(cudaMemcpy(dens, dens_d, size_grid2*sizeof(float), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(energy, energy_d, size_grid*sizeof(float), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(temperature, temperature_d, size_grid*sizeof(float), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(label, label_d, size_grid*sizeof(float), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(vrad, vrad_d, size_grid2*sizeof(float), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(vtheta, vtheta_d, size_grid2*sizeof(float), cudaMemcpyDeviceToHost));

}

__host__ void Substep1cudamalloc(float *vrad, float *vtheta)
{

  gpuErrchk(cudaMemcpy(vradint_d, vradint, size_grid2*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(pot_d, pot, size_grid*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(vrad_d, vrad, size_grid2*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(vthetaint_d, vthetaint, size_grid2*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(vtheta_d, vtheta, size_grid2*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(invdiffRmed_d, invdiffRmed, NRAD*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(invRinf_d, invRinf, NRAD*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(powRmed_d, powRmed, NRAD*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(invdiffRsup_d, invdiffRsup, NRAD*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(temperatureint_d, temperatureint, size_grid*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(densint_d, densint, size_grid2*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(vradnew_d, vradnew, size_grid2*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(vthetanew_d, vthetanew, size_grid2*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(energyint_d, energyint, size_grid*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(qplus_d, qplus, size_grid*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(energynew_d, energynew, size_grid*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(EnergyMed_d, EnergyMed, NRAD*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(SigmaMed_d, SigmaMed, NRAD*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(CoolingTimeMed_d, CoolingTimeMed, NRAD*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(QplusMed_d, QplusMed, NRAD*sizeof(float), cudaMemcpyHostToDevice));
}

__host__ float CircumPlanetaryMasshost(float *dens, float xpl, float ypl)
{
  float *mdcp0, *mdcp0_d;

  mdcp0 = (float *)malloc(size_grid*sizeof(float));

  gpuErrchk(cudaMalloc((void**)&Surf_d,NRAD*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&mdcp0_d,size_grid*sizeof(float)));

  gpuErrchk(cudaMemcpy(Surf_d, Surf, NRAD*sizeof(float), cudaMemcpyHostToDevice ));
  gpuErrchk(cudaMemcpy(mdcp0_d, mdcp0, size_grid*sizeof(float), cudaMemcpyHostToDevice));

  CircumPlanetaryMass<<<dimGrid2, dimBlock2>>> (dens_d, Surf_d, CellAbscissa_d, CellOrdinate_d, xpl, ypl, NRAD, NSEC, HillRadius, mdcp0_d);
  gpuErrchk(cudaDeviceSynchronize());

  // reduction mdcp
  mdcp = DeviceReduce(mdcp0_d, size_grid);
  cudaFree(mdcp0_d);
  free(mdcp0);

  return mdcp;
}


__host__ void Substep1host(float *dens, float *vrad, float *vtheta, float dt, int i)
{

  if(i == 0) Substep1cudamalloc(vrad, vtheta);

  Substep1<<<dimGrid2, dimBlock2>>>(press_d, dens_d, vradint_d, invdiffRmed_d,pot_d,Rinf_d,
    invRinf_d, vrad_d, vthetaint_d, vtheta_d, Rmed_d,  dt, NRAD, NSEC, OmegaFrame1, ZMPlus,
    IMPOSEDDISKDRIFT, SIGMASLOPE, powRmed_d);
  gpuErrchk(cudaDeviceSynchronize());


    /*if (SelfGravity){
      selfgravityupdate = YES;
      compute_selfgravity(Rho, VradInt, VthetaInt, dt, selfgravityupdate);
    }*/

  ComputeViscousTerms (vradint, vthetaint, dens, 0);
  //UpdateVelocitiesWithViscosity(vradint, vthetaint, dens, dt);

  //if (!Evanescent) ApplySubKeplerianBoundary(VthetaInt);


}

__host__ void Substep3host(float *dens, float dt)
{

  for (int i = 0; i < NRAD; i++) viscosity_array[i] = FViscosity(Rmed[i]);
  gpuErrchk(cudaMemcpy(viscosity_array_d, viscosity_array, (NRAD+1)*sizeof(float), cudaMemcpyHostToDevice));

  Substep3<<<dimGrid2, dimBlock2>>>(dens_d, qplus_d, viscosity_array_d, Trr_d, Trp_d , Tpp_d, divergence_d,
     NRAD, NSEC, Rmed_d, Cooling, energynew_d, dt, EnergyMed_d, SigmaMed_d, CoolingTimeMed_d, energy_d,
     ADIABATICINDEX, QplusMed_d);
  gpuErrchk(cudaDeviceSynchronize());
}

__host__ void ActualiseGasVtheta(float *vtheta, float *vthetanew)
{
  gpuErrchk(cudaMemcpy(vtheta_d, vthetanew_d, size_grid2*sizeof(float), cudaMemcpyDeviceToDevice));
  gpuErrchk(cudaDeviceSynchronize());
}

__host__ void ActualiseGasVrad(float *vrad, float *vradnew)
{
  gpuErrchk(cudaMemcpy(vrad_d, vradnew_d, size_grid2*sizeof(float), cudaMemcpyDeviceToDevice));
  gpuErrchk(cudaDeviceSynchronize());
}

__host__ void ActualiseGasEnergy(float *energy, float *energynew)
{
  gpuErrchk(cudaMemcpy(energy_d, energynew_d, size_grid*sizeof(float), cudaMemcpyDeviceToDevice));
  gpuErrchk(cudaDeviceSynchronize());
}

__host__ void CreateArrays()
{
  EnergyMed = (float *) malloc(sizeof(float)*NRAD);
  SigmaMed = (float *) malloc(sizeof(float)*NRAD);
  SigmaInf = (float *) malloc(sizeof(float)*NRAD);
  vt_int = (float *) malloc(sizeof(float)*NRAD);
  GLOBAL_bufarray = (float *)malloc(sizeof(float)*NRAD);
  vradint = (float *) malloc(sizeof(float)*size_grid2);
  pot = (float *) malloc(sizeof(float)*size_grid);
  vthetaint = (float *) malloc(sizeof(float)*size_grid2);
  powRmed = (float *) malloc(sizeof(float)*NRAD);
  temperatureint = (float *)malloc(sizeof(float)*size_grid);
  densint = (float *)malloc(sizeof(float)*size_grid2);
  vradnew = (float *)malloc(sizeof(float)*size_grid2);
  vthetanew = (float *)malloc(sizeof(float)*size_grid2);
  energyint = (float *)malloc(sizeof(float)*size_grid);
  energynew = (float *)malloc(sizeof(float)*size_grid);
  forcesxi = (float *)malloc(sizeof(float)*dimfxy);
  forcesyi = (float *)malloc(sizeof(float)*dimfxy);
  forcesxo = (float *)malloc(sizeof(float)*dimfxy);
  forcesyo = (float *)malloc(sizeof(float)*dimfxy);
  QplusMed = (float *)malloc(sizeof(float)*NRAD);
  CoolingTimeMed = (float *)malloc(sizeof(float)*NRAD);
  viscosity_array = (float *)malloc(sizeof(float)*(NRAD+1));
  Drr = (float *)malloc(sizeof(float)*size_grid);
  Dpp = (float *)malloc(sizeof(float)*size_grid);
  divergence = (float *)malloc(sizeof(float)*size_grid);
  Drp = (float *)malloc(sizeof(float)*size_grid);
  Trr = (float *)malloc(sizeof(float)*size_grid);
  Tpp = (float *)malloc(sizeof(float)*size_grid);
  Trp = (float *)malloc(sizeof(float)*size_grid);
  qplus = (float *)malloc(sizeof(float)*size_grid);
  ThetaMomP = (float *)malloc(sizeof(float)*size_grid);
  ThetaMomM = (float *)malloc(sizeof(float)*size_grid);
  RadMomP = (float *)malloc(sizeof(float)*size_grid);
  RadMomM = (float *)malloc(sizeof(float)*size_grid);
  extlabel = (float *)malloc(sizeof(float)*size_grid);
  densStar = (float *)malloc(sizeof(float)*size_grid2);
  QStar = (float *)malloc(sizeof(float)*size_grid);
  Qbase = (float *)malloc(sizeof(float)*size_grid);
  cs0 = (float *)malloc(sizeof(float)*NSEC);
  cs1 = (float *)malloc(sizeof(float)*NSEC);
  csnrm1 = (float *)malloc(sizeof(float)*NSEC);
  csnrm2 = (float *)malloc(sizeof(float)*NSEC);
  mean_dens = (float *)malloc(sizeof(float)*NSEC);
  mean_dens2 = (float *)malloc(sizeof(float)*NSEC);
  mean_energy = (float *)malloc(sizeof(float)*NSEC);
  mean_energy2 = (float *)malloc(sizeof(float)*NSEC);
  Gr = (float *)malloc(sizeof(float)*NRAD*NSEC);
  Gphi = (float *)malloc(sizeof(float)*NRAD*NSEC);
  SG_Accr = (float *)malloc(sizeof(float)*NRAD*NSEC);
  SG_Acct = (float *)malloc(sizeof(float)*NRAD*NSEC);
}

__host__ void cudamalloc(float *label, float *dens)
{
  gpuErrchk(cudaMalloc((void**)&Rmed_d, NRAD*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&dens_d, size_grid2*sizeof(float)));
  gpuErrchk(cudaMalloc(&forcesxi_d, dimfxy*sizeof(float)));
  gpuErrchk(cudaMalloc(&forcesxo_d, dimfxy*sizeof(float)));
  gpuErrchk(cudaMalloc(&forcesyi_d, dimfxy*sizeof(float)));
  gpuErrchk(cudaMalloc(&forcesyo_d, dimfxy*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&qplus_d,size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&energynew_d,size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&EnergyMed_d,NRAD*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&SigmaMed_d,NRAD*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&CoolingTimeMed_d,NRAD*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&QplusMed_d,NRAD*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&temperatureint_d,size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&densint_d,size_grid2*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&vradnew_d,size_grid2*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&vthetanew_d,size_grid2*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&energyint_d, size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&vradint_d, size_grid2*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&pot_d, size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&vthetaint_d, size_grid2*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&invdiffRmed_d, NRAD*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&invRinf_d,NRAD*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&powRmed_d,NRAD*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&invdiffRsup_d, NRAD*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&Drr_d,size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&Dpp_d,size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&divergence_d,size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&Drp_d,size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&Trr_d,size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&Tpp_d,size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&Trp_d,size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&Trp_d,size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&Rsup_d,NRAD*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&invRmed_d,NRAD*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&label_d, size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&gridfield_d, size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&GLOBAL_bufarray_d, NRAD*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&RadMomP_d,size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&RadMomM_d,size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&ThetaMomP_d,size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&ThetaMomM_d,size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&Qbase_d,size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&QStar_d,size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&extlabel_d,size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&cs0_d, NSEC*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&cs1_d, NSEC*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&csnrm1_d, NSEC*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&csnrm2_d, NSEC*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&mean_dens_d, NSEC*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&mean_energy_d, NSEC*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&mean_dens_d2, NSEC*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&mean_energy_d2, NSEC*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&SGP_Kt_d, 2*NRAD*NSEC*sizeof(cufftReal)));
  gpuErrchk(cudaMalloc((void**)&SGP_Kr_d, 2*NRAD*NSEC*sizeof(cufftReal)));
  gpuErrchk(cudaMalloc((void**)&SGP_St_d, 2*NRAD*NSEC*sizeof(cufftReal)));
  gpuErrchk(cudaMalloc((void**)&SGP_Sr_d, 2*NRAD*NSEC*sizeof(cufftReal)));
  gpuErrchk(cudaMalloc((void**)&SGP_Kt_dc, 2*NRAD*(NSEC/2 + 1)*sizeof(cufftComplex)));
  gpuErrchk(cudaMalloc((void**)&SGP_Kr_dc, 2*NRAD*(NSEC/2 + 1)*sizeof(cufftComplex)));
  gpuErrchk(cudaMalloc((void**)&SGP_St_dc, 2*NRAD*(NSEC/2 + 1)*sizeof(cufftComplex)));
  gpuErrchk(cudaMalloc((void**)&SGP_Sr_dc, 2*NRAD*(NSEC/2 + 1)*sizeof(cufftComplex)));
  gpuErrchk(cudaMalloc((void**)&Gr_dc, 2*NRAD*(NSEC/2 + 1)*sizeof(cufftComplex)));
  gpuErrchk(cudaMalloc((void**)&Gphi_dc, 2*NRAD*(NSEC/2 + 1)*sizeof(cufftComplex)));
  gpuErrchk(cudaMalloc((void**)&Gr_d, 2*NRAD*NSEC*sizeof(cufftReal)));
  gpuErrchk(cudaMalloc((void**)&Gphi_d, 2*NRAD*NSEC*sizeof(cufftReal)));
  gpuErrchk(cudaMalloc((void**)&Radii_d, (NRAD+1)*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&SG_Accr_d, NRAD*NSEC*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&SG_Acct_d, NRAD*NSEC*sizeof(float)));

  gpuErrchk(cudaMemcpy(Rmed_d, Rmed, NRAD*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(dens_d, dens, size_grid2*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(Drr_d, Drr, size_grid*sizeof(float), cudaMemcpyHostToDevice ));
  gpuErrchk(cudaMemcpy(Dpp_d, Dpp, size_grid*sizeof(float), cudaMemcpyHostToDevice ));
  gpuErrchk(cudaMemcpy(divergence_d, divergence, size_grid*sizeof(float), cudaMemcpyHostToDevice ));
  gpuErrchk(cudaMemcpy(Drp_d, Drp, size_grid*sizeof(float), cudaMemcpyHostToDevice ));
  gpuErrchk(cudaMemcpy(Trr_d, Trr, size_grid*sizeof(float), cudaMemcpyHostToDevice ));
  gpuErrchk(cudaMemcpy(Tpp_d, Tpp, size_grid*sizeof(float), cudaMemcpyHostToDevice ));
  gpuErrchk(cudaMemcpy(Trp_d, Trp, size_grid*sizeof(float), cudaMemcpyHostToDevice ));
  gpuErrchk(cudaMemcpy(Rsup_d, Rsup, NRAD*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(invRmed_d, invRmed, NRAD*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(label_d, label, size_grid*sizeof(float), cudaMemcpyHostToDevice ));
  gpuErrchk(cudaMemcpy(RadMomP_d, RadMomP, size_grid*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(RadMomM_d, RadMomM, size_grid*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(ThetaMomP_d, ThetaMomP, size_grid*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(ThetaMomM_d, ThetaMomM, size_grid*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(Qbase_d, Qbase, size_grid*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(QStar_d, QStar, size_grid*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(extlabel_d,extlabel, size_grid*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(cs0_d, cs0, NSEC*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(cs1_d, cs1, NSEC*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(csnrm1_d, csnrm1, NSEC*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(csnrm2_d, csnrm2, NSEC*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(mean_dens_d, mean_dens, NSEC*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(mean_energy_d, mean_energy, NSEC*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(mean_dens_d2, mean_dens2, NSEC*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(mean_energy_d2, mean_energy2, NSEC*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(Radii_d, Radii, (NRAD+1)*sizeof(float), cudaMemcpyHostToDevice));

}
