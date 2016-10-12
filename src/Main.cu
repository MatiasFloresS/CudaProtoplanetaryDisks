#include "Main.cuh"
#include "Fondam.cuh"

extern float *SigmaInf_d, *AspectRatioRmed_d, *vrad_d,  OMEGAFRAME, OmegaFrame1, *Pressure, *CellAbscissa, \
*CellOrdinate, HillRadius, PhysicalTimeInitial, PhysicalTime, CVNR, *CellAbscissa_d, *CellOrdinate_d, \
*Temperature_d, *energy_d, *Temperature, *vtheta_d, *Pressure_d, *Rinf_d, *SoundSpeed_d, \
*viscosity_array_d, *QStar_d, *vt_cent, THICKNESSSMOOTHING, G, *SoundSpeed, *AspectRatioRmed, *Kr_aux, *Kt_aux, \
*RadMomP, *RadMomM, *ThetaMomP, *ThetaMomM, *Work, *QRStar, *Extlabel, *dq, *RadMomP_d, *RadMomM_d, *ThetaMomP_d, \
*ThetaMomM_d, *Work_d, *QRStar_d, *Extlabel_d, *dq_d, *DivergenceVelocity, *DRP, *DRR, *DPP, *TAURR, *TAURP, *TAUPP, \
*DivergenceVelocity_d, *DRP_d, *DRR_d, *DPP_d, *TAURR_d, *TAURP_d, *TAUPP_d, *Radii2, *vt_cent_d, *Rinf, *Rmed, \
*Rsup, *Surf, *invRinf, *invSurf, *invdiffRsup, *invdiffRmed, *invRmed, *Radii, *TemperInt, *TemperInt_d, *Rhostar, \
*Rhostar_d, *VradInt, *VradInt_d;

float *SigmaMed, *SigmaInf, *EnergyMed, mdcp, ScalingFactor = 1.0, \
*forcesxi, *forcesyi, *forcesxo, *forcesyo, *pot, *vthetaint, *powRmed, *densint_d, \
*densint, *vradnew, *vthetanew, *energyint, *fieldsrc, *vt_int, *GLOBAL_bufarray, *Surf_d, \
*pot_d, *vthetaint_d, *invdiffRmed_d, *invRinf_d, *powRmed_d, *vthetanew_d, *vradnew_d, \
*energyint_d, *invdiffRsup_d, *CoolingTimeMed, *QplusMed , *viscosity_array, \
*qplus_d, *energynew_d, *EnergyMed_d, *SigmaMed_d, *CoolingTimeMed_d, *QplusMed_d, *energynew, *qplus, \
*densStar, *QStar, *Qbase, *gridfield_d, *GLOBAL_bufarray_d, \
*Rsup_d, *invRmed_d, *label_d, *QStar_d, *Qbase_d, \
*cs0, *cs1, *csnrm1, *csnrm2, *mean_dens, *mean_dens2, *mean_energy, *mean_energy2, *cs0_d, *cs1_d, *csnrm1_d, \
*csnrm2_d, *mean_dens_d, *mean_dens_d2, *mean_energy_d, *mean_energy_d2, *forcesxi_d, *forcesyi_d, *forcesxo_d, \
*forcesyo_d, dphi, invdphi, onethird, *SGP_Kr, *SGP_Kt, *Radii_d, *SGP_St, *SGP_Sr, *Rmed_d, *dens_d, *Kr_aux_d, \
*Kt_aux_d, *SG_Accr, *SG_Acct, *SG_Acct_d, *SG_Accr_d, *array, *array_d, SGP_eps, SGP_rstep, \
SGP_tstep, *mdcp0, *mdcp0_d;

extern int NRAD, NSEC, SelfGravity, Corotating, FREQUENCY, Adiabaticc, Cooling;
static int StillWriteOneOutput, InnerOutputCounter=0;
int nrad2pot, nsec2pot, blocksize = 32, size_grid, dimfxy=11, TimeStep = 0, NbRestart = 0, blocksize2 = 256, \
nrad2potSG;

bool verbose = false, Restart = false, TimeToWrite;

dim3 dimGrid2, dimBlock2, dimGrid, dimBlock, dimGrid3;
cufftHandle planf, planb;

cufftComplex *SGP_Kt_dc, *SGP_Kr_dc, *SGP_St_dc, *SGP_Sr_dc, *Gr_dc, *Gphi_dc, *Gr_d, *Gphi_d, *SGP_Kt_d, \
*SGP_Kr_d, *SGP_Sr_d, *SGP_St_d;

__host__ int main (int argc, char *argv[])
{
  bool disable = false, TimeInfo = false, Profiling = false, Stockholm = false, SGUpdate = NO;
  char ParameterFile[256];

  PlanetarySystem *sys;
  Force *force;

  float *label, *dens, *energy, *vrad, *vtheta, foostep = 0.;
  int i;

  if (argc == 1) PrintUsage (argv[0]);

  strcpy (ParameterFile, "");
  for (i = 1; i < argc; i++)
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

  ReadVariables(ParameterFile);

  size_grid = NRAD*NSEC;
  if(!IsPow2(NRAD)) nrad2pot = NearestPowerOf2(NRAD+1);
  if(!IsPow2(NSEC)) nsec2pot = NearestPowerOf2(NSEC);
  if(!IsPow2(2*NRAD)) nrad2potSG = NearestPowerOf2(2*NRAD);

/* Create cufftplan2d */
  if (SelfGravity == YES)
  {
    if ((cufftPlan2d(&planf, 2*NRAD, NSEC, CUFFT_C2C)) != CUFFT_SUCCESS)
    {
      printf("cufft plan error\n");
      exit(-1);
    }

    if ((cufftPlan2d(&planb, 2*NRAD, NSEC , CUFFT_C2C)) != CUFFT_SUCCESS)
    {
      printf("cufft plan error\n");
      exit(-1);
    }
  }

  /* dim gridsize and blocksize */
  dim3 dimG2( nsec2pot/blocksize, nrad2pot/blocksize );
  dim3 dimB2( blocksize, blocksize );
  dimGrid2 = dimG2;
  dimBlock2 = dimB2;

  dim3 dimG( nsec2pot/blocksize2, 1);
  dim3 dimB( blocksize2, 1);
  dimGrid = dimG;
  dimBlock = dimB;

  dim3 dimG3 (nsec2pot/blocksize, nrad2potSG/blocksize);
  dimGrid3 = dimG3;
  /*  ---------------------------  */

  dphi = 2.0*M_PI/(float)NSEC;
  invdphi = 1.0/dphi;
  onethird = 1.0/3.0;

  if (verbose == YES) TellEverything();
  if (disable == YES) exit(0);
  printf("Allocating arrays...\n");

  /* local arrays */

  dens   = (float *)malloc(size_grid*sizeof(float));
  vrad   = (float *)malloc(size_grid*sizeof(float));
  vtheta = (float *)malloc(size_grid*sizeof(float));
  energy = (float *)malloc(size_grid*sizeof(float));
  label  = (float *)malloc(size_grid*sizeof(float));

  /* global arrays */
  CreateArrays();

  printf("done.\n");

  FillPolar1DArrays ();

  SGP_eps = THICKNESSSMOOTHING * ASPECTRATIO;
  SGP_rstep = log(Radii[NRAD]/Radii[0])/(float)NRAD;
  SGP_tstep = 2.0*PI/(float)NSEC;

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
  if ( Adiabaticc == YES) InitGasEnergy (energy);

  Cudamalloc(label, dens, vrad, vtheta);

  if ( SelfGravity )
  {
    /* If SelfGravity = YES or Z, planets are initialized feeling disk
       potential. Only the surface density is required to calculate
       the radial self-gravity acceleration. The disk radial and
       azimutal velocities are not updated */

    Compute_selfgravity(dens, vrad, vtheta, foostep, SGUpdate);
    Init_planetarysys_withSG (sys);
  }

  ListPlanets (sys);
  OmegaFrame1 = OMEGAFRAME;

  if (Corotating) OmegaFrame1 = GetPsysInfo (sys, FREQUENCY);

  /* Only gas velocities remain to be initialized */
  Initialization (dens, vrad, vtheta, energy, label, sys);


  /* Initial gas_density is used to compute the circumplanetary mass with initial
     density field */
  mdcp = CircumPlanetaryMass (dens, sys);
  printf("mdcp = %.10f\n",mdcp );

  EmptyPlanetSystemFile (sys);
  PhysicalTimeInitial = PhysicalTime;

  MultiplyPolarGridbyConstant(dens);

  for (int i = 0; i <= 1000; i++)
  {
    InnerOutputCounter++;

    if (InnerOutputCounter == 1)
    {
      InnerOutputCounter = 0;
      WriteBigPlanetSystemFile (sys, TimeStep);
      UpdateLog(force, sys, dens, energy, TimeStep, PhysicalTime, dimfxy);
    }

    if (NINTERM * (TimeStep = (i / NINTERM)) == i)
    {
      /* Outputs are done here */
      TimeToWrite = YES;

      DeviceToHostcudaMemcpy(dens, energy, label, Temperature, vrad, vtheta); // Traigo los valores desde la GPU
      SendOutput (TimeStep, dens, vrad, vtheta, energy, label);
      WritePlanetSystemFile (sys, TimeStep);

    }
    else TimeToWrite = NO;

    AlgoGas(force, dens, vrad, vtheta, energy, label, sys, i);
    if (NINTERM * TimeStep == i) printf("step = %d\n",TimeStep );
  }
  FreePlanetary (sys);
  FreeForce (force);


 FreeCuda();
 FreeArrays(dens, vrad, vtheta, energy, label);

 printf("termino\n" );

  if (SelfGravity) // && !SGZeroMode
  {
      cufftDestroy(planf);
      cufftDestroy(planb);
  }
	return EXIT_SUCCESS;
}

__host__ void FreeCuda ()
{

  /* cudaFree FillPolar1DArrays */
  cudaFree(Radii_d);
  cudaFree(Rinf_d);
  cudaFree(Rmed_d);
  cudaFree(Rsup_d);
  cudaFree(Surf_d);
  cudaFree(invRinf_d);
  cudaFree(invRmed_d);
  cudaFree(invdiffRsup_d);
  cudaFree(invdiffRmed_d);
  cudaFree(vt_cent_d);
  cudaFree(Kr_aux_d);
  cudaFree(Kt_aux_d);

  /* cudaFree InitComputeAccel */
  cudaFree(CellAbscissa_d);
  cudaFree(CellOrdinate_d);

  /* cudaFree polar grid */
  cudaFree(dens_d);
  cudaFree(vrad_d);
  cudaFree(vtheta_d);
  cudaFree(energy_d);
  cudaFree(label_d);

  cudaFree(mdcp0_d);
  cudaFree(AspectRatioRmed_d);

  cudaFree(SoundSpeed_d);
  cudaFree(Pressure_d);
  cudaFree(Temperature_d);
  cudaFree(TemperInt_d);
  cudaFree(Rhostar_d);
  cudaFree(VradInt_d);
  cudaFree(vthetaint_d);

  cudaFree(viscosity_array_d);
  cudaFree(SigmaInf_d);

  cudaFree(pot_d);
  cudaFree(powRmed_d);
  cudaFree(densint_d);
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

  cudaFree(QStar_d);
  cudaFree(Qbase_d);
  cudaFree(GLOBAL_bufarray_d);
  cudaFree(gridfield_d);
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
  cudaFree(array_d);

  /* cudaFree InitTransport */
  cudaFree(RadMomP_d);
  cudaFree(RadMomM_d);
  cudaFree(ThetaMomP_d);
  cudaFree(ThetaMomM_d);
  cudaFree(Work_d);
  cudaFree(QRStar_d);
  cudaFree(Extlabel_d);
  cudaFree(dq_d);

  /* cudaFree InitViscosity */
  cudaFree(DivergenceVelocity_d);
  cudaFree(DRP_d);
  cudaFree(DRR_d);
  cudaFree(DPP_d);
  cudaFree(TAURR_d);
  cudaFree(TAURP_d);
  cudaFree(TAUPP_d);
}

__host__ void FreeArrays (float *dens, float *vrad, float *vtheta, float *energy, float *label)
{
  /* free FillPolar1DArrays */
  free(Radii);
  free(Radii2);
  free(Rinf);
  free(Rmed);
  free(Rsup);
  free(Surf);
  free(invRinf);
  free(invSurf);
  free(invRmed);
  free(invdiffRsup);
  free(invdiffRmed);
  free(vt_cent);
  free(Kr_aux);
  free(Kt_aux);

  /* free InitComputeAccel */
  cudaFree(CellAbscissa);
  cudaFree(CellOrdinate);

  /* free polar grid */
  free(dens);
  free(vrad);
  free(vtheta);
  free(energy);
  free(label);

  free(TemperInt);
  free(Temperature);
  free(Pressure);
  free(SoundSpeed);
  free(Rhostar);
  free(VradInt);
  free(vthetaint);

  free(array);
  free(mdcp0);
  free(EnergyMed);
  free(SigmaMed);
  free(SigmaInf);
  free(vt_int);
  free(GLOBAL_bufarray);
  free(pot);
  free(powRmed);
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

  free(cs0);
  free(cs1);
  free(csnrm1);
  free(csnrm2);
  free(mean_dens);
  free(mean_dens2);
  free(mean_energy);
  free(mean_energy2);
  free(qplus);
  free(QStar);
  free(Qbase);
  free(densStar);

  free(SG_Accr);
  free(SG_Acct);

  /* free InitTransport */
  free(RadMomP);
  free(RadMomM);
  free(ThetaMomP);
  free(ThetaMomM);
  free(Work);
  free(QRStar);
  free(Extlabel);
  free(dq);

  /* free InitViscosity */
  free(DivergenceVelocity);
  free(DRP);
  free(DRR);
  free(DPP);
  free(TAURR);
  free(TAURP);
  free(TAUPP);

}

__host__ void DeviceToHostcudaMemcpy (float *dens, float *energy, float *label, float *Temperature, float *vrad, float *vtheta)
{
  gpuErrchk(cudaMemcpy(dens, dens_d,               size_grid*sizeof(float), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(energy, energy_d,           size_grid*sizeof(float), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(Temperature, Temperature_d, size_grid*sizeof(float), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(label, label_d,             size_grid*sizeof(float), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(vrad, vrad_d,               size_grid*sizeof(float), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(vtheta, vtheta_d,           size_grid*sizeof(float), cudaMemcpyDeviceToHost));
}

__host__ float CircumPlanetaryMass (float *dens, PlanetarySystem *sys)
{
  float xpl, ypl;

  xpl = sys->x[0];
  ypl = sys->y[0];

  CircumPlanetaryMassKernel<<<dimGrid2, dimBlock2>>> (dens_d, Surf_d, CellAbscissa_d, CellOrdinate_d, xpl, ypl, NRAD, NSEC, \
    HillRadius, mdcp0_d);
  gpuErrchk(cudaDeviceSynchronize());

  // reduction mdcp
  mdcp = DeviceReduce(mdcp0_d, size_grid);

  return mdcp;
}

__host__ void CreateArrays ()
{
  EnergyMed       = (float *)malloc(NRAD*sizeof(float));
  SigmaMed        = (float *)malloc(NRAD*sizeof(float));
  SigmaInf        = (float *)malloc(NRAD*sizeof(float));
  vt_int          = (float *)malloc(NRAD*sizeof(float));
  GLOBAL_bufarray = (float *)malloc(NRAD*sizeof(float));
  powRmed         = (float *)malloc(NRAD*sizeof(float));
  QplusMed        = (float *)malloc(NRAD*sizeof(float));
  CoolingTimeMed  = (float *)malloc(NRAD*sizeof(float));
  cs0             = (float *)malloc(NSEC*sizeof(float));
  cs1             = (float *)malloc(NSEC*sizeof(float));
  csnrm1          = (float *)malloc(NSEC*sizeof(float));
  csnrm2          = (float *)malloc(NSEC*sizeof(float));
  mean_dens       = (float *)malloc(NSEC*sizeof(float));
  mean_dens2      = (float *)malloc(NSEC*sizeof(float));
  mean_energy     = (float *)malloc(NSEC*sizeof(float));
  mean_energy2    = (float *)malloc(NSEC*sizeof(float));
  forcesxi        = (float *)malloc(dimfxy*sizeof(float));
  forcesyi        = (float *)malloc(dimfxy*sizeof(float));
  forcesxo        = (float *)malloc(dimfxy*sizeof(float));
  forcesyo        = (float *)malloc(dimfxy*sizeof(float));
  viscosity_array = (float *)malloc((NRAD+1)*sizeof(float));
  pot             = (float *)malloc(size_grid*sizeof(float));
  vthetaint       = (float *)malloc(size_grid*sizeof(float));
  densint         = (float *)malloc(size_grid*sizeof(float));
  vradnew         = (float *)malloc(size_grid*sizeof(float));
  vthetanew       = (float *)malloc(size_grid*sizeof(float));
  energyint       = (float *)malloc(size_grid*sizeof(float));
  energynew       = (float *)malloc(size_grid*sizeof(float));
  qplus           = (float *)malloc(size_grid*sizeof(float));
  densStar        = (float *)malloc(size_grid*sizeof(float));
  QStar           = (float *)malloc(size_grid*sizeof(float));
  Qbase           = (float *)malloc(size_grid*sizeof(float));
  SG_Accr         = (float *)malloc(size_grid*sizeof(float));
  SG_Acct         = (float *)malloc(size_grid*sizeof(float));
  array           = (float *)malloc(size_grid*sizeof(float));
  mdcp0           = (float *)malloc(size_grid*sizeof(float));

}

__host__ void Cudamalloc (float *label, float *dens, float *vrad, float *vtheta)
{
  gpuErrchk(cudaMalloc((void**)&Radii_d, (NRAD+1)*sizeof(float)));

  gpuErrchk(cudaMalloc(&forcesxi_d, dimfxy*sizeof(float)));
  gpuErrchk(cudaMalloc(&forcesxo_d, dimfxy*sizeof(float)));
  gpuErrchk(cudaMalloc(&forcesyi_d, dimfxy*sizeof(float)));
  gpuErrchk(cudaMalloc(&forcesyo_d, dimfxy*sizeof(float)));

  gpuErrchk(cudaMalloc((void**)&EnergyMed_d,       NRAD*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&SigmaMed_d,        NRAD*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&CoolingTimeMed_d,  NRAD*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&QplusMed_d,        NRAD*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&GLOBAL_bufarray_d, NRAD*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&Rmed_d,            NRAD*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&Rinf_d,            NRAD*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&invRmed_d,         NRAD*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&Rsup_d,            NRAD*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&invdiffRmed_d,     NRAD*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&invRinf_d,         NRAD*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&powRmed_d,         NRAD*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&invdiffRsup_d,     NRAD*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&Surf_d,            NRAD*sizeof(float)));

  gpuErrchk(cudaMalloc((void**)&cs0_d,          NSEC*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&cs1_d,          NSEC*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&csnrm1_d,       NSEC*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&csnrm2_d,       NSEC*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&mean_dens_d,    NSEC*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&mean_energy_d,  NSEC*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&mean_dens_d2,   NSEC*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&mean_energy_d2, NSEC*sizeof(float)));

  gpuErrchk(cudaMalloc((void**)&qplus_d,          size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&energynew_d,      size_grid*sizeof(float)));

  gpuErrchk(cudaMalloc((void**)&densint_d,        size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&vradnew_d,        size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&vthetanew_d,      size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&energyint_d,      size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&pot_d,            size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&vthetaint_d,      size_grid*sizeof(float)));

  gpuErrchk(cudaMalloc((void**)&label_d,          size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&gridfield_d,      size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&Qbase_d,          size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&QStar_d,          size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&SG_Accr_d,        size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&SG_Acct_d,        size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&vrad_d,           size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&vtheta_d,         size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&dens_d,           size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&array_d,          size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&mdcp0_d,          size_grid*sizeof(float)));

  gpuErrchk(cudaMalloc((void**)&Kr_aux_d, 2*size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&Kt_aux_d, 2*size_grid*sizeof(float)));

  gpuErrchk(cudaMalloc((void**)&SGP_Kt_d,  2*size_grid*sizeof(cufftComplex)));
  gpuErrchk(cudaMalloc((void**)&SGP_Kr_d,  2*size_grid*sizeof(cufftComplex)));
  gpuErrchk(cudaMalloc((void**)&SGP_St_d,  2*size_grid*sizeof(cufftComplex)));
  gpuErrchk(cudaMalloc((void**)&SGP_Sr_d,  2*size_grid*sizeof(cufftComplex)));
  gpuErrchk(cudaMalloc((void**)&SGP_Kt_dc, 2*size_grid*sizeof(cufftComplex)));
  gpuErrchk(cudaMalloc((void**)&SGP_Kr_dc, 2*size_grid*sizeof(cufftComplex)));
  gpuErrchk(cudaMalloc((void**)&SGP_St_dc, 2*size_grid*sizeof(cufftComplex)));
  gpuErrchk(cudaMalloc((void**)&SGP_Sr_dc, 2*size_grid*sizeof(cufftComplex)));
  gpuErrchk(cudaMalloc((void**)&Gr_dc,     2*size_grid*sizeof(cufftComplex)));
  gpuErrchk(cudaMalloc((void**)&Gphi_dc,   2*size_grid*sizeof(cufftComplex)));
  gpuErrchk(cudaMalloc((void**)&Gr_d,      2*size_grid*sizeof(cufftComplex)));
  gpuErrchk(cudaMalloc((void**)&Gphi_d,    2*size_grid*sizeof(cufftComplex)));

  gpuErrchk(cudaMemcpy(Radii_d, Radii,             (NRAD+1)*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(Rmed_d, Rmed,               NRAD*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(Rinf_d, Rinf,               NRAD*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(invRmed_d, invRmed,         NRAD*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(Rsup_d, Rsup,               NRAD*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(invdiffRmed_d, invdiffRmed, NRAD*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(invRinf_d, invRinf,         NRAD*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(powRmed_d, powRmed,         NRAD*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(invdiffRsup_d, invdiffRsup, NRAD*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(Surf_d, Surf,               NRAD*sizeof(float), cudaMemcpyHostToDevice));

  gpuErrchk(cudaMemcpy(dens_d, dens,   size_grid*sizeof(float), cudaMemcpyHostToDevice));

  gpuErrchk(cudaMemcpy(Kr_aux_d, Kr_aux, 2*size_grid*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(Kt_aux_d, Kt_aux, 2*size_grid*sizeof(float), cudaMemcpyHostToDevice));

}

__host__ void ExecuteExeC2Cforward ()
{
  if ((cufftExecC2C(planf, (cufftComplex *)SGP_St_d, (cufftComplex *)SGP_St_dc, CUFFT_FORWARD))!= \
  CUFFT_SUCCESS)
  {
    printf("cufft execC2C error\n");
    exit(-1);
  }
  gpuErrchk(cudaDeviceSynchronize());

  if ((cufftExecC2C(planf,(cufftComplex *)SGP_Kt_d, (cufftComplex *)SGP_Kt_dc, CUFFT_FORWARD))!= \
  CUFFT_SUCCESS)
  {
    printf("cufft execC2C error\n");
    exit(-1);
  }
  gpuErrchk(cudaDeviceSynchronize());

  if ((cufftExecC2C(planf, (cufftComplex *)SGP_Sr_d, (cufftComplex *)SGP_Sr_dc, CUFFT_FORWARD))!= \
    CUFFT_SUCCESS)
  {
      printf("cufft execC2C error\n");
      exit(-1);
  }
  gpuErrchk(cudaDeviceSynchronize());

  if ((cufftExecC2C(planf, (cufftComplex *)SGP_Kr_d, (cufftComplex *)SGP_Kr_dc, CUFFT_FORWARD))!= \
    CUFFT_SUCCESS)
  {
      printf("cufft execC2C error\n");
      exit(-1);
  }
  gpuErrchk(cudaDeviceSynchronize());

}

__host__ void ExecuteExeC2Cbackward ()
{
  if ((cufftExecC2C(planb, (cufftComplex *)Gphi_dc, (cufftComplex *)Gphi_d, CUFFT_INVERSE))!= \
    CUFFT_SUCCESS)
  {
    printf("cufft execC2C error\n");
    exit(-1);
  }
  gpuErrchk(cudaDeviceSynchronize());

  if ((cufftExecC2C(planb, (cufftComplex *)Gr_dc, (cufftComplex *)Gr_d, CUFFT_INVERSE))!= \
    CUFFT_SUCCESS)
  {
    printf("cufft execC2C error\n");
    exit(-1);
  }
  gpuErrchk(cudaDeviceSynchronize());
}

__host__ void Init_planetarysys_withSG (PlanetarySystem *sys)
{
  // !SGZeroMode case

  gpuErrchk(cudaMemcpy(SG_Accr, SG_Accr_d, size_grid*sizeof(float), cudaMemcpyDeviceToHost));
  Make1Dprofile (SG_Accr);

  /* Planetary system initialization in self-gravity cases:
     planets are put in a fixed circular orbit, we need to know radial sg acceleration
     felt by planets.*/

  float r, dist, ri, rip1, dr, sgacc;
  int ipl;

  for (int k = 0; k < sys->nb; k++)
  {
    r = sys->x[k];
    /* dist denotes the planet's semi-major axis */
    dist = (float) (r / (1. + ECCENTRICITY));
    ipl = 0;
    while (Rmed[ipl] <= dist) ipl++;
    ri = Rmed[ipl];
    rip1 = Rmed[ipl+1];
    dr = rip1 - ri;
    sgacc = (dist - ri)*GLOBAL_bufarray[ipl+1] + (rip1 - dist)*GLOBAL_bufarray[ipl];
    sgacc /= dr;

    /* sgacc is the radial sg acc. at the planet's semi-major axis */
    sys->vy[k] *= (float) sqrt(1. - dist*dist*sgacc);
  }
}

__host__ void Compute_selfgravity (float *dens, float *vrad, float *vtheta, float DeltaT, bool SGUpdate)
{
  Fft();
  ExecuteExeC2Cforward();
  Fftmul();
  ExecuteExeC2Cbackward();
  Sg_Acc();
  if (SGUpdate) Update_sgvelocity(DeltaT);
}

__host__ void MultiplyPolarGridbyConstant (float *dens)
{
  MultiplyPolarGridbyConstantKernel<<<dimGrid2, dimBlock2>>>(dens_d, NRAD, NSEC, ScalingFactor);
  gpuErrchk(cudaDeviceSynchronize());
}

__host__ void Fftmul ()
{
  FftmulKernel<<<dimGrid3, dimBlock2>>>(Gr_dc, Gphi_dc, SGP_Kr_dc, SGP_Kt_dc, SGP_Sr_dc, SGP_St_dc,
    NSEC, G, NRAD);
  gpuErrchk(cudaDeviceSynchronize());
}

__host__ void Fft ()
{
  FftKernel<<<dimGrid3, dimBlock2>>>(Radii_d, SGP_Kr_d, SGP_Kt_d, SGP_eps, NRAD, NSEC, SGP_Sr_d,
    SGP_St_d, dens_d, Rmed_d, Kr_aux_d, Kt_aux_d);
  gpuErrchk(cudaDeviceSynchronize());
}

__host__ void Sg_Acc ()
{
  Sg_AccKernel<<<dimGrid2, dimBlock2>>>(SG_Accr_d, SG_Acct_d, dens_d, SGP_rstep, SGP_tstep, SGP_eps,
    NRAD, NSEC, Rmed_d, Gr_d, Gphi_d, G);
  gpuErrchk(cudaDeviceSynchronize());
}

__host__ void Update_sgvelocity (float DeltaT)
{
  Update_sgvelocityKernel <<<dimGrid2, dimBlock2>>>(vrad_d, vtheta_d, SG_Accr_d, SG_Acct_d, Rinf_d, Rmed_d,
    invdiffRmed_d, DeltaT , NRAD,  NSEC);
  gpuErrchk(cudaDeviceSynchronize());
}
