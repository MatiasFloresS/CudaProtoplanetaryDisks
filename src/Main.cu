#include "Main.cuh"
#include "Param.cuh"

/* extern double device arrays */
extern double *SigmaInf_d, *Vrad_d, *CellAbscissa_d, *CellOrdinate_d;
extern double *Temperature_d, *Energy_d, *Vtheta_d, *Pressure_d, *SoundSpeed_d;
extern double *viscosity_array_d, *QStar_d, *ExtLabel_d, *dq_d, *DRP_d, *vt_cent_d;
extern double *RadMomP_d, *RadMomM_d, *ThetaMomP_d, *ThetaMomM_d, *Work_d, *QRStar_d;
extern double *DivergenceVelocity_d, *DRR_d, *DPP_d, *TAURR_d, *TAURP_d, *TAUPP_d;
extern double *TemperInt_d, *DensStar_d, *VradInt_d, *LostByDisk_d, *VthetaRes_d, *VMed_d;
extern double *TempShift_d, *Vmoy_d, *newDT_d, *DT1D_d, *DT2D_d, *Vresidual_d, *Vazimutal_d;
extern double *Vradial_d;

/* extern double values */
extern double OMEGAFRAME;
extern double PhysicalTimeInitial, PhysicalTime;
extern double THICKNESSSMOOTHING;

/* extern double host arrays */
extern double *Pressure, *CellAbscissa, *CellOrdinate, *Temperature, *vt_cent;
extern double *SoundSpeed, *Kr_aux, *Kt_aux, *RadMomP, *RadMomM;
extern double *ThetaMomP, *ThetaMomM, *Work, *QRStar, *ExtLabel, *dq, *DivergenceVelocity;
extern double *DRP, *DRR, *DPP, *TAURR, *TAURP, *TAUPP, *Radii;
extern double *Surf, *invSurf, *powRmed;
extern double *DensStar, *VradInt,  *TemperInt, *Potential, *VthetaInt, *DensInt, *VradNew;
extern double *VthetaNew, *EnergyInt, *EnergyNew, *VthetaRes, *TempShift;
double *example;

/* double host arrays */
double *SigmaMed, *SigmaInf, *EnergyMed;
double *fieldsrc, *vt_int, *GLOBAL_bufarray, *CoolingTimeMed, *QplusMed , *viscosity_array;
double *cs1, *Qplus, *QStar, *Qbase, *cs0, *csnrm1, *csnrm2, *mean_dens, *mean_dens2;
double *mean_energy, *mean_energy2, *array, *mdcp0;
double *SG_Accr, *SG_Acct, *GLOBAL_AxiSGAccr;

/* double device arrays */
double *DensInt_d, *Surf_d, *Potential_d, *VthetaInt_d, *powRmed_d, *invSurf_d;
double *VthetaNew_d, *VradNew_d, *EnergyInt_d, *Qplus_d, *EnergyNew_d;
double *EnergyMed_d, *SigmaMed_d, *CoolingTimeMed_d, *QplusMed_d, *gridfield_d, *GLOBAL_bufarray_d;
double *Label_d, *QStar_d, *Qbase_d, *Qbase2_d, *cs0_d, *cs1_d, *csnrm1_d, *csnrm2_d, *mean_dens_d;
double *mean_dens_d2, *mean_energy_d, *mean_energy_d2;
double *SGP_Kr, *SGP_Kt, *Radii_d, *SGP_St, *SGP_Sr, *Dens_d, *fxi_d, *fxo_d, *fyi_d, *fyo_d;
double *Kr_aux_d, *Kt_aux_d, *SG_Acct_d, *SG_Accr_d, *array_d, *mdcp0_d, *axifield_d, *GLOBAL_AxiSGAccr_d;

double mdcp, SGP_tstep, SGP_eps, SGP_rstep;

extern double HillRadius;
extern double *invdiffRmed, *Rinf_d, *Rinf, *invRinf, *Rmed, *invRmed, *Rsup, *invdiffRsup;

double OmegaFrame;
double *invdiffRmed_d, *invRinf_d, *invRmed_d, *Rmed_d, *Rsup_d, *invdiffRsup_d;

double *q0, *PlanetMasses, *q1;

extern int NRAD, NSEC, Cooling;
extern int *NoSplitAdvection_d;
extern int *Nshift_d;

int nrad2pot, nsec2pot, size_grid, nrad2potSG, nsec2potplus, *CFL_d, *CFL;
int blocksize2D = 16;
int blocksize1D = 256;

int         TimeToWrite, Restart = NO; // OpenInner = NO;
int             TimeStep = 0, NbRestart = 0, verbose = NO;
int             dimfxy = 11;
static int      InnerOutputCounter = 0, StillWriteOneOutput;
extern int  Corotating;
extern int  SelfGravity, SGZeroMode, Adiabatic;
double           ScalingFactor = 1.0;

dim3 dimGrid2, dimBlock2, dimGrid, dimBlock, dimGrid3, dimGrid4;

cufftHandle planf, planb;

cufftComplex *SGP_Kt_dc, *SGP_Kr_dc, *SGP_St_dc, *SGP_Sr_dc, *Gr_dc, *Gphi_dc, *Gr_d, *Gphi_d, *SGP_Kt_d,       \
*SGP_Kr_d, *SGP_Sr_d, *SGP_St_d;



__host__ int main (int argc, char *argv[])
{
  int device;
  printf("enter gpu device: ");
  scanf("%d", &device);

  //cudaSetDevice(1); Using gpu nvidia m4000 8 gb
  cudaSetDevice(device); // Using gpu nvidia m4000 8gb

  double     *Dens;
  double     *Vrad;
  double     *Vtheta;
  double     *Energy;
  double     *Label;
  int       i;
  double     foostep = 0.;
  int   disable = NO, TimeInfo = NO, Profiling = NO;
  int   Stockholm = NO, SGUpdate = NO;
  char      ParameterFile[256];
  char      configplanet[100];
  PlanetarySystem *sys;
  Force *force;

  if (argc == 1) PrintUsage (argv[0]);
  strcpy (ParameterFile, "");
  for (i = 1; i < argc; i++){
    if (*(argv[i]) == '-'){
      if (strspn (argv[i], "-secndovtpfamzib0123456789") != strlen (argv[i]))
        PrintUsage (argv[0]);
      if (strchr (argv[i], 'n'))
        disable = YES;
      if (strchr (argv[i], 'e'))
        Stockholm = YES;
      if (strchr (argv[i], 'v'))
        verbose = YES;
      if (strchr (argv[i], 't'))
        TimeInfo = YES;
      if (strchr (argv[i], 'c'))
        SloppyCFL = YES;
      if (strchr (argv[i], 'p'))
        Profiling = YES;
      if (strchr (argv[i], 'd'))
        debug = YES;
      if (strchr (argv[i], 'b'))
        CentrifugalBalance = YES;
      if (strchr (argv[i], 'm'))
        Merge = YES;
      if (strchr (argv[i], 'a'))
        MonitorIntegral = YES;
      if (strchr (argv[i], 'z'))
        FakeSequential = YES;
      if (strchr (argv[i], 'i')){
        StoreSigma = YES;
	      if (Adiabatic)
          StoreEnergy = YES;
      }
      if (strchr (argv[i], '0'))
        OnlyInit = YES;
      if ((argv[i][1] >= '1') && (argv[i][1] <= '9')){
	       GotoNextOutput = YES;
	       StillWriteOneOutput = (int)(argv[i][1]-'0');
      }
      if (strchr (argv[i], 's')){
        Restart = YES;
	      i++;
	      NbRestart = atoi(argv[i]);
	      if ((NbRestart < 0)){
          printf ("Incorrect restart number\n");
	        PrintUsage (argv[0]);
        }
      }
      if (strchr (argv[i], 'o')){
        OverridesOutputdir = YES;
	      i++;
	      sprintf (NewOutputdir, "%s", argv[i]);
      }
      else {
        if (strchr (argv[i], 'f')){
	        i++;
	        ScalingFactor = atof(argv[i]);
	        printf ("Scaling factor = %g\n", ScalingFactor);
	        if ((ScalingFactor <= 0)){
	          printf ("Incorrect scaling factor\n");
	          PrintUsage (argv[0]);
          }
        }
      }
    }
    else strcpy (ParameterFile, argv[i]);
  }
  if ((StoreSigma || StoreEnergy) && !(Restart)){
    printf ("You cannot use tabulated surface density\n");
    printf ("or surface internal energy in a non-restart run.\n");
    printf ("Aborted\n");
    exit (0);
  }
  if (ParameterFile[0] == 0) PrintUsage (argv[0]);
  ReadVariables(ParameterFile);

  /* Si elige la opcion SelfGravity, se crean los planes 2D de la cufft */
  if (SelfGravity){
    if ((cufftPlan2d(&planf, 2*NRAD, NSEC, CUFFT_C2C)) != CUFFT_SUCCESS){
      printf("cufft plan error\n");
      exit(-1);
    }

    if ((cufftPlan2d(&planb, 2*NRAD, NSEC , CUFFT_C2C)) != CUFFT_SUCCESS){
      printf("cufft plan error\n");
      exit(-1);
    }
  }

  /* size grid */
  size_grid = (NRAD+1)*NSEC;

  /* aca falta ordenar --------------------------------------->*/
  if(!IsPow2(NRAD+1)) nrad2pot = NearestPowerOf2(NRAD+1);
  if(!IsPow2(NSEC)) nsec2pot = NearestPowerOf2(NSEC);
  if(!IsPow2(2*(NRAD+1))) nrad2potSG = NearestPowerOf2(2*(NRAD+1));

  /* dim gridsize and blocksize of */

  dim3 dimG( nsec2pot/blocksize1D, 1);
  dim3 dimB( blocksize1D, 1);
  dimGrid = dimG;
  dimBlock = dimB;

  dim3 dimG2( nsec2pot/blocksize2D, nrad2pot/blocksize2D);
  dim3 dimB2( blocksize2D, blocksize2D );
  dimGrid2 = dimG2;
  dimBlock2 = dimB2;


  dim3 dimG3 (nsec2pot/blocksize2D, nrad2potSG/blocksize2D);
  dimGrid3 = dimG3;

  dim3 dimG4 (nrad2pot/blocksize1D, 1);
  dimGrid4 = dimG4;

  if (verbose == YES)
    TellEverything();
  if (disable == YES)
    exit(0);
  printf("Allocating arrays...\n");
  /* local arrays */
  Dens   = (double *)malloc(size_grid*sizeof(double));
  Vrad   = (double *)malloc(size_grid*sizeof(double));
  Vtheta = (double *)malloc(size_grid*sizeof(double));
  Energy = (double *)malloc(size_grid*sizeof(double));
  Label  = (double *)malloc(size_grid*sizeof(double));

  /* global arrays */
  CreateArrays();

  printf("done.\n");
  FillPolar1DArrays ();
  force = AllocateForce (dimfxy);

  /* string to char configplanet */
  strncpy(configplanet, PLANETCONFIG.c_str(), sizeof(configplanet));
  configplanet[sizeof(configplanet)-1]=0;

  /* Here planets are initialized feeling star potential but they do
     not feel disk potential  */
  sys = InitPlanetarySystem (configplanet);

  /* Gas density initialization */
  InitGasDensity (Dens);

  /* If energy equation is taken into account, we initialize the gas
     thermal energy  */
  if (Adiabatic)
    InitGasEnergy (Energy);

  Cudamalloc(Label, Dens, Vrad, Vtheta);

  if (SelfGravity){
    SGP_eps = THICKNESSSMOOTHING * ASPECTRATIO;
    SGP_rstep = (double)(log(Radii[NRAD]/Radii[0])/(double)NRAD);
    SGP_tstep = 2.0*PI/(double)NSEC;

    /* If SelfGravity = YES or Z, planets are initialized feeling disk
       potential. Only the surface density is required to calculate
       the radial self-gravity acceleration. The disk radial and
       azimutal velocities are not updated */

    /* Here we copy Vrad to Vradial -> device to device */
    gpuErrchk(cudaMemcpy(Vradial_d, Vrad_d, size_grid*sizeof(double), cudaMemcpyDeviceToDevice));
    /* Here we copy Vtheta to Vazimutal -> device to device */
    gpuErrchk(cudaMemcpy(Vazimutal_d, Vtheta_d, size_grid*sizeof(double), cudaMemcpyDeviceToDevice));

    compute_selfgravity(Dens, foostep, SGUpdate, 1);

    /* Here we copy Vradial to Vrad -> device to device */
    gpuErrchk(cudaMemcpy(Vrad_d, Vradial_d, size_grid*sizeof(double), cudaMemcpyDeviceToDevice));
    /* Here we copy Vazimutal to Vtheta -> device to device */
    gpuErrchk(cudaMemcpy(Vtheta_d, Vazimutal_d, size_grid*sizeof(double), cudaMemcpyDeviceToDevice));
    Init_planetarysys_withSG (sys);
  }

  ListPlanets (sys);
  OmegaFrame = OMEGAFRAME;
  if (Corotating) OmegaFrame = GetPsysInfo (sys, FREQUENCY);

  printf("OmegaFrame%.15g\n",OmegaFrame );

  /* Only gas velocities remain to be initialized */
  Initialization (Dens, Vrad, Vtheta, Energy, Label, sys);

  /* Initial gas_density is used to compute the circumplanetary mass with initial
     density field */

  mdcp = CircumPlanetaryMass (Dens, sys);

  EmptyPlanetSystemFile (sys);
  PhysicalTimeInitial = PhysicalTime;

  //MultiplyPolarGridbyConstant(Dens);

  for (int i = 0; i <= NTOT; i++){
    InnerOutputCounter++;

    if (InnerOutputCounter == 1){
      InnerOutputCounter = 0;
      //WriteBigPlanetSystemFile (sys, TimeStep);
      //UpdateLog(force, sys, Dens, Energy, TimeStep, PhysicalTime, dimfxy);
    }

    if (NINTERM * (TimeStep = (i / NINTERM)) == i){
      /* Outputs are done here */
      printf("%d\n", i);
      TimeToWrite = YES;
      //DeviceToHostcudaMemcpy(Dens, Energy, Label, Temperature, Vrad, Vtheta); // Traigo los valores desde la GPU
      //SendOutput (TimeStep, Dens, Vrad, Vtheta, Energy, Label);
      //WritePlanetSystemFile (sys, TimeStep);
    }
    else TimeToWrite = NO;
    /* Algorithm loop begins here *
    /***********************/
    /* Hydrodynamical Part */
    /***********************/

    AlgoGas(force, Dens, Vrad, Vtheta, Energy, Label, sys, i);
  }
  //DeviceToHostcudaMemcpy(Dens, Energy, Label, Temperature, Vrad, Vtheta); // Traigo los valores desde la GPU

  //gpuErrchk(cudaMemcpy(Pressure, Pressure_d,           size_grid*sizeof(double), cudaMemcpyDeviceToHost));

  /*binFile(Vrad, NRAD*NSEC, "gvrad");
  binFile(Vtheta, NRAD*NSEC, "gvtheta");
  binFile(Dens, NRAD*NSEC, "gdens");
  binFile(Energy, NRAD*NSEC, "genergy");
  binFile(Temperature, NRAD*NSEC, "gtemp");
  binFile(Pressure, NRAD*NSEC, "gpress");
*/
  FreePlanetary (sys);
  FreeForce (force);



 FreeCuda();
 FreeArrays(Dens, Vrad, Vtheta, Energy, Label);

 if (SelfGravity){
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
  cudaFree(powRmed_d);
  cudaFree(vt_cent_d);
  cudaFree(Kr_aux_d);
  cudaFree(Kt_aux_d);

  /* cudaFree InitComputeAccel */
  cudaFree(CellAbscissa_d);
  cudaFree(CellOrdinate_d);

  /* cudaFree polar grid */
  cudaFree(Dens_d);
  cudaFree(Vrad_d);
  cudaFree(Vtheta_d);
  cudaFree(Energy_d);
  cudaFree(Label_d);

  /* cudaFree InitEuler */
  cudaFree(SoundSpeed_d);
  cudaFree(Pressure_d);
  cudaFree(Temperature_d);
  cudaFree(TemperInt_d);
  cudaFree(DensStar_d);
  cudaFree(VradInt_d);
  cudaFree(VthetaInt_d);
  cudaFree(Potential_d);
  cudaFree(DensInt_d);
  cudaFree(VradNew_d);
  cudaFree(VthetaNew_d);
  cudaFree(EnergyInt_d);

  /* cudaFree ReduceCS and ReduceMean*/
  cudaFree(cs0_d);
  cudaFree(cs1_d);
  cudaFree(csnrm1_d);
  cudaFree(csnrm2_d);
  cudaFree(mean_dens_d);
  cudaFree(mean_energy_d);
  cudaFree(mean_dens_d2);
  cudaFree(mean_energy_d2);

  /* cudaFree ComputeForce */
  cudaFree(fxi_d);
  cudaFree(fxo_d);
  cudaFree(fyi_d);
  cudaFree(fyo_d);

  cudaFree(Qplus_d);
  cudaFree(EnergyNew_d);
  cudaFree(EnergyMed_d);
  cudaFree(SigmaMed_d);
  cudaFree(CoolingTimeMed_d);
  cudaFree(QplusMed_d);

  cudaFree(viscosity_array_d);
  cudaFree(SigmaInf_d);
  cudaFree(mdcp0_d);

  cudaFree(QStar_d);
  cudaFree(Qbase_d);
  cudaFree(Qbase2_d);

  /* cudaFree Make1Dprofile */
  cudaFree(gridfield_d);
  cudaFree(GLOBAL_bufarray_d);

  /* cudaFree Crashed */
  cudaFree(array_d);

  /* cudaFree SelfGravity */
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
  cudaFree(axifield_d);
  cudaFree(GLOBAL_AxiSGAccr_d);

  /* cudaFree InitTransport */
  cudaFree(RadMomP_d);
  cudaFree(RadMomM_d);
  cudaFree(ThetaMomP_d);
  cudaFree(ThetaMomM_d);
  cudaFree(Work_d);
  cudaFree(QRStar_d);
  cudaFree(ExtLabel_d);
  cudaFree(dq_d);

  /* cudaFree InitViscosity */
  cudaFree(DivergenceVelocity_d);
  cudaFree(DRP_d);
  cudaFree(DRR_d);
  cudaFree(DPP_d);
  cudaFree(TAURR_d);
  cudaFree(TAURP_d);
  cudaFree(TAUPP_d);

  cudaFree(LostByDisk_d);
  cudaFree(VthetaRes_d);
  cudaFree(VMed_d);
  cudaFree(Nshift_d);
  cudaFree(NoSplitAdvection_d);
  cudaFree(TempShift_d);
  cudaFree(Vmoy_d);
  cudaFree(DT1D_d);
  cudaFree(DT2D_d);
  cudaFree(newDT_d);
  cudaFree(Vresidual_d);

  cudaFree(CFL_d);
}


__host__ void FreeArrays (double *Dens, double *Vrad, double *Vtheta, double *Energy, double *Label)
{
  /* free FillPolar1DArrays */
  free(Radii);
  free(Rinf);
  free(Rmed);
  free(Rsup);
  free(Surf);
  free(invRinf);
  free(invSurf);
  free(invRmed);
  free(invdiffRsup);
  free(invdiffRmed);
  free(powRmed);
  free(vt_cent);
  free(Kr_aux);
  free(Kt_aux);

  /* free InitComputeAccel */
  free(CellAbscissa);
  free(CellOrdinate);

  /* free polar grid */
  free(Dens);
  free(Vrad);
  free(Vtheta);
  free(Energy);
  free(Label);

  /* free ComputeForce */


  /* free InitEuler */
  free(TemperInt);
  free(Temperature);
  free(Pressure);
  free(SoundSpeed);
  free(DensStar);
  free(VradInt);
  free(VthetaInt);
  free(DensInt);
  free(VradNew);
  free(VthetaNew);
  free(EnergyInt);
  free(EnergyNew);
  free(Potential);

  free(VthetaRes);
  free(TempShift);
  free(q0);
  free(q1);
  free(PlanetMasses);

  free(SG_Accr);
  free(SG_Acct);

  free(array);
  free(mdcp0);
  free(EnergyMed);
  free(SigmaMed);
  free(SigmaInf);
  free(vt_int);
  free(GLOBAL_bufarray);
  free(QplusMed);
  free(CoolingTimeMed);
  free(viscosity_array);
  free(Qplus);
  free(QStar);
  free(Qbase);
  free(GLOBAL_AxiSGAccr);

  /* free ReduceCS and ReduceMean*/
  free(cs0);
  free(cs1);
  free(csnrm1);
  free(csnrm2);
  free(mean_dens);
  free(mean_dens2);
  free(mean_energy);
  free(mean_energy2);

  /* free InitTransport */
  free(RadMomP);
  free(RadMomM);
  free(ThetaMomP);
  free(ThetaMomM);
  free(Work);
  free(QRStar);
  free(ExtLabel);
  free(dq);

  /* free InitViscosity */
  free(DivergenceVelocity);
  free(DRP);
  free(DRR);
  free(DPP);
  free(TAURR);
  free(TAURP);
  free(TAUPP);

  free(CFL);

}


__host__ void DeviceToHostcudaMemcpy (double *Dens, double *Energy, double *Label, double *Temperature, double *Vrad, double *Vtheta)
{
  gpuErrchk(cudaMemcpy(Vrad, Vrad_d,               size_grid*sizeof(double), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(Dens, Dens_d,               size_grid*sizeof(double), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(Temperature, Temperature_d, size_grid*sizeof(double), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(Label, Label_d,             size_grid*sizeof(double), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(Vtheta, Vtheta_d,           size_grid*sizeof(double), cudaMemcpyDeviceToHost));
  if (Adiabatic)
    gpuErrchk(cudaMemcpy(Energy, Energy_d,           size_grid*sizeof(double), cudaMemcpyDeviceToHost));
  if (Cooling)
    gpuErrchk(cudaMemcpy(Qplus, Qplus_d,           size_grid*sizeof(double), cudaMemcpyDeviceToHost));
}


__host__ void CreateArrays () // ordenar
{
  CFL   = (int *)malloc(sizeof(int));

  EnergyMed         = (double *)malloc((NRAD+1)*sizeof(double));
  SigmaMed          = (double *)malloc((NRAD+1)*sizeof(double));
  SigmaInf          = (double *)malloc((NRAD+1)*sizeof(double));
  vt_int            = (double *)malloc((NRAD+1)*sizeof(double));
  GLOBAL_bufarray   = (double *)malloc((NRAD+1)*sizeof(double));
  GLOBAL_AxiSGAccr  = (double *)malloc((NRAD+1)*sizeof(double));
  QplusMed          = (double *)malloc((NRAD+1)*sizeof(double));
  CoolingTimeMed    = (double *)malloc((NRAD+1)*sizeof(double));
  viscosity_array = (double *)malloc((NRAD+1)*sizeof(double));

  cs0             = (double *)malloc(NSEC*sizeof(double));
  cs1             = (double *)malloc(NSEC*sizeof(double));
  csnrm1          = (double *)malloc(NSEC*sizeof(double));
  csnrm2          = (double *)malloc(NSEC*sizeof(double));
  mean_dens       = (double *)malloc(NSEC*sizeof(double));
  mean_dens2      = (double *)malloc(NSEC*sizeof(double));
  mean_energy     = (double *)malloc(NSEC*sizeof(double));
  mean_energy2    = (double *)malloc(NSEC*sizeof(double));


  if (SelfGravity){
    SG_Accr         = (double *)malloc(size_grid*sizeof(double));
    SG_Acct         = (double *)malloc(size_grid*sizeof(double));
  }

  example           = (double *)malloc(NRAD*NSEC*sizeof(double));
  QStar           = (double *)malloc(size_grid*sizeof(double));
  Qbase           = (double *)malloc(size_grid*sizeof(double));
  array           = (double *)malloc(size_grid*sizeof(double));
  mdcp0           = (double *)malloc(size_grid*sizeof(double));
  q0              = (double *)malloc(400*sizeof(double));
  q1              = (double *)malloc(400*sizeof(double));
  PlanetMasses    = (double *)malloc(100*sizeof(double));

}


__host__ void Cudamalloc (double *Label, double *Dens, double *Vrad, double *Vtheta) // arreglar
{
  /* cudaMalloc SelfGravity*/

  if (SelfGravity){
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
    gpuErrchk(cudaMalloc((void**)&Kr_aux_d,  2*size_grid*sizeof(double)));
    gpuErrchk(cudaMalloc((void**)&Kt_aux_d,  2*size_grid*sizeof(double)));
    gpuErrchk(cudaMalloc((void**)&SG_Accr_d, size_grid*sizeof(double)));
    gpuErrchk(cudaMalloc((void**)&SG_Acct_d, size_grid*sizeof(double)));
    gpuErrchk(cudaMalloc((void**)&axifield_d, NRAD*sizeof(double)));
  }

  gpuErrchk(cudaMalloc((void**)&Vradial_d,   size_grid*sizeof(double)));
  gpuErrchk(cudaMalloc((void**)&Vazimutal_d, size_grid*sizeof(double)));

  gpuErrchk(cudaMemset(Vradial_d, 0,    size_grid*sizeof(double)));
  gpuErrchk(cudaMemset(Vazimutal_d, 0,  size_grid*sizeof(double)));

  /* cudaMalloc ComputeForce */

  gpuErrchk(cudaMalloc(&fxi_d, NRAD*NSEC*sizeof(double)));
  gpuErrchk(cudaMalloc(&fxo_d, NRAD*NSEC*sizeof(double)));
  gpuErrchk(cudaMalloc(&fyi_d, NRAD*NSEC*sizeof(double)));
  gpuErrchk(cudaMalloc(&fyo_d, NRAD*NSEC*sizeof(double)));

  gpuErrchk(cudaMemset(fxi_d, 0, NRAD*NSEC*sizeof(double)));
  gpuErrchk(cudaMemset(fxo_d, 0, NRAD*NSEC*sizeof(double)));
  gpuErrchk(cudaMemset(fyi_d, 0, NRAD*NSEC*sizeof(double)));
  gpuErrchk(cudaMemset(fyo_d, 0, NRAD*NSEC*sizeof(double)));

  /* cudaMalloc FillPolar1DArrays */
  gpuErrchk(cudaMalloc((void**)&Radii_d,           (NRAD+1)*sizeof(double)));
  gpuErrchk(cudaMalloc((void**)&Rmed_d,            (NRAD+1)*sizeof(double)));
  gpuErrchk(cudaMalloc((void**)&Rinf_d,            (NRAD+1)*sizeof(double)));
  gpuErrchk(cudaMalloc((void**)&invRmed_d,         (NRAD+1)*sizeof(double)));
  gpuErrchk(cudaMalloc((void**)&Rsup_d,            (NRAD+1)*sizeof(double)));
  gpuErrchk(cudaMalloc((void**)&invdiffRmed_d,     (NRAD+1)*sizeof(double)));
  gpuErrchk(cudaMalloc((void**)&invRinf_d,         (NRAD+1)*sizeof(double)));
  gpuErrchk(cudaMalloc((void**)&powRmed_d,         (NRAD+1)*sizeof(double)));
  gpuErrchk(cudaMalloc((void**)&invdiffRsup_d,     (NRAD+1)*sizeof(double)));
  gpuErrchk(cudaMalloc((void**)&Surf_d,            (NRAD+1)*sizeof(double)));
  gpuErrchk(cudaMalloc((void**)&invSurf_d,         (NRAD+1)*sizeof(double)));

  gpuErrchk(cudaMemset(Radii_d, 0,        (NRAD+1)*sizeof(double)));
  gpuErrchk(cudaMemset(Rmed_d, 0,         (NRAD+1)*sizeof(double)));
  gpuErrchk(cudaMemset(Rinf_d, 0,         (NRAD+1)*sizeof(double)));
  gpuErrchk(cudaMemset(invRmed_d, 0,      (NRAD+1)*sizeof(double)));
  gpuErrchk(cudaMemset(Rsup_d, 0,         (NRAD+1)*sizeof(double)));
  gpuErrchk(cudaMemset(invdiffRmed_d, 0,  (NRAD+1)*sizeof(double)));
  gpuErrchk(cudaMemset(invRinf_d, 0,      (NRAD+1)*sizeof(double)));
  gpuErrchk(cudaMemset(powRmed_d, 0,      (NRAD+1)*sizeof(double)));
  gpuErrchk(cudaMemset(invdiffRsup_d, 0,  (NRAD+1)*sizeof(double)));
  gpuErrchk(cudaMemset(Surf_d, 0,         (NRAD+1)*sizeof(double)));
  gpuErrchk(cudaMemset(invSurf_d, 0,      (NRAD+1)*sizeof(double)));


  /* cudaMalloc med */
  gpuErrchk(cudaMalloc((void**)&EnergyMed_d,       (NRAD+1)*sizeof(double)));
  gpuErrchk(cudaMalloc((void**)&SigmaMed_d,        (NRAD+1)*sizeof(double)));
  gpuErrchk(cudaMalloc((void**)&CoolingTimeMed_d,  (NRAD+1)*sizeof(double)));
  gpuErrchk(cudaMalloc((void**)&QplusMed_d,        (NRAD+1)*sizeof(double)));

  gpuErrchk(cudaMemset(EnergyMed_d, 0,      (NRAD+1)*sizeof(double)));
  gpuErrchk(cudaMemset(SigmaMed_d, 0,       (NRAD+1)*sizeof(double)));
  gpuErrchk(cudaMemset(CoolingTimeMed_d, 0, (NRAD+1)*sizeof(double)));
  gpuErrchk(cudaMemset(QplusMed_d, 0,       (NRAD+1)*sizeof(double)));


  /* cudaMalloc ReduceCs */
  gpuErrchk(cudaMalloc((void**)&cs0_d,          NSEC*sizeof(double)));
  gpuErrchk(cudaMalloc((void**)&cs1_d,          NSEC*sizeof(double)));
  gpuErrchk(cudaMalloc((void**)&csnrm1_d,       NSEC*sizeof(double)));
  gpuErrchk(cudaMalloc((void**)&csnrm2_d,       NSEC*sizeof(double)));

  gpuErrchk(cudaMemset(cs0_d, 0,    NSEC*sizeof(double)));
  gpuErrchk(cudaMemset(cs1_d, 0,    NSEC*sizeof(double)));
  gpuErrchk(cudaMemset(csnrm1_d, 0, NSEC*sizeof(double)));
  gpuErrchk(cudaMemset(csnrm2_d, 0, NSEC*sizeof(double)));

  /* cudaMalloc ReduceMean */
  gpuErrchk(cudaMalloc((void**)&mean_dens_d,    NSEC*sizeof(double)));
  gpuErrchk(cudaMalloc((void**)&mean_dens_d2,   NSEC*sizeof(double)));
  gpuErrchk(cudaMalloc((void**)&mean_energy_d,  NSEC*sizeof(double)));
  gpuErrchk(cudaMalloc((void**)&mean_energy_d2, NSEC*sizeof(double)));

  gpuErrchk(cudaMemset(mean_dens_d, 0,    NSEC*sizeof(double)));
  gpuErrchk(cudaMemset(mean_dens_d2, 0,   NSEC*sizeof(double)));
  gpuErrchk(cudaMemset(mean_energy_d, 0,  NSEC*sizeof(double)));
  gpuErrchk(cudaMemset(mean_energy_d2, 0, NSEC*sizeof(double)));

  gpuErrchk(cudaMalloc((void**)&Qplus_d,            size_grid*sizeof(double)));
  gpuErrchk(cudaMalloc((void**)&EnergyNew_d,        size_grid*sizeof(double)));
  gpuErrchk(cudaMalloc((void**)&GLOBAL_bufarray_d,  (NRAD+1)*sizeof(double)));

  gpuErrchk(cudaMemset(Qplus_d, 0,            size_grid*sizeof(double)));
  gpuErrchk(cudaMemset(EnergyNew_d, 0,        size_grid*sizeof(double)));
  gpuErrchk(cudaMemset(GLOBAL_bufarray_d, 0,  (NRAD+1)*sizeof(double)));


  /* cudaMalloc polar grid */
  gpuErrchk(cudaMalloc((void**)&Vrad_d,           size_grid*sizeof(double)));
  gpuErrchk(cudaMalloc((void**)&Vtheta_d,         size_grid*sizeof(double)));
  gpuErrchk(cudaMalloc((void**)&Dens_d,           size_grid*sizeof(double)));
  gpuErrchk(cudaMalloc((void**)&Label_d,          size_grid*sizeof(double)));

  gpuErrchk(cudaMemset(Vrad_d, 0,   size_grid*sizeof(double)));
  gpuErrchk(cudaMemset(Vtheta_d, 0, size_grid*sizeof(double)));
  gpuErrchk(cudaMemset(Dens_d, 0,   size_grid*sizeof(double)));
  gpuErrchk(cudaMemset(Label_d, 0,  size_grid*sizeof(double)));


  gpuErrchk(cudaMalloc((void**)&Vresidual_d,      size_grid*sizeof(double)));
  gpuErrchk(cudaMalloc((void**)&newDT_d,          NRAD*sizeof(double)));
  gpuErrchk(cudaMalloc((void**)&DT1D_d,           NRAD*sizeof(double)));
  gpuErrchk(cudaMalloc((void**)&gridfield_d,      size_grid*sizeof(double)));
  gpuErrchk(cudaMalloc((void**)&Qbase_d,          size_grid*sizeof(double)));
  gpuErrchk(cudaMalloc((void**)&Qbase2_d,         size_grid*sizeof(double)));
  gpuErrchk(cudaMalloc((void**)&QStar_d,          size_grid*sizeof(double)));
  gpuErrchk(cudaMalloc((void**)&array_d,          size_grid*sizeof(double)));
  gpuErrchk(cudaMalloc((void**)&mdcp0_d,          NRAD*NSEC*sizeof(double)));
  gpuErrchk(cudaMalloc((void**)&Vmoy_d,           size_grid*sizeof(double)));
  gpuErrchk(cudaMalloc((void**)&DT2D_d,           NRAD*NSEC*sizeof(double)));
  gpuErrchk(cudaMalloc((void**)&CFL_d,            sizeof(int)));

  gpuErrchk(cudaMemset(Vresidual_d, 0,    size_grid*sizeof(double)));
  gpuErrchk(cudaMemset(newDT_d, 0,        NRAD*sizeof(double)));
  gpuErrchk(cudaMemset(DT1D_d, 0,         NRAD*sizeof(double)));
  gpuErrchk(cudaMemset(gridfield_d, 0,    size_grid*sizeof(double)));
  gpuErrchk(cudaMemset(Qbase_d, 0,        size_grid*sizeof(double)));
  gpuErrchk(cudaMemset(Qbase2_d, 0,       size_grid*sizeof(double)));
  gpuErrchk(cudaMemset(QStar_d, 0,        size_grid*sizeof(double)));
  gpuErrchk(cudaMemset(array_d, 0,        size_grid*sizeof(double)));
  gpuErrchk(cudaMemset(mdcp0_d, 0,        NRAD*NSEC*sizeof(double)));
  gpuErrchk(cudaMemset(Vmoy_d, 0,         size_grid*sizeof(double)));
  gpuErrchk(cudaMemset(DT2D_d, 0,         NRAD*NSEC*sizeof(double)));
  gpuErrchk(cudaMemset(CFL_d, 0,          sizeof(int)));

  /* cudaMemcpy Host to Device */
  gpuErrchk(cudaMemcpy(Radii_d, Radii,             (NRAD+1)*sizeof(double), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(Rmed_d, Rmed,               (NRAD+1)*sizeof(double), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(Rinf_d, Rinf,               (NRAD+1)*sizeof(double), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(invRmed_d, invRmed,         (NRAD+1)*sizeof(double), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(Rsup_d, Rsup,               (NRAD+1)*sizeof(double), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(invdiffRmed_d, invdiffRmed, (NRAD+1)*sizeof(double), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(invRinf_d, invRinf,         (NRAD+1)*sizeof(double), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(powRmed_d, powRmed,         (NRAD+1)*sizeof(double), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(invdiffRsup_d, invdiffRsup, (NRAD+1)*sizeof(double), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(Surf_d, Surf,               (NRAD+1)*sizeof(double), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(invSurf_d, invSurf,         (NRAD+1)*sizeof(double), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(Dens_d, Dens,               size_grid*sizeof(double), cudaMemcpyHostToDevice));

}

__host__ void binFile(double *array, int sizeArray, char *name)
{
  FILE *f;
  char filename[100];
  sprintf(filename, "../output/%s.raw",name);
  f = fopen(filename, "w");
  fwrite(array, sizeof(double), sizeArray, f);
  fclose(f);
}
