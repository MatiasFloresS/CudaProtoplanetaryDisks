#include "Main.cuh"

#define CFLSECURITY 0.5 /* Maximum fraction of zone size */
              /* swept in one timestep */

#define CVNR 1.41 /* Shocks are spread over CVNR zones:   */
              /* Von Neumann-Richtmyer viscosity constant */
              /* Beware of misprint in Stone and Norman's */
              /* paper : use C2^2 instead of C2           */

extern int NRAD, NSEC, LogGrid, size_grid, SelfGravity, ViscosityAlpha;
extern int Adiabatic, Cooling, Corotating, IsDisk, Evanescent, FastTransport;
extern int CentrifugalBalance, ZMPlus = NO, SloppyCFL, *CFL_d, *CFL;

extern string OUTPUTDIR;

extern double RMAX, RMIN, ADIABATICINDEX, FLARINGINDEX, ASPECTRATIO;
extern double SIGMA0, SIGMASLOPE, IMPOSEDDISKDRIFT, PhysicalTime, DT, MASSTAPER;
extern double SGP_eps, TRANSITIONWIDTH, TRANSITIONRATIO, TRANSITIONRADIUS;
extern double LAMBDADOUBLING, PhysicalTimeInitial, mdcp;

extern double *GLOBAL_bufarray, *vt_int, *SigmaInf, *CoolingTimeMed, *QplusMed , *viscosity_array;
extern double *SG_Accr, *array, *Qplus, *SigmaMed, *QplusMed, *EnergyMed, *CellOrdinate, *CellAbscissa;

extern double *Dens_d, *SG_Accr_d, *SG_Acct_d, *GLOBAL_bufarray_d, *array_d;
extern double *Qplus_d, *EnergyInt_d, *EnergyNew_d, *VradNew_d, *Potential_d;
extern double *VthetaInt_d, *powRmed_d, *VthetaNew_d, *SigmaMed_d, *QplusMed_d;
extern double *CoolingTimeMed_d, *EnergyMed_d, *DensInt_d, *DivergenceVelocity_d, *TAURR_d, *TAURP_d;
extern double *TAUPP_d, *Vmoy_d, *CellOrdinate_d, *CellAbscissa_d, *mdcp0_d, *Surf_d;

extern double *invdiffRmed_d, *invRinf_d, *Rmed_d, *invRmed_d, *invdiffRsup_d, *Rsup_d;

double *invdiffRmed, *invRinf, *Rinf, *Rinf_d, *invRmed, *Rmed, *invdiffRsup, *Rsup;

double *Pressure, *SoundSpeed, *Temperature, *vt_cent, *DensStar;
double *invSurf, *Radii, *Surf, *powRmed, *VradInt;
double *VthetaInt, *DensInt, *VradNew, *VthetaNew, *EnergyInt, *Potential, *EnergyNew, *TemperInt;

double *Vtheta_d, *SigmaInf_d, *Vrad_d, *SoundSpeed_d, *Energy_d, *Pressure_d;
double *Temperature_d, *viscosity_array_d, *vt_cent_d, *DensStar_d, *TemperInt_d, *VradInt_d, *DT1D_d;
double *DT2D_d, *newDT_d, *Vresidual_d, *Vradial_d, *Vazimutal_d;


double *DT2D;
double exces_mdcp = 0.0, mdcp1, MassTaper;

int CrashedDens, CrashedEnergy;

extern dim3 dimGrid2, dimBlock2, dimGrid4, dimBlock;

int init = 0;

extern double OmegaFrame, HillRadius;

Pair DiskOnPrimaryAcceleration;



__host__ void FillPolar1DArrays ()
{
  FILE *input, *output;
  int i,j;
  double drrsep, temporary;
  double *Radii2, *Rmed2;
  string InputName, OutputName;
  drrsep = (RMAX-RMIN)/(double)NRAD;
  InputName = OUTPUTDIR + "radii.dat";
  OutputName = OUTPUTDIR + "used_rad.dat";

  /* Creo los arreglos de FillPolar1DArrays */
  Radii       = (double *)malloc((NRAD+1)*sizeof(double));
  Radii2       = (double *)malloc((NRAD+1)*sizeof(double));
  Rmed2       = (double *)malloc(NRAD*sizeof(double));
  Rinf        = (double *)malloc(NRAD*sizeof(double));
  Rmed        = (double *)malloc(NRAD*sizeof(double));
  Rsup        = (double *)malloc(NRAD*sizeof(double));
  Surf        = (double *)malloc(NRAD*sizeof(double));
  invRinf     = (double *)malloc(NRAD*sizeof(double));
  invSurf     = (double *)malloc(NRAD*sizeof(double));
  invRmed     = (double *)malloc(NRAD*sizeof(double));
  invdiffRsup = (double *)malloc(NRAD*sizeof(double));
  invdiffRmed = (double *)malloc(NRAD*sizeof(double));
  vt_cent     = (double *)malloc(NRAD*sizeof(double));
  powRmed     = (double *)malloc(NRAD*sizeof(double));
  DT2D     = (double *)malloc(size_grid*sizeof(double));

  char InputCharName[100];
  char OutputCharName[100];
  /* string to char InputName */
  strncpy(InputCharName, InputName.c_str(), sizeof(InputCharName));
  InputCharName[sizeof(InputCharName)-1]=0;

  input = fopen (InputCharName, "r");
  if (input == NULL){
    printf("Warning : no `radii.dat' file found. Using default.\n");
    if (LogGrid == YES){
      for (i = 0; i <= NRAD; i++){
        /* Usamos doubles para calcular los valores de los arrays, luego
           los pasamos a double */
        Radii2[i] = RMIN*exp((double)i/(double)NRAD*log((double)RMAX/(double)RMIN));
        Radii[i] = Radii2[i];
      }
    }
    else {
      for (i = 0; i <= NRAD; i++)
        Radii[i] = RMIN+drrsep*i;
    }
  }
  else {
    printf("Reading 'radii.dat' file.\n");
    for (i = 0; i <= NRAD; i++){
      fscanf (input, "%f", &temporary);
      Radii[i] = (double)temporary;
    }
  }

  double numero=0;
  for (i = 0; i < NRAD; i++){
    Rinf[i] = Radii2[i];
    Rsup[i] = Radii2[i+1];
    Rmed2[i] = 2.0/3.0*(Radii2[i+1]*Radii2[i+1]*Radii2[i+1]-Radii2[i]*Radii2[i]*Radii2[i]);
    Rmed2[i] = Rmed2[i] / (Radii2[i+1]*Radii2[i+1]-Radii2[i]*Radii2[i]);
    Rmed[i] = Rmed2[i];
    //printf("Rmed %.10f\n", Rmed[i]);
    //printf("Rmed2 %.10f\n", Rmed2[i]);
    Surf[i] = M_PI*(Rsup[i]*Rsup[i]-Rinf[i]*Rinf[i])/(double)NSEC;
    invRmed[i] = 1.0/Rmed[i];
    invSurf[i] = 1.0/Surf[i];
    printf("invSurf%.15g\n", invSurf[i]);
    invdiffRsup[i] = 1.0/(Rsup[i]-Rinf[i]);
    invRinf[i] = 1.0/Rinf[i];
  }
  //printf("sumaSurf = %g\n", numero);

  Rinf[NRAD] = Radii2[NRAD];

  for (i = 1; i < NRAD; i++) {
    invdiffRmed[i] = (double)1.0/(double)(Rmed2[i]-Rmed2[i-1]);
    powRmed[i] = pow(Rmed[i],-2.5+SIGMASLOPE);
  }

  /* string to char OutputName */
  strncpy(OutputCharName, OutputName.c_str(), sizeof(OutputCharName));
  OutputCharName[sizeof(OutputCharName)-1]=0;

  output = fopen (OutputCharName, "w");
  if (output == NULL){
    printf ("Can't write %s.\nProgram stopped.\n", OutputCharName);
    exit (1);
  }
  for (i = 0; i <= NRAD; i++){
    fprintf (output, "%.10f\n", Radii[i]);
  }
  fclose (output);
  if (input != NULL) fclose (input);
}



__host__ void InitEuler (double *Vrad, double *Vtheta, double *Dens, double *Energy)
{
  InitTransport ();
  InitViscosity ();
  DensStar        = (double *)malloc(size_grid*sizeof(double));
  DensInt         = (double *)malloc(size_grid*sizeof(double));
  VradNew         = (double *)malloc(size_grid*sizeof(double));
  VradInt         = (double *)malloc(size_grid*sizeof(double));
  VthetaNew       = (double *)malloc(size_grid*sizeof(double));
  VthetaInt       = (double *)malloc(size_grid*sizeof(double));
  EnergyNew       = (double *)malloc(size_grid*sizeof(double));
  EnergyInt       = (double *)malloc(size_grid*sizeof(double));
  TemperInt       = (double *)malloc(size_grid*sizeof(double));
  Potential       = (double *)malloc(size_grid*sizeof(double));
  Pressure        = (double *)malloc(size_grid*sizeof(double));
  SoundSpeed      = (double *)malloc(size_grid*sizeof(double));
  Temperature     = (double *)malloc(size_grid*sizeof(double));
  Qplus           = (double *)malloc(size_grid*sizeof(double));

  Computecudamalloc (Energy);

  InitComputeAccel ();
  /* Rho and Energy are already initialized: cf main.cu*/
  ComputeSoundSpeed ();
  ComputePressureField ();
  ComputeTemperatureField ();
  InitGasVelocities (Vrad, Vtheta);

}



__host__ void AlgoGas (Force *force, double *Dens, double *Vrad, double *Vtheta, double *Energy, double *Label,
  PlanetarySystem *sys, int initialization)
{
  double dt, dtemp =0.0;
  double OmegaNew, domega;
  int gastimestepcfl = 1;
  CrashedDens = 0;
  CrashedEnergy = 0;

  if (Adiabatic){
    ComputeSoundSpeed();
    /* it is necesary to update computation of soundspeed if one uses
      alphaviscosity in Fviscosity. It is not necesary in locally
      isothermal runs since cs is constant. It is computed here for
      the needs of ConditionCFL. */
  }
  if (IsDisk == YES){
    if (SloppyCFL == YES){
      gastimestepcfl = ConditionCFL(Vrad, Vtheta, DT-dtemp);     // case ./bin/fargoGPU -c
    }
  }

  dt = DT / gastimestepcfl;
  //printf("dt %.10f\n", dt);

  while (dtemp < 0.9999999999*DT){
    MassTaper = PhysicalTime/(MASSTAPER*2.0*M_PI);
    MassTaper = (MassTaper > 1.0 ? 1.0 : pow(sin(MassTaper*M_PI/2.0), 2.0));
    //printf("dtemp = %.10f\n",dtemp );
    //printf("MassTaper = %g\n",MassTaper );

    if(IsDisk == YES){
      if (SloppyCFL == NO){
        //printf("DT-dtemp = %.10f\n",DT-dtemp );
        gastimestepcfl = 1;
        gastimestepcfl = ConditionCFL(Vrad, Vtheta ,DT-dtemp);
        dt = (DT-dtemp)/(double)gastimestepcfl;
        //printf("gastimestepcfl %d\n", gastimestepcfl);
        //printf("dt %.15f\n", dt);
      }
      AccreteOntoPlanets(Dens, Vrad, Vtheta, dt, sys); // si existe acrecion entra
    }
    //printf("dt %g\n",dt );
    dtemp += dt;
    //printf("dtemp %.10f\n", dtemp);
    DiskOnPrimaryAcceleration.x = 0.0;
    DiskOnPrimaryAcceleration.y = 0.0;
    if (Corotating == YES) GetPsysInfo (sys, MARK);


    if (IsDisk == YES){
      /* Indirect term star's potential computed here */
      DiskOnPrimaryAcceleration = ComputeAccel (force, Dens, 0.0, 0.0, 0.0, 0.0);

      /* Gravitational potential from star and planet(s) is computed and stored here */
      FillForcesArrays (sys, Dens, Energy);

      /* Planet's velocities are update here from gravitational interaction with disk */
      AdvanceSystemFromDisk (force, Dens, Energy, sys, dt);
    }

    /* Planet's positions and velocities are update from gravitational interaction with star
       and other planets */
    AdvanceSystemRK5 (sys,dt);

    /* Below we correct vtheta, planet's position and velocities if we work in a frame non-centered on the star */
    if (Corotating == YES){
      OmegaNew = GetPsysInfo(sys, GET) / dt;
      //printf("omeganew %.10f\n", OmegaNew );
      domega = OmegaNew - OmegaFrame;
      //printf("OmegaFrame  %.10f\n", OmegaFrame);
      //printf("domega  %.15f\n", domega);
      if (IsDisk == YES) CorrectVtheta (Vtheta, domega);
      OmegaFrame = OmegaNew;
    }
    //printf("OmegaFrame %.10f\n", OmegaFrame);
    RotatePsys (sys, OmegaFrame*dt);

    /* Now we update gas */
    if (IsDisk == YES){
      //ApplyBoundaryCondition (Dens, Energy, Vrad, Vtheta, dt);
      gpuErrchk(cudaMemcpy(Dens, Dens_d,     size_grid*sizeof(double), cudaMemcpyDeviceToHost));
      gpuErrchk(cudaMemcpy(Energy, Energy_d, size_grid*sizeof(double), cudaMemcpyDeviceToHost));
      CrashedDens = DetectCrash (Dens);
      CrashedEnergy = DetectCrash (Energy);
      if (CrashedDens == YES || CrashedEnergy == YES){
        fprintf(stdout, "\nCrash! at time %d\n", PhysicalTime);
        printf("c");
      }
      else
        printf(".");
      // if (ZMPlus) compute_anisotropic_pressurecoeff(sys);

      ComputePressureField ();
      Substep1 (Dens, Vrad, Vtheta, dt, init);
      Substep2 (dt);
      ActualiseGasVrad (Vrad, VradNew);
      ActualiseGasVtheta (Vtheta, VthetaNew);

      //gpuErrchk(cudaMemcpy(Vrad_d, VradInt_d, size_grid*sizeof(double), cudaMemcpyDeviceToDevice));
      //gpuErrchk(cudaMemcpy(Vtheta_d, VthetaInt_d, size_grid*sizeof(double), cudaMemcpyDeviceToDevice));


      /*FILE *f;
      f = fopen("vradnew.txt","w");
      for (int i = 0; i < NRAD*NSEC; i++) {
        fprintf(f, "%.15f\n", Vrad[i]);
      }
      fclose(f);

      f = fopen("vthetanew.txt","w");
      for (int i = 0; i < NRAD*NSEC; i++) {
        fprintf(f, "%.15f\n", Vtheta[i]);
      }
      fclose(f);

      exit(1);*/


      //ApplyBoundaryCondition (Dens, Energy, Vrad, Vtheta, dt);

      if (Adiabatic){
        gpuErrchk(cudaMemcpy(Vradial_d, Vrad_d, size_grid*sizeof(double), cudaMemcpyDeviceToDevice));
        gpuErrchk(cudaMemcpy(Vazimutal_d, Vtheta_d, size_grid*sizeof(double), cudaMemcpyDeviceToDevice));
        ComputeViscousTerms (Vrad, Vtheta, Dens);
        gpuErrchk(cudaMemcpy(Vrad_d, Vradial_d, size_grid*sizeof(double), cudaMemcpyDeviceToDevice));
        gpuErrchk(cudaMemcpy(Vtheta_d, Vazimutal_d, size_grid*sizeof(double), cudaMemcpyDeviceToDevice));

        Substep3 (Dens, dt);

        ActualiseGasEnergy (Energy, EnergyNew);
      }
      Transport (Dens, Vrad, Vtheta, Energy, Label, dt);
      //ApplyBoundaryCondition(Dens, Energy, Vrad, Vtheta, dt);
      ComputeTemperatureField ();

      mdcp1 = CircumPlanetaryMass (Dens, sys);
      exces_mdcp = mdcp1 - mdcp;

      //printf("%g - %g \n",mdcp1, mdcp );
      //printf("exces_mdcp %g \n",exces_mdcp);
    }
    init = init + 1;

    PhysicalTime += dt;
  }
  printf("\n" );
}



__host__ void Substep1 (double *Dens, double *Vrad, double *Vtheta, double dt, int initialization)
{
  int selfgravityupdate;
  if(initialization == 0) Substep1cudamalloc(Vrad, Vtheta);

  Substep1Kernel<<<dimGrid2, dimBlock2>>>(Pressure_d, Dens_d, VradInt_d, invdiffRmed_d, Potential_d, Rinf_d,
    invRinf_d, Vrad_d, VthetaInt_d, Vtheta_d, Rmed_d,  dt, NRAD, NSEC, OmegaFrame, ZMPlus,
    IMPOSEDDISKDRIFT, SIGMASLOPE, powRmed_d);
  gpuErrchk(cudaDeviceSynchronize());
  //exit(1);

  /*gpuErrchk(cudaMemcpy(VradInt, VradInt_d, size_grid*sizeof(double), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(VthetaInt, VthetaInt_d, size_grid*sizeof(double), cudaMemcpyDeviceToHost));


  FILE *f;
  f = fopen("Vradint.txt","w");
  for (int i = 0; i < (NRAD+1)*NSEC; i++) {
    fprintf(f, "%.15f\n", VradInt[i]);
  }

  fclose(f);

  f = fopen("Vthetaint.txt","w");
  for (int i = 0; i < (NRAD+1)*NSEC; i++) {
    fprintf(f, "%.15f\n", VthetaInt[i]);
  }
  fclose(f);
  exit(1);*/

  if (SelfGravity){
    selfgravityupdate = YES;

    /* We copy VradInt to Vradial -> device to device */
    gpuErrchk(cudaMemcpy(Vradial_d, VradInt_d, size_grid*sizeof(double), cudaMemcpyDeviceToDevice));
    gpuErrchk(cudaMemcpy(Vazimutal_d, VthetaInt_d, size_grid*sizeof(double), cudaMemcpyDeviceToDevice));

    compute_selfgravity(Dens, dt, selfgravityupdate, 0);
    /* Vradialto VradInt -> device to device */
    gpuErrchk(cudaMemcpy(VradInt_d, Vradial_d, size_grid*sizeof(double), cudaMemcpyDeviceToDevice));
    gpuErrchk(cudaMemcpy(VthetaInt_d, Vazimutal_d, size_grid*sizeof(double), cudaMemcpyDeviceToDevice));
  }

  gpuErrchk(cudaMemcpy(Vradial_d, VradInt_d, size_grid*sizeof(double), cudaMemcpyDeviceToDevice));
  gpuErrchk(cudaMemcpy(Vazimutal_d, VthetaInt_d, size_grid*sizeof(double), cudaMemcpyDeviceToDevice));
  ComputeViscousTerms (VradInt, VthetaInt, Dens);


  UpdateVelocitiesWithViscosity(VradInt, VthetaInt, Dens, dt);


  gpuErrchk(cudaMemcpy(VradInt, VradInt_d, size_grid*sizeof(double), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(VthetaInt, VthetaInt_d, size_grid*sizeof(double), cudaMemcpyDeviceToHost));


  if (!Evanescent)
    ApplySubKeplerianBoundary(VthetaInt);

/*
      FILE *f;
      f = fopen("Vradint.txt","w");
      for (int i = 0; i < (NRAD+1)*NSEC; i++) {
        fprintf(f, "%.15f\n", VradInt[i]);
      }

      fclose(f);

      f = fopen("Vthetaint.txt","w");
      for (int i = 0; i < (NRAD+1)*NSEC; i++) {
        fprintf(f, "%.15f\n", VthetaInt[i]);
      }
      fclose(f);
      exit(1);*/
}



__host__ void Substep2 (double dt)
{
  Substep2Kernel<<<dimGrid2, dimBlock2>>>(Dens_d, VradInt_d, VthetaInt_d, TemperInt_d, NRAD, NSEC, CVNR, invdiffRmed_d,
  invdiffRsup_d, DensInt_d, Adiabatic, Rmed_d, dt, VradNew_d, VthetaNew_d, Energy_d, EnergyInt_d);
  gpuErrchk(cudaDeviceSynchronize());
}



__host__ void Substep3 (double *Dens, double dt)
{
  for (int i = 0; i < NRAD; i++) viscosity_array[i] = FViscosity(Rmed[i]);
  gpuErrchk(cudaMemcpy(viscosity_array_d, viscosity_array, (NRAD+1)*sizeof(double), cudaMemcpyHostToDevice));

  Substep3Kernel<<<dimGrid2, dimBlock2>>>(Dens_d, Qplus_d, viscosity_array_d, TAURR_d, TAURP_d , TAUPP_d, DivergenceVelocity_d,
     NRAD, NSEC, Rmed_d, Cooling, EnergyNew_d, dt, EnergyMed_d, SigmaMed_d, CoolingTimeMed_d, Energy_d,
     ADIABATICINDEX, QplusMed_d);
  gpuErrchk(cudaDeviceSynchronize());
}


__host__ void Computecudamalloc (double *Energy)
{

  CoolingTimeMed  = (double *)malloc(size_grid*sizeof(double));
  QplusMed        = (double *)malloc(size_grid*sizeof(double));
  viscosity_array = (double *)malloc((NRAD+1)*sizeof(double));

  gpuErrchk(cudaMalloc((void**)&TemperInt_d,   size_grid*sizeof(double)));
  gpuErrchk(cudaMalloc((void**)&Temperature_d, size_grid*sizeof(double)));
  gpuErrchk(cudaMalloc((void**)&Pressure_d,    size_grid*sizeof(double)));
  gpuErrchk(cudaMalloc((void**)&SoundSpeed_d,  size_grid*sizeof(double)));
  gpuErrchk(cudaMalloc((void**)&DensStar_d,    size_grid*sizeof(double)));
  gpuErrchk(cudaMalloc((void**)&VradInt_d,     size_grid*sizeof(double)));
  gpuErrchk(cudaMalloc((void**)&DensInt_d,     size_grid*sizeof(double)));
  gpuErrchk(cudaMalloc((void**)&VradNew_d,     size_grid*sizeof(double)));
  gpuErrchk(cudaMalloc((void**)&VthetaNew_d,   size_grid*sizeof(double)));
  gpuErrchk(cudaMalloc((void**)&Potential_d,   size_grid*sizeof(double)));
  gpuErrchk(cudaMalloc((void**)&VthetaInt_d,   size_grid*sizeof(double)));

  gpuErrchk(cudaMemset(TemperInt_d, 0, size_grid*sizeof(double)));
  gpuErrchk(cudaMemset(Temperature_d, 0, size_grid*sizeof(double)));
  gpuErrchk(cudaMemset(Pressure_d, 0, size_grid*sizeof(double)));
  gpuErrchk(cudaMemset(SoundSpeed_d, 0, size_grid*sizeof(double)));
  gpuErrchk(cudaMemset(DensStar_d, 0, size_grid*sizeof(double)));
  gpuErrchk(cudaMemset(VradInt_d, 0, size_grid*sizeof(double)));
  gpuErrchk(cudaMemset(DensInt_d, 0, size_grid*sizeof(double)));
  gpuErrchk(cudaMemset(VradNew_d, 0, size_grid*sizeof(double)));
  gpuErrchk(cudaMemset(VthetaNew_d, 0, size_grid*sizeof(double)));
  gpuErrchk(cudaMemset(Potential_d, 0, size_grid*sizeof(double)));
  gpuErrchk(cudaMemset(VthetaInt_d, 0, size_grid*sizeof(double)));


  gpuErrchk(cudaMalloc((void**)&SigmaInf_d,        NRAD*sizeof(double)));
  gpuErrchk(cudaMalloc((void**)&vt_cent_d,         (NRAD+1)*sizeof(double)));
  gpuErrchk(cudaMalloc((void**)&viscosity_array_d, (NRAD+1)*sizeof(double)));

  gpuErrchk(cudaMemcpy(SigmaInf_d, SigmaInf,               NRAD*sizeof(double), cudaMemcpyHostToDevice));

  gpuErrchk(cudaMalloc((void**)&Energy_d,   size_grid*sizeof(double)));
  gpuErrchk(cudaMalloc((void**)&EnergyInt_d,   size_grid*sizeof(double)));
  gpuErrchk(cudaMemset(Energy_d, 0, size_grid*sizeof(double)));

  if (Adiabatic)
  gpuErrchk(cudaMemcpy(Energy_d, Energy,  size_grid*sizeof(double), cudaMemcpyHostToDevice));

}


__host__ double ConstructSequence (double *u, double *v, int n)
{
  int i;
  double lapl = 0.0;

  for (i = 1; i < n; i++) u[i] = 2.0*v[i]-u[i-1];
  for (i = 1; i < n-1; i++) lapl += fabs(u[i+1]+u[i-1]-2.0*u[i]);

  return lapl;
}


__host__ void Init_azimutalvelocity_withSG (double *Vtheta)
{
  // !SGZeroMode
  gpuErrchk(cudaMemcpy(SG_Accr, SG_Accr_d, size_grid*sizeof(double), cudaMemcpyDeviceToHost));
  Make1Dprofile(1);

  Azimutalvelocity_withSGKernel<<<dimGrid2, dimBlock2>>>(Vtheta_d, Rmed_d, FLARINGINDEX, SIGMASLOPE, ASPECTRATIO,
    GLOBAL_bufarray_d, NRAD, NSEC);
  gpuErrchk(cudaDeviceSynchronize());
}


__host__ int DetectCrash (double *array)
{
  int Crash = NO;
  double numCrush;

  gpuErrchk(cudaMemcpy(array_d, array, size_grid*sizeof(double), cudaMemcpyHostToDevice));
  CrashKernel<<<dimGrid2, dimBlock2>>>(array_d, NRAD, NSEC, Crash);
  gpuErrchk(cudaDeviceSynchronize());

  numCrush = DeviceReduce(array_d, size_grid);
  if (numCrush > 0.0) Crash = true;
  return Crash;
}

__host__ void ComputePressureField ()
{
  ComputePressureFieldKernel<<<dimGrid2, dimBlock2>>>(SoundSpeed_d, Dens_d, Pressure_d, Adiabatic, NRAD,
    NSEC, ADIABATICINDEX, Energy_d);
  gpuErrchk(cudaDeviceSynchronize());

  // gpuErrchk(cudaMemcpy(Pressure, Pressure_d, size_grid*sizeof(double), cudaMemcpyDeviceToHost));
  //
  // FILE *f;
  // f = fopen("press.txt", "w");
  // for (int i = 0; i < size_grid; i++) {
  //   fprintf(f, "%.10f\n", Pressure[i]);
  // }
  // fclose(f);

}


__host__ void ComputeSoundSpeed ()
{
  ComputeSoundSpeedKernel<<<dimGrid2, dimBlock2>>>(SoundSpeed_d, Dens_d, Rmed_d, Energy_d, NSEC, NRAD,
    Adiabatic, ADIABATICINDEX, FLARINGINDEX, ASPECTRATIO, TRANSITIONWIDTH, TRANSITIONRADIUS,
    TRANSITIONRATIO, PhysicalTime, PhysicalTimeInitial, LAMBDADOUBLING);
  gpuErrchk(cudaDeviceSynchronize());

  // gpuErrchk(cudaMemcpy(SoundSpeed, SoundSpeed_d, size_grid*sizeof(double), cudaMemcpyDeviceToHost));
  //
  // FILE *f;
  // f = fopen("cs.txt", "w");
  // for (int i = 0; i < size_grid; i++) {
  //   fprintf(f, "%.10f\n", SoundSpeed[i]);
  // }
  // fclose(f);
}


__host__ void ComputeTemperatureField ()
{
  ComputeTemperatureFieldKernel<<<dimGrid2, dimBlock2>>>(Dens_d, Temperature_d, Pressure_d, Energy_d,
    ADIABATICINDEX, Adiabatic, NSEC, NRAD);
  gpuErrchk(cudaDeviceSynchronize());

  // gpuErrchk(cudaMemcpy(Temperature, Temperature_d, size_grid*sizeof(double), cudaMemcpyDeviceToHost));
  //
  // FILE *f;
  // f = fopen("temp.txt", "w");
  // for (int i = 0; i < size_grid; i++) {
  //   fprintf(f, "%.10f\n", Temperature[i]);
  // }
  // fclose(f);
}


__host__ void ActualiseGasVtheta (double *Vtheta, double *VthetaNew)
{
  gpuErrchk(cudaMemcpy(Vtheta_d, VthetaNew_d, size_grid*sizeof(double), cudaMemcpyDeviceToDevice));
  gpuErrchk(cudaDeviceSynchronize());
}


__host__ void ActualiseGasVrad (double *Vrad, double *VradNew)
{
  gpuErrchk(cudaMemcpy(Vrad_d, VradNew_d, size_grid*sizeof(double), cudaMemcpyDeviceToDevice));
  gpuErrchk(cudaDeviceSynchronize());
}


__host__ void ActualiseGasEnergy (double *Energy, double *EnergyNew)
{
  gpuErrchk(cudaMemcpy(Energy_d, EnergyNew_d, size_grid*sizeof(double), cudaMemcpyDeviceToDevice));
  gpuErrchk(cudaDeviceSynchronize());
}


__host__ void Substep1cudamalloc (double *Vrad, double *Vtheta)
{
  gpuErrchk(cudaMemcpy(EnergyMed_d, EnergyMed,           NRAD*sizeof(double), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(SigmaMed_d, SigmaMed,             NRAD*sizeof(double), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(QplusMed_d, QplusMed,             NRAD*sizeof(double), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(CoolingTimeMed_d, CoolingTimeMed, NRAD*sizeof(double), cudaMemcpyHostToDevice));
}


__host__ int ConditionCFL (double *Vrad, double *Vtheta , double DeltaT)
{
  gpuErrchk(cudaMemset(Vmoy_d, 0, NRAD*sizeof(double)));
  ConditionCFLKernel1D<<<dimGrid4, dimBlock>>>(Rsup_d, Rinf_d, Rmed_d, NRAD, NSEC, Vtheta_d, Vmoy_d);
  gpuErrchk(cudaDeviceSynchronize());


  gpuErrchk(cudaMemset(DT2D_d, 0, NRAD*NSEC*sizeof(double)));
  gpuErrchk(cudaMemset(DT1D_d, 0, NRAD*sizeof(double)));
  gpuErrchk(cudaMemset(CFL_d, 0, sizeof(int)));
  ConditionCFLKernel2D1<<<dimGrid2, dimBlock2>>>(Rsup_d, Rinf_d, Rmed_d, NSEC, NRAD,
    Vresidual_d, Vtheta_d, Vmoy_d, FastTransport, SoundSpeed_d, Vrad_d, CVNR, DT2D_d,
    CFLSECURITY);
  gpuErrchk(cudaDeviceSynchronize());

  ConditionCFLKernel2D2<<<dimGrid4, dimBlock>>>(newDT_d, DT2D_d, DT1D_d, Vmoy_d, invRmed_d,
    CFL_d, NSEC, NRAD, CFLSECURITY, DeltaT);
  gpuErrchk(cudaDeviceSynchronize());

  gpuErrchk(cudaMemcpy(CFL, CFL_d,  sizeof(int), cudaMemcpyDeviceToHost));

  return CFL[0];
}


__host__ double CircumPlanetaryMass (double *Dens, PlanetarySystem *sys)
{
  double xpl, ypl, mdcp0;

  xpl = sys->x[0];
  ypl = sys->y[0];

  CircumPlanetaryMassKernel<<<dimGrid2, dimBlock2>>> (Dens_d, Surf_d, CellAbscissa_d, CellOrdinate_d, xpl, ypl, NRAD, NSEC, \
    HillRadius, mdcp0_d);
  gpuErrchk(cudaDeviceSynchronize());

  // reduction mdcp
  mdcp0 = DeviceReduce(mdcp0_d, NRAD*NSEC);

  return mdcp0;
}
