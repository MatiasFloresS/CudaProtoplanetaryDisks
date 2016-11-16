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

extern float RMAX, RMIN, ADIABATICINDEX, FLARINGINDEX, ASPECTRATIO;
extern float SIGMA0, SIGMASLOPE, IMPOSEDDISKDRIFT, PhysicalTime, DT, MASSTAPER;
extern float SGP_eps, TRANSITIONWIDTH, TRANSITIONRATIO, TRANSITIONRADIUS;
extern float LAMBDADOUBLING, PhysicalTimeInitial, mdcp;

extern float *GLOBAL_bufarray, *vt_int, *SigmaInf, *CoolingTimeMed, *QplusMed , *viscosity_array;
extern float *SG_Accr, *array, *Qplus, *SigmaMed, *QplusMed, *EnergyMed, *CellOrdinate, *CellAbscissa;

extern float *Dens_d, *SG_Accr_d, *SG_Acct_d, *GLOBAL_bufarray_d, *array_d;
extern float *Qplus_d, *EnergyInt_d, *EnergyNew_d, *VradNew_d, *Potential_d;
extern float *VthetaInt_d, *powRmed_d, *VthetaNew_d, *SigmaMed_d, *QplusMed_d;
extern float *CoolingTimeMed_d, *EnergyMed_d, *DensInt_d, *DivergenceVelocity_d, *TAURR_d, *TAURP_d;
extern float *TAUPP_d, *Vmoy_d, *CellOrdinate_d, *CellAbscissa_d, *mdcp0_d, *Surf_d;

extern double *invdiffRmed_d, *invRinf_d, *Rmed_d, *invRmed_d, *invdiffRsup_d, *Rsup_d;

double *invdiffRmed, *invRinf, *Rinf, *Rinf_d, *invRmed, *Rmed, *invdiffRsup, *Rsup;

float *Pressure, *SoundSpeed, *Temperature, *vt_cent, *DensStar;
float *invSurf, *Radii, *Surf, *powRmed, *VradInt;
float *VthetaInt, *DensInt, *VradNew, *VthetaNew, *EnergyInt, *Potential, *EnergyNew, *TemperInt;

float *Vtheta_d, *SigmaInf_d, *Vrad_d, *SoundSpeed_d, *Energy_d, *Pressure_d;
float *Temperature_d, *viscosity_array_d, *vt_cent_d, *DensStar_d, *TemperInt_d, *VradInt_d, *DT1D_d;
float *DT2D_d, *newDT_d, *Vresidual_d, *Vradial_d, *Vazimutal_d;


float *DT2D;
float exces_mdcp = 0.0, mdcp1, MassTaper;

int CrashedDens, CrashedEnergy;

extern dim3 dimGrid2, dimBlock2, dimGrid4, dimBlock;

int init = 0;

extern double OmegaFrame, HillRadius;

Pair DiskOnPrimaryAcceleration;



__host__ void FillPolar1DArrays ()
{
  FILE *input, *output;
  int i,j;
  float drrsep, temporary;
  double *Radii2, *Rmed2;
  string InputName, OutputName;
  drrsep = (RMAX-RMIN)/(float)NRAD;
  InputName = OUTPUTDIR + "radii.dat";
  OutputName = OUTPUTDIR + "used_rad.dat";

  /* Creo los arreglos de FillPolar1DArrays */
  Radii       = (float *)malloc((NRAD+1)*sizeof(float));
  Radii2       = (double *)malloc((NRAD+1)*sizeof(double));
  Rmed2       = (double *)malloc(NRAD*sizeof(double));
  Rinf        = (double *)malloc(NRAD*sizeof(double));
  Rmed        = (double *)malloc(NRAD*sizeof(double));
  Rsup        = (double *)malloc(NRAD*sizeof(double));
  Surf        = (float *)malloc(NRAD*sizeof(float));
  invRinf     = (double *)malloc(NRAD*sizeof(double));
  invSurf     = (float *)malloc(NRAD*sizeof(float));
  invRmed     = (double *)malloc(NRAD*sizeof(double));
  invdiffRsup = (double *)malloc(NRAD*sizeof(double));
  invdiffRmed = (double *)malloc(NRAD*sizeof(double));
  vt_cent     = (float *)malloc(NRAD*sizeof(float));
  powRmed     = (float *)malloc(NRAD*sizeof(float));
  DT2D     = (float *)malloc(size_grid*sizeof(float));

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
           los pasamos a float */
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
      Radii[i] = (float)temporary;
    }
  }

  float numero=0;
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



__host__ void InitEuler (float *Vrad, float *Vtheta, float *Dens, float *Energy)
{
  InitTransport ();
  InitViscosity ();
  DensStar        = (float *)malloc(size_grid*sizeof(float));
  DensInt         = (float *)malloc(size_grid*sizeof(float));
  VradNew         = (float *)malloc(size_grid*sizeof(float));
  VradInt         = (float *)malloc(size_grid*sizeof(float));
  VthetaNew       = (float *)malloc(size_grid*sizeof(float));
  VthetaInt       = (float *)malloc(size_grid*sizeof(float));
  EnergyNew       = (float *)malloc(size_grid*sizeof(float));
  EnergyInt       = (float *)malloc(size_grid*sizeof(float));
  TemperInt       = (float *)malloc(size_grid*sizeof(float));
  Potential       = (float *)malloc(size_grid*sizeof(float));
  Pressure        = (float *)malloc(size_grid*sizeof(float));
  SoundSpeed      = (float *)malloc(size_grid*sizeof(float));
  Temperature     = (float *)malloc(size_grid*sizeof(float));
  Qplus           = (float *)malloc(size_grid*sizeof(float));

  Computecudamalloc (Energy);

  InitComputeAccel ();
  /* Rho and Energy are already initialized: cf main.cu*/
  ComputeSoundSpeed ();
  ComputePressureField ();
  ComputeTemperatureField ();
  InitGasVelocities (Vrad, Vtheta);

}



__host__ void AlgoGas (Force *force, float *Dens, float *Vrad, float *Vtheta, float *Energy, float *Label,
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

  while (dtemp < 0.99999*DT){
    MassTaper = PhysicalTime/(MASSTAPER*2.0*M_PI);
    MassTaper = (MassTaper > 1.0 ? 1.0 : pow(sin(MassTaper*M_PI/2.0), 2.0));
    //printf("dtemp = %.10f\n",dtemp );
    //printf("MassTaper = %.10f\n",MassTaper );

    if(IsDisk == YES){
      if (SloppyCFL == NO){
        //printf("DT-dtemp = %.10f\n",DT-dtemp );
        gastimestepcfl = 1;
        gastimestepcfl = ConditionCFL(Vrad, Vtheta ,DT-dtemp);
        dt = (double)(DT-dtemp)/(double)gastimestepcfl;
        printf("gastimestepcfl %d\n", gastimestepcfl);
        //printf("dt %.10f\n", dt);
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
      //printf("domega  %.10f\n", domega);
      if (IsDisk == YES) CorrectVtheta (Vtheta, domega);
      OmegaFrame = OmegaNew;
    }
    //printf("OmegaFrame %.10f\n", OmegaFrame);
    RotatePsys (sys, OmegaFrame*dt);

    /* Now we update gas */
    if (IsDisk == YES){
      //ApplyBoundaryCondition (Dens, Energy, Vrad, Vtheta, dt);
      gpuErrchk(cudaMemcpy(Dens, Dens_d,     size_grid*sizeof(float), cudaMemcpyDeviceToHost));
      gpuErrchk(cudaMemcpy(Energy, Energy_d, size_grid*sizeof(float), cudaMemcpyDeviceToHost));
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

      /*gpuErrchk(cudaMemcpy(Vrad, Vrad_d, size_grid*sizeof(float), cudaMemcpyDeviceToHost));
      gpuErrchk(cudaMemcpy(Vtheta, Vtheta_d, size_grid*sizeof(float), cudaMemcpyDeviceToHost));

      FILE *f;
      f = fopen("vradnew.txt","w");
      for (int i = 0; i < NRAD*NSEC; i++) {
        fprintf(f, "%.10f\n", Vrad[i]);
      }
      fclose(f);

      f = fopen("vthetanew.txt","w");
      for (int i = 0; i < NRAD*NSEC; i++) {
        fprintf(f, "%.10f\n", Vtheta[i]);
      }
      fclose(f);

      */
      exit(1);

      //ApplyBoundaryCondition (Dens, Energy, Vrad, Vtheta, dt);

      if (Adiabatic){
        gpuErrchk(cudaMemcpy(Vradial_d, Vrad_d, size_grid*sizeof(float), cudaMemcpyDeviceToDevice));
        gpuErrchk(cudaMemcpy(Vazimutal_d, Vtheta_d, size_grid*sizeof(float), cudaMemcpyDeviceToDevice));
        ComputeViscousTerms (Vrad, Vtheta, Dens);
        gpuErrchk(cudaMemcpy(Vrad_d, Vradial_d, size_grid*sizeof(float), cudaMemcpyDeviceToDevice));
        gpuErrchk(cudaMemcpy(Vtheta_d, Vazimutal_d, size_grid*sizeof(float), cudaMemcpyDeviceToDevice));

        Substep3 (Dens, dt);

        ActualiseGasEnergy (Energy, EnergyNew);
      }
      //Transport (Dens, Vrad, Vtheta, Energy, Label, dt);
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
}



__host__ void Substep1 (float *Dens, float *Vrad, float *Vtheta, float dt, int initialization)
{
  int selfgravityupdate;
  if(initialization == 0) Substep1cudamalloc(Vrad, Vtheta);

  Substep1Kernel<<<dimGrid2, dimBlock2>>>(Pressure_d, Dens_d, VradInt_d, invdiffRmed_d, Potential_d, Rinf_d,
    invRinf_d, Vrad_d, VthetaInt_d, Vtheta_d, Rmed_d,  dt, NRAD, NSEC, OmegaFrame, ZMPlus,
    IMPOSEDDISKDRIFT, SIGMASLOPE, powRmed_d);
  gpuErrchk(cudaDeviceSynchronize());
  //exit(1);

  gpuErrchk(cudaMemcpy(VradInt, VradInt_d, size_grid*sizeof(float), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(VthetaInt, VthetaInt_d, size_grid*sizeof(float), cudaMemcpyDeviceToHost));


  FILE *f;
  f = fopen("Vradint.txt","w");
  for (int i = 0; i < (NRAD+1)*NSEC; i++) {
    fprintf(f, "%.10f\n", VradInt[i]);
  }

  fclose(f);

  f = fopen("Vthetaint.txt","w");
  for (int i = 0; i < (NRAD+1)*NSEC; i++) {
    fprintf(f, "%.10f\n", VthetaInt[i]);
  }
  fclose(f);

  if (SelfGravity){
    selfgravityupdate = YES;

    /* We copy VradInt to Vradial -> device to device */
    gpuErrchk(cudaMemcpy(Vradial_d, VradInt_d, size_grid*sizeof(float), cudaMemcpyDeviceToDevice));
    gpuErrchk(cudaMemcpy(Vazimutal_d, VthetaInt_d, size_grid*sizeof(float), cudaMemcpyDeviceToDevice));

    compute_selfgravity(Dens, dt, selfgravityupdate, 0);
    /* Vradialto VradInt -> device to device */
    gpuErrchk(cudaMemcpy(VradInt_d, Vradial_d, size_grid*sizeof(float), cudaMemcpyDeviceToDevice));
    gpuErrchk(cudaMemcpy(VthetaInt_d, Vazimutal_d, size_grid*sizeof(float), cudaMemcpyDeviceToDevice));
  }

  gpuErrchk(cudaMemcpy(Vradial_d, VradInt_d, size_grid*sizeof(float), cudaMemcpyDeviceToDevice));
  gpuErrchk(cudaMemcpy(Vazimutal_d, VthetaInt_d, size_grid*sizeof(float), cudaMemcpyDeviceToDevice));
  ComputeViscousTerms (VradInt, VthetaInt, Dens);
  gpuErrchk(cudaMemcpy(VradInt_d, Vradial_d, size_grid*sizeof(float), cudaMemcpyDeviceToDevice));
  gpuErrchk(cudaMemcpy(VthetaInt_d, Vazimutal_d, size_grid*sizeof(float), cudaMemcpyDeviceToDevice));

  UpdateVelocitiesWithViscosity(VradInt, VthetaInt, Dens, dt);

  if (!Evanescent)
    ApplySubKeplerianBoundary(VthetaInt);
}



__host__ void Substep2 (float dt)
{
  Substep2Kernel<<<dimGrid2, dimBlock2>>>(Dens_d, VradInt_d, VthetaInt_d, TemperInt_d, NRAD, NSEC, CVNR, invdiffRmed_d,
  invdiffRsup_d, DensInt_d, Adiabatic, Rmed_d, dt, VradNew_d, VthetaNew_d, Energy_d, EnergyInt_d);
  gpuErrchk(cudaDeviceSynchronize());
}



__host__ void Substep3 (float *Dens, float dt)
{
  for (int i = 0; i < NRAD; i++) viscosity_array[i] = FViscosity(Rmed[i]);
  gpuErrchk(cudaMemcpy(viscosity_array_d, viscosity_array, (NRAD+1)*sizeof(float), cudaMemcpyHostToDevice));

  Substep3Kernel<<<dimGrid2, dimBlock2>>>(Dens_d, Qplus_d, viscosity_array_d, TAURR_d, TAURP_d , TAUPP_d, DivergenceVelocity_d,
     NRAD, NSEC, Rmed_d, Cooling, EnergyNew_d, dt, EnergyMed_d, SigmaMed_d, CoolingTimeMed_d, Energy_d,
     ADIABATICINDEX, QplusMed_d);
  gpuErrchk(cudaDeviceSynchronize());
}


__host__ void Computecudamalloc (float *Energy)
{

  CoolingTimeMed  = (float *)malloc(size_grid*sizeof(float));
  QplusMed        = (float *)malloc(size_grid*sizeof(float));
  viscosity_array = (float *)malloc((NRAD+1)*sizeof(float));

  gpuErrchk(cudaMalloc((void**)&TemperInt_d,   size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&Temperature_d, size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&Pressure_d,    size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&SoundSpeed_d,  size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&DensStar_d,    size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&VradInt_d,     size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&DensInt_d,     size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&VradNew_d,     size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&VthetaNew_d,   size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&Potential_d,   size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&VthetaInt_d,   size_grid*sizeof(float)));

  gpuErrchk(cudaMemset(TemperInt_d, 0, size_grid*sizeof(float)));
  gpuErrchk(cudaMemset(Temperature_d, 0, size_grid*sizeof(float)));
  gpuErrchk(cudaMemset(Pressure_d, 0, size_grid*sizeof(float)));
  gpuErrchk(cudaMemset(SoundSpeed_d, 0, size_grid*sizeof(float)));
  gpuErrchk(cudaMemset(DensStar_d, 0, size_grid*sizeof(float)));
  gpuErrchk(cudaMemset(VradInt_d, 0, size_grid*sizeof(float)));
  gpuErrchk(cudaMemset(DensInt_d, 0, size_grid*sizeof(float)));
  gpuErrchk(cudaMemset(VradNew_d, 0, size_grid*sizeof(float)));
  gpuErrchk(cudaMemset(VthetaNew_d, 0, size_grid*sizeof(float)));
  gpuErrchk(cudaMemset(Potential_d, 0, size_grid*sizeof(float)));
  gpuErrchk(cudaMemset(VthetaInt_d, 0, size_grid*sizeof(float)));


  gpuErrchk(cudaMalloc((void**)&SigmaInf_d,        NRAD*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&vt_cent_d,         (NRAD+1)*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&viscosity_array_d, (NRAD+1)*sizeof(float)));

  gpuErrchk(cudaMemcpy(SigmaInf_d, SigmaInf,               NRAD*sizeof(float), cudaMemcpyHostToDevice));

  gpuErrchk(cudaMalloc((void**)&Energy_d,   size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&EnergyInt_d,   size_grid*sizeof(float)));
  gpuErrchk(cudaMemset(Energy_d, 0, size_grid*sizeof(float)));

  if (Adiabatic)
  gpuErrchk(cudaMemcpy(Energy_d, Energy,  size_grid*sizeof(float), cudaMemcpyHostToDevice));

}


__host__ float ConstructSequence (float *u, float *v, int n)
{
  int i;
  float lapl = 0.0;

  for (i = 1; i < n; i++) u[i] = 2.0*v[i]-u[i-1];
  for (i = 1; i < n-1; i++) lapl += fabs(u[i+1]+u[i-1]-2.0*u[i]);

  return lapl;
}


__host__ void Init_azimutalvelocity_withSG (float *Vtheta)
{
  // !SGZeroMode
  gpuErrchk(cudaMemcpy(SG_Accr, SG_Accr_d, size_grid*sizeof(float), cudaMemcpyDeviceToHost));
  Make1Dprofile(1);

  Azimutalvelocity_withSGKernel<<<dimGrid2, dimBlock2>>>(Vtheta_d, Rmed_d, FLARINGINDEX, SIGMASLOPE, ASPECTRATIO,
    GLOBAL_bufarray_d, NRAD, NSEC);
  gpuErrchk(cudaDeviceSynchronize());
}


__host__ int DetectCrash (float *array)
{
  int Crash = NO;
  float numCrush;

  gpuErrchk(cudaMemcpy(array_d, array, size_grid*sizeof(float), cudaMemcpyHostToDevice));
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

  // gpuErrchk(cudaMemcpy(Pressure, Pressure_d, size_grid*sizeof(float), cudaMemcpyDeviceToHost));
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

  // gpuErrchk(cudaMemcpy(SoundSpeed, SoundSpeed_d, size_grid*sizeof(float), cudaMemcpyDeviceToHost));
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

  // gpuErrchk(cudaMemcpy(Temperature, Temperature_d, size_grid*sizeof(float), cudaMemcpyDeviceToHost));
  //
  // FILE *f;
  // f = fopen("temp.txt", "w");
  // for (int i = 0; i < size_grid; i++) {
  //   fprintf(f, "%.10f\n", Temperature[i]);
  // }
  // fclose(f);
}


__host__ void ActualiseGasVtheta (float *Vtheta, float *VthetaNew)
{
  gpuErrchk(cudaMemcpy(Vtheta_d, VthetaNew_d, size_grid*sizeof(float), cudaMemcpyDeviceToDevice));
  gpuErrchk(cudaDeviceSynchronize());
}

__host__ void ActualiseGasVrad (float *Vrad, float *VradNew)
{
  gpuErrchk(cudaMemcpy(Vrad_d, VradNew_d, size_grid*sizeof(float), cudaMemcpyDeviceToDevice));
  gpuErrchk(cudaDeviceSynchronize());
}

__host__ void ActualiseGasEnergy (float *Energy, float *EnergyNew)
{
  gpuErrchk(cudaMemcpy(Energy_d, EnergyNew_d, size_grid*sizeof(float), cudaMemcpyDeviceToDevice));
  gpuErrchk(cudaDeviceSynchronize());
}

__host__ void Substep1cudamalloc (float *Vrad, float *Vtheta)
{
  gpuErrchk(cudaMemcpy(EnergyMed_d, EnergyMed,           NRAD*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(SigmaMed_d, SigmaMed,             NRAD*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(QplusMed_d, QplusMed,             NRAD*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(CoolingTimeMed_d, CoolingTimeMed, NRAD*sizeof(float), cudaMemcpyHostToDevice));
}

__host__ int ConditionCFL (float *Vrad, float *Vtheta , float DeltaT)
{
  ConditionCFLKernel1D<<<dimGrid4, dimBlock>>>(Rsup_d, Rinf_d, Rmed_d, NRAD, NSEC, Vtheta_d, Vmoy_d);
  gpuErrchk(cudaDeviceSynchronize());


  gpuErrchk(cudaMemset(DT2D_d, 0, NRAD*NSEC*sizeof(float)));
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


__host__ float CircumPlanetaryMass (float *Dens, PlanetarySystem *sys)
{
  float xpl, ypl, mdcp0;

  xpl = sys->x[0];
  ypl = sys->y[0];

  CircumPlanetaryMassKernel<<<dimGrid2, dimBlock2>>> (Dens_d, Surf_d, CellAbscissa_d, CellOrdinate_d, xpl, ypl, NRAD, NSEC, \
    HillRadius, mdcp0_d);
  gpuErrchk(cudaDeviceSynchronize());

  // reduction mdcp
  mdcp0 = DeviceReduce(mdcp0_d, NRAD*NSEC);

  return mdcp0;
}
