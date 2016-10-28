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

extern float RMAX, RMIN, ADIABATICINDEX, FLARINGINDEX, OmegaFrame, ASPECTRATIO;
extern float SIGMA0, SIGMASLOPE, IMPOSEDDISKDRIFT, PhysicalTime, DT, MASSTAPER, HillRadius;
extern float SGP_eps, mdcp, TRANSITIONWIDTH, TRANSITIONRATIO, TRANSITIONRADIUS;
extern float LAMBDADOUBLING, PhysicalTimeInitial;

extern float *GLOBAL_bufarray, *vt_int, *SigmaInf, *CoolingTimeMed, *QplusMed , *viscosity_array;
extern float *SG_Accr, *array, *Qplus, *SigmaMed, *QplusMed, *EnergyMed, *CellOrdinate;

extern float *Dens_d, *Rmed_d, *SG_Accr_d, *SG_Acct_d, *GLOBAL_bufarray_d, *array_d, *invRinf_d;
extern float *Qplus_d, *EnergyInt_d, *EnergyNew_d, *VradNew_d, *invdiffRsup_d, *Potential_d;
extern float *VthetaInt_d, *powRmed_d, *invdiffRmed_d, *VthetaNew_d, *SigmaMed_d, *QplusMed_d;
extern float *CoolingTimeMed_d, *EnergyMed_d, *DensInt_d, *DivergenceVelocity_d, *TAURR_d, *TAURP_d;
extern float *TAUPP_d, *Rsup_d, *Vmoy_d, *invRmed_d, *CellOrdinate_d, *CellAbscissa_d, *mdcp0_d, *Surf_d;

float *Pressure, *SoundSpeed, *Temperature, *vt_cent, *DensStar, *invRmed, *invRinf;
float *invSurf, *invdiffRmed, *invdiffRsup, *Radii, *Rinf, *Rmed, *Rsup, *Surf, *powRmed, *VradInt;
float *VthetaInt, *DensInt, *VradNew, *VthetaNew, *EnergyInt, *Potential, *EnergyNew, *TemperInt;

float *Vtheta_d, *Rinf_d, *SigmaInf_d, *Vrad_d, *SoundSpeed_d, *Energy_d, *Pressure_d;
float *Temperature_d, *viscosity_array_d, *vt_cent_d, *DensStar_d, *TemperInt_d, *VradInt_d, *DT1D_d;
float *DT2D_d, *newDT_d, *Vresidual_d, *Vradial_d, *Vazimutal_d;


float *DT2D;
float exces_mdcp = 0.0, mdcp1, MassTaper;

int CrashedDens, CrashedEnergy;

extern dim3 dimGrid2, dimBlock2, dimGrid4, dimBlock;

int init = 0;
double *Radii2;

Pair DiskOnPrimaryAcceleration;



__host__ void FillPolar1DArrays ()
{
  FILE *input, *output;
  int i,j;
  float drrsep, temporary;
  string InputName, OutputName;
  drrsep = (RMAX-RMIN)/(float)NRAD;
  InputName = OUTPUTDIR + "radii.dat";
  OutputName = OUTPUTDIR + "used_rad.dat";

  /* Creo los arreglos de FillPolar1DArrays */
  Radii       = (float *)malloc((NRAD+1)*sizeof(float));
  Radii2      = (double *)malloc((NRAD+1)*sizeof(double));
  Rinf        = (float *)malloc(NRAD*sizeof(float));
  Rmed        = (float *)malloc(NRAD*sizeof(float));
  Rsup        = (float *)malloc(NRAD*sizeof(float));
  Surf        = (float *)malloc(NRAD*sizeof(float));
  invRinf     = (float *)malloc(NRAD*sizeof(float));
  invSurf     = (float *)malloc(NRAD*sizeof(float));
  invRmed     = (float *)malloc(NRAD*sizeof(float));
  invdiffRsup = (float *)malloc(NRAD*sizeof(float));
  invdiffRmed = (float *)malloc(NRAD*sizeof(float));
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
        Radii2[i] = RMIN*exp((double)i/(double)NRAD*log(RMAX / RMIN));
        Radii[i] = (float) Radii2[i];
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

  for (i = 0; i < NRAD; i++){
    Rinf[i] = (float) Radii2[i];
    Rsup[i] = (float) Radii2[i+1];
    Rmed[i] = (float)(2.0/3.0*(Radii2[i+1]*Radii2[i+1]*Radii2[i+1]-Radii2[i]*Radii2[i]*Radii2[i])); // 2/3*(Rsup[i]^3 - Rinf[i]^3)
    Rmed[i] = Rmed[i] / (float)(Radii2[i+1]*Radii2[i+1]-Radii2[i]*Radii2[i]); // Rmed /(Rsup[i]^2 - Rinf[i]^2)
    Surf[i] = (float) (M_PI*(Radii2[i+1]*Radii2[i+1]-Radii2[i]*Radii2[i])/(float)NSEC);  // (Rsup[i]^2 - Rinf[i]^2)
    invRmed[i] = 1.0/Rmed[i];
    invSurf[i] = 1.0/Surf[i];
    invdiffRsup[i] = (float) (1.0/(Radii2[i+1]-Radii2[i])); // 1.0/(Rsup[i] - Rinf[i])
    invRinf[i] = 1.0/Rinf[i];
  }

  Rinf[NRAD] = Radii2[NRAD];

  for (i = 1; i < NRAD; i++) {
    invdiffRmed[i] = 1.0/(Rmed[i]-Rmed[i-1]);
    powRmed[i] = (float) pow(Rmed[i],-2.5+SIGMASLOPE);
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
    fprintf (output, "%f\n", Radii[i]);
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

  // gpuErrchk(cudaMemcpy(Dens, Dens_d, size_grid*sizeof(float), cudaMemcpyDeviceToHost));
  // gpuErrchk(cudaMemcpy(Energy, Energy_d, size_grid*sizeof(float), cudaMemcpyDeviceToHost));
  // gpuErrchk(cudaMemcpy(SoundSpeed, SoundSpeed_d, size_grid*sizeof(float), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(CellOrdinate, CellOrdinate_d, size_grid*sizeof(float), cudaMemcpyDeviceToHost));
  //


  // FILE *f;
  // f = fopen("ord.txt", "w");
  // for (int i = 0; i < size_grid; i++) {
  //   fprintf(f, "%g\n", CellOrdinate[i]);
  // }
  // fclose(f);
  // exit(1);
  // f = fopen("pressure.txt", "w");
  // for (int i = 0; i < size_grid; i++) {
  //   fprintf(f, "%f\n", Pressure[i]);
  // }
  // fclose(f);
  // exit(1);
  //
  // f = fopen("vrad.txt", "w");
  // for (int i = 0; i < size_grid; i++) {
  //   fprintf(f, "%f\n", Vrad[i]);
  // }
  // fclose(f);
  //
  // f = fopen("vtheta.txt", "w");
  // for (int i = 0; i < size_grid; i++) {
  //   fprintf(f, "%f\n", Vtheta[i]);
  // }
  // fclose(f);
  //
  // exit(1);

}



__host__ void AlgoGas (Force *force, float *Dens, float *Vrad, float *Vtheta, float *Energy, float *Label,
  PlanetarySystem *sys, int initialization)
{
  float dt, dtemp =0.0;
  float OmegaNew, domega;
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

  while (dtemp < 0.99999*DT){
    MassTaper = PhysicalTime/(MASSTAPER*2.0*M_PI);
    MassTaper = (MassTaper > 1.0 ? 1.0 : pow(sin(MassTaper*M_PI/2.0), 2.0));
    if(IsDisk == YES){
      if (SloppyCFL == NO){
        gastimestepcfl = ConditionCFL(Vrad, Vtheta ,DT-dtemp);
        dt = (DT-dtemp)/(float)gastimestepcfl;
      }
      AccreteOntoPlanets(Dens, Vrad, Vtheta, dt, sys);
    }
    dtemp += dt;
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
      OmegaNew = GetPsysInfo(sys, GET);
      domega = OmegaNew - OmegaFrame;
      if (IsDisk == YES) CorrectVtheta (Vtheta, domega);
      OmegaFrame = OmegaNew;
    }
    RotatePsys (sys, OmegaFrame*dt);

    /* Now we update gas */
    if (IsDisk == YES){
      ApplyBoundaryCondition (Dens, Energy, Vrad, Vtheta, dt);
      gpuErrchk(cudaMemcpy(Dens, Dens_d,     size_grid*sizeof(float), cudaMemcpyDeviceToHost));
      gpuErrchk(cudaMemcpy(Energy, Energy_d, size_grid*sizeof(float), cudaMemcpyDeviceToHost));
      CrashedDens = DetectCrash (Dens);
      CrashedEnergy = DetectCrash (Energy);
      if (CrashedDens == YES || CrashedEnergy == YES){
        fprintf(stdout, "\nCrash! at time %d\n", PhysicalTime);
      }
      // if (ZMPlus) compute_anisotropic_pressurecoeff(sys);

      ComputePressureField ();
      Substep1 (Dens, Vrad, Vtheta, dt, init);
      Substep2 (dt);
      ActualiseGasVrad (Vrad, VradNew);
      ActualiseGasVtheta (Vtheta, VthetaNew);
      ApplyBoundaryCondition (Dens, Energy, Vrad, Vtheta, dt);

      if (Adiabatic){
        ComputeViscousTerms (Vrad, Vtheta, Dens, 1);
        Substep3 (Dens, dt);
        ActualiseGasEnergy (Energy, EnergyNew);
      }
      Transport (Dens, Vrad, Vtheta, Energy, Label, dt);
      ApplyBoundaryCondition(Dens, Energy, Vrad, Vtheta, dt);
      ComputeTemperatureField ();
      mdcp1 = CircumPlanetaryMass (Dens, sys);
      exces_mdcp = mdcp - mdcp1;
      init = init + 1;
    }

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


  gpuErrchk(cudaMemcpy(VthetaInt, VthetaInt_d, size_grid*sizeof(float), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(VradInt, VradInt_d, size_grid*sizeof(float), cudaMemcpyDeviceToHost));

  FILE *f;

  f = fopen("vthetaint.txt","w");
  for (int i = 0; i < NRAD*NSEC; i++) {
    fprintf(f, "%g\n", VthetaInt[i]);
  }
  fclose(f);

  f = fopen("vradint.txt","w");
  for (int i = 0; i < NRAD*NSEC; i++) {
    fprintf(f, "%g\n", VradInt[i]);
  }
  fclose(f);

  exit(1);

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

  ComputeViscousTerms (VradInt, VthetaInt, Dens, 0);
  UpdateVelocitiesWithViscosity(VradInt, VthetaInt, Dens, dt);

  if (!Evanescent) ApplySubKeplerianBoundary(VthetaInt);

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
  gpuErrchk(cudaMalloc((void**)&EnergyInt_d,   size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&Potential_d,   size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&VthetaInt_d,   size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&Energy_d,   size_grid*sizeof(float)));

  gpuErrchk(cudaMemset(Pressure_d, 0, size_grid*sizeof(float)));
  gpuErrchk(cudaMemset(SoundSpeed_d, 0, size_grid*sizeof(float)));
  gpuErrchk(cudaMemset(Energy_d, 0, size_grid*sizeof(float)));

  gpuErrchk(cudaMalloc((void**)&SigmaInf_d,        NRAD*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&vt_cent_d,         (NRAD+1)*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&viscosity_array_d, (NRAD+1)*sizeof(float)));

  gpuErrchk(cudaMemcpy(SigmaInf_d, SigmaInf,               NRAD*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(Energy_d, Energy,             size_grid*sizeof(float), cudaMemcpyHostToDevice));

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
}

__host__ void ComputeSoundSpeed ()
{
  ComputeSoundSpeedKernel<<<dimGrid2, dimBlock2>>>(SoundSpeed_d, Dens_d, Rmed_d, Energy_d, NSEC, NRAD,
    Adiabatic, ADIABATICINDEX, FLARINGINDEX, ASPECTRATIO, TRANSITIONWIDTH, TRANSITIONRADIUS,
    TRANSITIONRATIO, PhysicalTime, PhysicalTimeInitial, LAMBDADOUBLING);
  gpuErrchk(cudaDeviceSynchronize());
}

__host__ void ComputeTemperatureField ()
{
  ComputeTemperatureFieldKernel<<<dimGrid2, dimBlock2>>>(Dens_d, Temperature_d, Pressure_d, Energy_d,
    ADIABATICINDEX, Adiabatic, NSEC, NRAD);
  gpuErrchk(cudaDeviceSynchronize());
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
  //gpuErrchk(cudaMemcpy(TemperInt_d, TemperInt,           size_grid*sizeof(float), cudaMemcpyHostToDevice));
  //gpuErrchk(cudaMemcpy(DensInt_d, DensInt,               size_grid*sizeof(float), cudaMemcpyHostToDevice));
  //gpuErrchk(cudaMemcpy(VradNew_d, VradNew,               size_grid*sizeof(float), cudaMemcpyHostToDevice));
  //gpuErrchk(cudaMemcpy(VthetaNew_d, VthetaNew,           size_grid*sizeof(float), cudaMemcpyHostToDevice));
  //gpuErrchk(cudaMemcpy(energyInt_d, energyInt,           size_grid*sizeof(float), cudaMemcpyHostToDevice));
  //gpuErrchk(cudaMemcpy(Qplus_d, Qplus,                   size_grid*sizeof(float), cudaMemcpyHostToDevice));
  //gpuErrchk(cudaMemcpy(energyNew_d, energyNew,           size_grid*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(EnergyMed_d, EnergyMed,           NRAD*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(SigmaMed_d, SigmaMed,             NRAD*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(QplusMed_d, QplusMed,             NRAD*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(CoolingTimeMed_d, CoolingTimeMed, NRAD*sizeof(float), cudaMemcpyHostToDevice));
}

__host__ int ConditionCFL (float *Vrad, float *Vtheta , float DeltaT)
{
  ConditionCFLKernel1D<<<dimGrid4, dimBlock>>>(Rsup_d, Rinf_d, Rmed_d, NRAD, NSEC, Vtheta_d, Vmoy_d);
  gpuErrchk(cudaDeviceSynchronize());

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
  float xpl, ypl;

  xpl = sys->x[0];
  ypl = sys->y[0];

  CircumPlanetaryMassKernel<<<dimGrid2, dimBlock2>>> (Dens_d, Surf_d, CellAbscissa_d, CellOrdinate_d, xpl, ypl, NRAD, NSEC, \
    HillRadius, mdcp0_d);
  gpuErrchk(cudaDeviceSynchronize());

  // reduction mdcp
  mdcp = DeviceReduce(mdcp0_d, NRAD*NSEC);

  return mdcp;
}
