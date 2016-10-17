#include "Main.cuh"

extern int blocksize, nsec2pot, nrad2pot, NRAD, NSEC, YES, LogGrid, size_grid, SelfGravity, ViscosityAlpha,       \
Adiabaticc, Cooling, Corotating, MARK, NO, IsDisk;

extern bool CentrifugalBalance, ZMPlus = false, SloppyCFL;
extern string OUTPUTDIR;

extern float RMAX, RMIN, PI, MU, R, *GLOBAL_bufarray, ADIABATICINDEX, FLARINGINDEX, *vt_int, OmegaFrame1,         \
*SigmaInf, G, ASPECTRATIO, SIGMA0, SIGMASLOPE, IMPOSEDDISKDRIFT, *CoolingTimeMed, *QplusMed , *viscosity_array,   \
*Dens_d, *Rmed_d, *SG_Accr, *SG_Accr_d, *SG_Acct_d, *GLOBAL_bufarray_d, *array_d, *array, mdcp, PhysicalTime,     \
*Qplus, *Qplus_d, *energyInt_d, *energyNew_d, *VradNew_d, *invdiffRsup_d, *Potential_d, *invRinf_d, *VthetaInt_d, \
*powRmed_d, *invdiffRmed_d, *VthetaNew_d, *SigmaMed_d, *SigmaMed, *QplusMed_d, *QplusMed, *CoolingTimeMed_d,      \
*EnergyMed_d, *EnergyMed, *DensInt_d, CVNR, DT, MASSTAPER, *DivergenceVelocity_d, *TAURR_d, *TAURP_d, *TAUPP_d;

float *Pressure, *AspectRatioRmed, *SoundSpeed, *Temperature, *Vtheta_d, *vt_cent, *Rinf_d, *SigmaInf_d, *Vrad_d, \
*SoundSpeed_d, *energy_d, *AspectRatioRmed_d, *Pressure_d, *Temperature_d, *viscosity_array_d, *Kr_aux, *Kt_aux,  \
exces_mdcp = 0.0, mdcp1, MassTaper, *vt_cent_d, *DensStar_d, *DensStar, *invRmed, *invRinf, *invSurf, *invdiffRmed, \
*invdiffRsup, *Radii, *Rinf, *Rmed, *Rsup, *Surf, *TemperInt_d, *TemperInt, *VradInt, *VradInt_d, *powRmed,       \
*VthetaInt, *DensInt, *VradNew, *VthetaNew, *energyInt, *Potential, *energyNew;


bool CrashedDens, CrashedEnergy;

extern dim3 dimGrid2, dimBlock2;

int init = 0;
double *Radii2;

Pair DiskOnPrimaryAcceleration;

__host__ void FillPolar1DArrays ()
{
  FILE *input, *output;
  int i,j;
  float drrsep, temporary;
  double u, theta, base, kr, den_SGP_K, kt;
  string InputName, OutputName;
  drrsep = (RMAX-RMIN)/NRAD;
  InputName = OUTPUTDIR +"radii.dat";
  OutputName = OUTPUTDIR +"used_rad.dat";

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

  if(SelfGravity)
  {
    Kr_aux      = (float *)malloc(2*size_grid*sizeof(float));
    Kt_aux      = (float *)malloc(2*size_grid*sizeof(float));
  }

  char inputcharname[100];
  strncpy(inputcharname, InputName.c_str(), sizeof(inputcharname));
  inputcharname[sizeof(inputcharname)-1]=0;

  input = fopen (inputcharname, "r");
  if (input == NULL)
  {
    printf("Warning : no `radii.dat' file found. Using default.\n");
    if (LogGrid == YES)
    { printf("entro\n" );
      for (i = 0; i <= NRAD; i++)
      {
        Radii2[i] = RMIN*exp((double)i/(double)NRAD*log(RMAX / RMIN));
        Radii[i] = (float) Radii2[i];
      }

      if(SelfGravity)
      {
        for (i = 0; i < 2*NRAD; i++)
        {
          if(i<NRAD) u = log(Radii2[i]/Radii2[0]);
          else u = -log(Radii2[2*NRAD-i]/Radii2[0]);

          for (j = 0; j < NSEC; j++)
          {
            theta = 2.0*M_PI*(double)j  / (double)NSEC;
            base = 0.03*0.03 * exp(u) + 2.0* (cosh(u) - cos(theta));
            den_SGP_K = pow(base , -1.5);

            kr = 1.0 + 0.03*0.03 - cos(theta) * exp(-u);
            kr *= den_SGP_K;

            kt = sin(theta) * den_SGP_K;
            Kr_aux[i*NSEC+j] = (float) kr;
            Kt_aux[i*NSEC+j] = (float) kt;
          }
        }
      }
    }
    else {
      for (i = 0; i <= NRAD; i++) Radii[i] = RMIN+drrsep*i;
    }
  }
  else
  {
    printf("Reading 'radii.dat' file.\n");
    for (i = 0; i <= NRAD; i++)
    {
      fscanf (input, "%f", &temporary);
      Radii[i] = (float)temporary;
    }
  }

  for (i = 0; i < NRAD; i++)
  {
    Rinf[i] = Radii2[i];
    Rsup[i] = Radii2[i+1];
    Rmed[i] = 2.0/3.0*(Radii2[i+1]*Radii2[i+1]*Radii2[i+1]-Radii2[i]*Radii2[i]*Radii2[i]); // 2/3*(Rsup^3 - Rinf^3)
    Rmed[i] = Rmed[i] / (Radii2[i+1]*Radii2[i+1]-Radii2[i]*Radii2[i]); // Rmed /(Rsup^2 - Rinf^2)
    Surf[i] = M_PI*(Radii2[i+1]*Radii2[i+1]-Radii2[i]*Radii2[i])/(float)NSEC;  // (Rsup^2 - Rinf^2)
    invRmed[i] = 1.0/Rmed[i];
    invSurf[i] = 1.0/Surf[i];
    invdiffRsup[i] = 1.0/(Radii2[i+1]-Radii2[i]); // 1.0 / (Rsup - Rinf)
    invRinf[i] = 1.0/Rinf[i];
  }

  Rinf[NRAD]=Radii[NRAD];

  for (i = 1; i < NRAD; i++) invdiffRmed[i] = 1.0/(Rmed[i]-Rmed[i-1]);
  for (i = 0; i < NRAD; i++) powRmed[i] = pow(Rmed[i],-2.5+SIGMASLOPE);

  char outputcharname[100];
  strncpy(outputcharname, OutputName.c_str(), sizeof(outputcharname));
  outputcharname[sizeof(outputcharname)-1]=0;
  output = fopen (outputcharname, "w");
  if (output == NULL)
  {
    printf ("Can't write %s.\nProgram stopped.\n", outputcharname);
    exit (1);
  }
  for (i = 0; i <= NRAD; i++)fprintf (output, "%f\n", Radii[i]);

  fclose (output);
  if (input != NULL) fclose (input);
}

__host__ void InitEuler (float *Vrad, float *Vtheta, float *Dens, float *energy)
{
  InitTransport ();
  InitViscosity ();
  Pressure        = (float *)malloc(size_grid*sizeof(float));
  SoundSpeed      = (float *)malloc(size_grid*sizeof(float));
  Temperature     = (float *)malloc(size_grid*sizeof(float));
  TemperInt       = (float *)malloc(size_grid*sizeof(float));
  DensStar         = (float *)malloc(size_grid*sizeof(float));
  VradInt         = (float *)malloc(size_grid*sizeof(float));
  AspectRatioRmed = (float *)malloc(NRAD*sizeof(float));
  Potential       = (float *)malloc(size_grid*sizeof(float));
  VthetaInt       = (float *)malloc(size_grid*sizeof(float));
  DensInt         = (float *)malloc(size_grid*sizeof(float));
  VradNew         = (float *)malloc(size_grid*sizeof(float));
  VthetaNew       = (float *)malloc(size_grid*sizeof(float));
  energyInt       = (float *)malloc(size_grid*sizeof(float));
  energyNew       = (float *)malloc(size_grid*sizeof(float));

  for (int i = 0; i < NRAD; i++) AspectRatioRmed[i] = AspectRatio(Rmed[i]);

  Computecudamalloc (energy);

  InitComputeAccel ();
  ComputeSoundSpeed ();
  ComputePressureField (Dens, energy);
  ComputeTemperatureField ();
  InitGasVelocities (Vrad, Vtheta);

}

__host__ void InitGasVelocities (float *Vrad, float *Vtheta)
{
  float r1, t1, r2, t2, r, ri;
  int i;


  /* Pressure is already initialized: cf initeuler in SourceEuler.c ...
    Initialization of azimutal velocity with exact centrifugal balance */

  if (CentrifugalBalance)
  {
    /* vt_int \equiv Romega = grad(P)/sigma + \partial(phi)/\partial(r) - acc_sg_radial
    ./bin/fargoGPU  -b in/template.par */

    gpuErrchk(cudaMemcpy(Pressure, Pressure_d, size_grid*sizeof(float), cudaMemcpyDeviceToHost));
    Make1Dprofile(Pressure);

    /* global axisymmetric pressure field */
    for (i = 1; i < NRAD; i++)
    {
      vt_int[i] = ( GLOBAL_bufarray[i] - GLOBAL_bufarray[i-1]) / \
      (.5*(Sigma(Rmed[i]) + Sigma(Rmed[i-1])))/(Rmed[i]-Rmed[i-1]);
    }
    /* Case of a disk with self-gravity */
    if ( SelfGravity ) // Better test with CL rigid!
    {
      gpuErrchk(cudaMemcpy(SG_Accr, SG_Accr_d, size_grid*sizeof(float), cudaMemcpyDeviceToHost));
      Make1Dprofile(SG_Accr);

      for (i = 1; i < NRAD; i++)
        vt_int[i] -= ((Radii[i] - Rmed[i-1]) * GLOBAL_bufarray[i] + \
          (Rmed[i] - Radii[i])* GLOBAL_bufarray[i-1]) / (Rmed[i]-Rmed[i-1]);
    }

    for (i = 1; i < NRAD; i++)
      vt_int[i] = sqrt(vt_int[i]*Radii[i]) - Radii[i]*OmegaFrame1;

    t1 = vt_cent[0] = vt_int[1]+.75*(vt_int[1]-vt_int[2]);
    r1 = ConstructSequence (vt_cent, vt_int, NRAD);
    vt_cent[0] += .25*(vt_int[1]-vt_int[2]);
    t2 = vt_cent[0];
    r2 = ConstructSequence (vt_cent, vt_int, NRAD);
    t1 = t1-r1/(r2-r1)*(t2-t1);
    vt_cent[0] = t1;
    ConstructSequence (vt_cent, vt_int, NRAD);
    vt_cent[NRAD] = vt_cent[NRAD-1];
  }

  if (!CentrifugalBalance && SelfGravity)
    Init_azimutalvelocity_withSG (Vtheta);

  if (ViscosityAlpha)
  {
    gpuErrchk(cudaMemcpy(SoundSpeed, SoundSpeed_d, size_grid*sizeof(float), cudaMemcpyDeviceToHost));
    Make1Dprofile(SoundSpeed);
  }


  if (Cooling)
  {
    FillCoolingTime();
    FillQplus();
  }

  for (i = 0; i <= NRAD; i++)
  {
    if (i == NRAD) viscosity_array[i] = FViscosity(Rmed[NRAD-1]);
    else viscosity_array[i] = FViscosity(Rmed[i]);
  }

  InitVelocities(Vrad, Vtheta);
}

__host__ void AlgoGas (Force *force, float *Dens, float *Vrad, float *Vtheta, float *energy, float *label,
  PlanetarySystem *sys, int initialization)
{

  float dt = 0.0;
  int GasTimeStepsCFL = 1;
  CrashedDens = false;
  CrashedEnergy = false;
  if (Adiabaticc)
  {
    for (int i = 0; i < NRAD; i++) AspectRatioRmed[i] = AspectRatio(Rmed[i]);
    gpuErrchk(cudaMemcpy(AspectRatioRmed_d, AspectRatioRmed, NRAD*sizeof(float), cudaMemcpyHostToDevice));

    ComputeSoundSpeed();

    /* it is necesary to update computation of soundspeed if one uses
      alphaviscosity in Fviscosity. It is not necesary in locally
      isothermal runs since cs is constant. It is computed here for
      the needs of ConditionCFL. */
  }
  if (IsDisk == YES)
  {
    // communicateBoundaries -> es para mpi ?
    if (SloppyCFL == YES)
    {
      //gastimestepcfl = ConditionCFL(vrad, vtheta, DT-dtemp);      case ./bin/fargoGPU -c
    }

  }
  float dtemp = 0.;
  dt = DT / GasTimeStepsCFL; // es 1

  while (dtemp < 0.99999*DT)
  {
    MassTaper = PhysicalTime/(MASSTAPER*2.0*M_PI);
    MassTaper = (MassTaper > 1.0 ? 1.0 : pow(sin(MassTaper*M_PI/2.0), 2.0));

    if(IsDisk == YES)
    {
      // communicateBoundaries -> mismo que arriba
      if (SloppyCFL == NO)
      {
        // gastimestepcfl = 1;
        //gastimestepcfl = ConditionCFL(vrad,vtheta,DT-dtemp);
        dt = (DT-dtemp)/(float)GasTimeStepsCFL;
      }
      AccreteOntoPlanets(Dens, Vrad, Vtheta, dt, sys);
    }
    dtemp += dt;
    DiskOnPrimaryAcceleration.x = 0.0;
    DiskOnPrimaryAcceleration.y = 0.0;
    if (Corotating == YES) GetPsysInfo (sys, MARK);

    if (IsDisk == YES)
    {
      /* Indirect term star's potential computed here */
      DiskOnPrimaryAcceleration = ComputeAccel (force, Dens, 0.0, 0.0, 0.0, 0.0);
      /* Gravitational potential from star and planet(s) is computed and stored here */
      FillForcesArrays (sys, Dens, energy);
      /* Planet's velocities are update here from gravitational interaction with disk */
      // AdvanceSystemFromDisk (force, dens, energy, sys, dt);

    }

    /* Planet's positions and velocities are update from gravitational interaction with star
       and other planets */
    // AdvanceSystemRK5 (sys,dt);

    /* Below we correct vtheta, planet's position and velocities if we work in a frame non-centered on the star */
    if (Corotating == YES)
    {
      // OmegaNew = GetPsysInfo(sys, GET) / dt;
      // domega = OmegaNew - OmegaFrame1;
      // if (IsDisk == YES) CorrectVtheta (vtheta, domega);
      // OmegaFrame1 = OmegaNew;
    }
    // RotatePsys (sys, OmegaFrame1*dt);

    /* Now we update gas */
    if (IsDisk == YES)
    {
      ApplyBoundaryCondition (Dens, energy, Vrad, Vtheta, dt);
      gpuErrchk(cudaMemcpy(Dens, Dens_d,     size_grid*sizeof(float), cudaMemcpyDeviceToHost));
      gpuErrchk(cudaMemcpy(energy, energy_d, size_grid*sizeof(float), cudaMemcpyDeviceToHost));
      CrashedDens = DetectCrash (Dens);
      CrashedEnergy = DetectCrash (energy);
      if (CrashedDens == YES || CrashedEnergy == YES)
      {
        //fprintf(stdout, "\nCrash! at time %d\n", PhysicalTime);
      }
      // if (ZMPlus) compute_anisotropic_pressurecoeff(sys);

      ComputePressureField (Dens, energy);
      Substep1 (Dens, Vrad, Vtheta, dt, init);
      Substep2 (dt);
      ActualiseGasVrad (Vrad, VradNew);
      ActualiseGasVtheta (Vtheta, VthetaNew);
      ApplyBoundaryCondition (Dens, energy, Vrad, Vtheta, dt);

      if (Adiabaticc)
      {
        ComputeViscousTerms (Vrad, Vtheta, Dens, 1);
        Substep3 (Dens, dt);
        ActualiseGasEnergy (energy, energyNew);
      }
      Transport (Dens, Vrad, Vtheta, energy, label, dt);
      ApplyBoundaryCondition(Dens, energy, Vrad, Vtheta, dt);
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
  bool selfgravityupdate;
  if(initialization == 0) Substep1cudamalloc(Vrad, Vtheta);

  Substep1Kernel<<<dimGrid2, dimBlock2>>>(Pressure_d, Dens_d, VradInt_d, invdiffRmed_d, Potential_d, Rinf_d,
    invRinf_d, Vrad_d, VthetaInt_d, Vtheta_d, Rmed_d,  dt, NRAD, NSEC, OmegaFrame1, ZMPlus,
    IMPOSEDDISKDRIFT, SIGMASLOPE, powRmed_d);
  gpuErrchk(cudaDeviceSynchronize());

    if (SelfGravity){
      selfgravityupdate = YES;
      Compute_selfgravity(Dens, VradInt, VthetaInt, dt, selfgravityupdate);
    }

  ComputeViscousTerms (VradInt, VthetaInt, Dens, 0);
  UpdateVelocitiesWithViscosity(VradInt, VthetaInt, Dens, dt);

  //if (!Evanescent) ApplySubKeplerianBoundary(VthetaInt);

}

__host__ void Substep2 (float dt)
{
  Substep2Kernel<<<dimGrid2, dimBlock2>>>(Dens_d, VradInt_d, VthetaInt_d, TemperInt_d, NRAD, NSEC, CVNR, invdiffRmed_d,
  invdiffRsup_d, DensInt_d, Adiabaticc, Rmed_d, dt, VradNew_d, VthetaNew_d, energy_d, energyInt_d);
  gpuErrchk(cudaDeviceSynchronize());
}

__host__ void Substep3 (float *Dens, float dt)
{
  for (int i = 0; i < NRAD; i++) viscosity_array[i] = FViscosity(Rmed[i]);
  gpuErrchk(cudaMemcpy(viscosity_array_d, viscosity_array, (NRAD+1)*sizeof(float), cudaMemcpyHostToDevice));

  Substep3Kernel<<<dimGrid2, dimBlock2>>>(Dens_d, Qplus_d, viscosity_array_d, TAURR_d, TAURP_d , TAUPP_d, DivergenceVelocity_d,
     NRAD, NSEC, Rmed_d, Cooling, energyNew_d, dt, EnergyMed_d, SigmaMed_d, CoolingTimeMed_d, energy_d,
     ADIABATICINDEX, QplusMed_d);
  gpuErrchk(cudaDeviceSynchronize());
}

__host__ void Computecudamalloc (float *energy)
{

  CoolingTimeMed  = (float *)malloc(size_grid*sizeof(float));
  QplusMed        = (float *)malloc(size_grid*sizeof(float));
  viscosity_array = (float *)malloc((NRAD+1)*sizeof(float));

  gpuErrchk(cudaMalloc((void**)&TemperInt_d,   size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&Temperature_d, size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&Pressure_d,    size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&SoundSpeed_d,  size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&DensStar_d,     size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&VradInt_d,     size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&AspectRatioRmed_d, NRAD*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&DensInt_d,        size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&VradNew_d,        size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&VthetaNew_d,      size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&energyInt_d,      size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&Potential_d,            size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&VthetaInt_d,      size_grid*sizeof(float)));


  gpuErrchk(cudaMalloc((void**)&SigmaInf_d,        NRAD*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&vt_cent_d,         (NRAD+1)*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&energy_d,       size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&viscosity_array_d, (NRAD+1)*sizeof(float)));

  gpuErrchk(cudaMemcpy(SigmaInf_d, SigmaInf,               NRAD*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(energy_d, energy,             size_grid*sizeof(float), cudaMemcpyHostToDevice));

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
  Make1Dprofile(SG_Accr);

  Azimutalvelocity_withSGKernel<<<dimGrid2, dimBlock2>>>(Vtheta_d, Rmed_d, FLARINGINDEX, SIGMASLOPE, ASPECTRATIO, G, \
    GLOBAL_bufarray_d, NRAD, NSEC);
  gpuErrchk(cudaDeviceSynchronize());
}

__host__ bool DetectCrash (float *array)
{
  bool Crash = false;
  float numCrush;

  gpuErrchk(cudaMemcpy(array_d, array, size_grid*sizeof(float), cudaMemcpyHostToDevice));
  CrashKernel<<<dimGrid2, dimBlock2>>>(array_d, NRAD, NSEC, Crash);
  gpuErrchk(cudaDeviceSynchronize());

  numCrush = DeviceReduce(array_d, size_grid);
  if (numCrush > 0.0) Crash = true;
  return Crash;
}

__host__ void ComputePressureField (float *Dens, float *energy)
{
  ComputePressureFieldKernel<<<dimGrid2, dimBlock2>>>(SoundSpeed_d, Dens_d, Pressure_d, Adiabaticc, NRAD,
    NSEC, ADIABATICINDEX, energy_d);
  gpuErrchk(cudaDeviceSynchronize());
}

__host__ void ComputeSoundSpeed ()
{
  ComputeSoundSpeedKernel<<<dimGrid2, dimBlock2>>>(SoundSpeed_d, Dens_d, Rmed_d, energy_d, NSEC, NRAD,
    Adiabaticc, ADIABATICINDEX, FLARINGINDEX, AspectRatioRmed_d);
  gpuErrchk(cudaDeviceSynchronize());
}

__host__ void ComputeTemperatureField ()
{

  ComputeTemperatureFieldKernel<<<dimGrid2, dimBlock2>>>(Dens_d, Temperature_d, Pressure_d, energy_d, MU, R,
    ADIABATICINDEX, Adiabaticc, NSEC, NRAD);
  gpuErrchk(cudaDeviceSynchronize());
}

__host__ void InitVelocities (float *Vrad, float *Vtheta)
{

  gpuErrchk(cudaMemcpy(viscosity_array_d, viscosity_array, (NRAD+1)*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(vt_cent_d, vt_cent,     (NRAD+1)*sizeof(float), cudaMemcpyHostToDevice));

  InitGasVelocitiesKernel<<<dimGrid2, dimBlock2>>>(viscosity_array_d, NSEC, NRAD, SelfGravity, Rmed_d,
  G, ASPECTRATIO, FLARINGINDEX, SIGMASLOPE, CentrifugalBalance, Vrad_d, Vtheta_d, ViscosityAlpha,
  IMPOSEDDISKDRIFT, SIGMA0, SigmaInf_d, OmegaFrame1, Rinf_d, vt_cent_d);

  gpuErrchk(cudaDeviceSynchronize());
  gpuErrchk(cudaMemcpy(Vrad, Vrad_d,     size_grid*sizeof(float), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(Vtheta, Vtheta_d, size_grid*sizeof(float), cudaMemcpyDeviceToHost));

  //for (int i = 0; i < NSEC; i++) vrad[i] = vrad[i+NSEC*NRAD] = 0.0;

  gpuErrchk(cudaMemcpy(Vrad_d, Vrad,     size_grid*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(Vtheta_d, Vtheta, size_grid*sizeof(float), cudaMemcpyHostToDevice));
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

__host__ void ActualiseGasEnergy (float *energy, float *energyNew)
{
  gpuErrchk(cudaMemcpy(energy_d, energyNew_d, size_grid*sizeof(float), cudaMemcpyDeviceToDevice));
  gpuErrchk(cudaDeviceSynchronize());
}

__host__ void Substep1cudamalloc (float *Vrad, float *Vtheta)
{
  //gpuErrchk(cudaMemcpy(VradInt_d, VradInt,               size_grid*sizeof(float), cudaMemcpyHostToDevice));
  //gpuErrchk(cudaMemcpy(Potential_d, Potential,           size_grid*sizeof(float), cudaMemcpyHostToDevice));
  //gpuErrchk(cudaMemcpy(VthetaInt_d, VthetaInt,           size_grid*sizeof(float), cudaMemcpyHostToDevice));
  //gpuErrchk(cudaMemcpy(Vrad_d, Vrad,                     size_grid*sizeof(float), cudaMemcpyHostToDevice));
  //gpuErrchk(cudaMemcpy(Vtheta_d, Vtheta,                 size_grid*sizeof(float), cudaMemcpyHostToDevice));
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
