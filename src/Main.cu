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

extern float *SigmaInf_d, *AspectRatioRmed_d, *cs0_d, *cs1_d, *csnrm1_d, *csnrm2_d, *mean_dens_d, *mean_dens_d2, *vrad_d;
extern float *mean_energy_d, *mean_energy_d2, *cs0, *cs1, *csnrm1, *csnrm2, *mean_dens, *mean_dens2, *mean_energy;
extern float OMEGAFRAME, OmegaFrame1, *press, *CellAbscissa, *CellOrdinate, HillRadius, PhysicalTimeInitial, PhysicalTime;
extern float *Rmed_d, CVNR, *CellAbscissa_d, *CellOrdinate_d, *dens_d, *temperature_d, *gas_label_d, *energy_d, *mean_energy2;
extern float *temperature, *vtheta_d, *press_d, *Rinf_d, *cosns_d, *sinns_d, *SoundSpeed_d, *viscosity_array_d;
extern float *forcesxi_d, *forcesyi_d, *forcesxo_d, *forcesyo_d, *ThetaMomP_d, *ThetaMomM_d, *RadMomP_d, *RadMomM_d;
float *Rinf, *Rmed, *Rsup, *Surf, *invRinf, *invSurf, *invdiffSurf, *invdiffRsup, *invdiffRmed, *invRmed, *Radii;
float *SigmaMed, *SigmaInf, *EnergyMed, *cosns, *sinns, mdcp, exces_mdcp = 0.0, ScalingFactor = 1.0;
float *forcesxi, *forcesyi, *forcesxo, *forcesyo, *vradint, *pot, *vrad, *vthetaint, *vtheta, *powRmed, *densint_d;
float *temperatureint, *densint, *vradnew, *vthetanew, *energyint, *fieldsrc, *vt_int, *GLOBAL_bufarray, *Surf_d;
float *vradint_d, *pot_d, *vthetaint_d, *invdiffRmed_d, *invRinf_d, *powRmed_d, *vthetanew_d, *vradnew_d;
float *temperatureint_d, *energyint_d, *invdiffRsup_d, *CoolingTimeMed, *QplusMed , *viscosity_array;
float *Drr, *Drr_d, *Dpp, *Dpp_d, *Drp, *Drp_d, *divergence, *divergence_d, *Trr, *Trr_d, *Tpp, *Tpp_d, *Trp , *Trp_d;
float *qplus_d, *energynew_d, *EnergyMed_d, *SigmaMed_d, *CoolingTimeMed_d, *QplusMed_d, *energynew, *qplus;
float *RadMomP, *RadMomM, *ThetaMomP, *ThetaMomM;

extern int NRAD, NSEC, SelfGravity, Corotating, FREQUENCY, Adiabaticc, IsDisk, Cooling;
static int StillWriteOneOutput, InnerOutputCounter=0;
int nrad2pot, nsec2pot, blocksize = 32, size_grid, dimfxy=11, TimeStep = 0, NbRestart = 0, blocksize2 = 1024;
bool ZMPlus = false, verbose = false, Restart = false, TimeToWrite;

__host__ int main(int argc, char *argv[])
{

  bool disable = false, TimeInfo = false, Profiling = false, Stockholm = false;
  char ParameterFile[256];

  PlanetarySystem *sys;
  Force *force;

  float *gas_label, *dens, *energy, xpl, ypl;

  if (argc == 1) PrintUsage (argv[0]);

  strcpy (ParameterFile, "");
  for (int i = 1; i < argc; i++) {
    if (*(argv[i]) == '-') {
      if (strspn (argv[i], "-secndovtpfamzib0123456789") != strlen (argv[i]))
	PrintUsage (argv[0]);
      if (strchr (argv[i], 'n'))
	disable = true;
      if (strchr (argv[i], 'e'))
	Stockholm = true;
      if (strchr (argv[i], 'v'))
	verbose = true;
      if (strchr (argv[i], 't'))
	TimeInfo = true;
      if (strchr (argv[i], 'c'))
	SloppyCFL = true;
      if (strchr (argv[i], 'p'))
	Profiling = true;
      if (strchr (argv[i], 'd'))
	debug = true;
      if (strchr (argv[i], 'b'))
	CentrifugalBalance = true;
      if (strchr (argv[i], 'm'))
	Merge = true;
      if (strchr (argv[i], 'a'))
	MonitorIntegral = true;
      if (strchr (argv[i], 'z'))
	FakeSequential = true;
      if (strchr (argv[i], 'i')) {
	StoreSigma = true;
	if (Adiabaticc)
	  StoreEnergy = true;
      }
      if (strchr (argv[i], '0'))
	OnlyInit = true;
      if ((argv[i][1] >= '1') && (argv[i][1] <= '9')) {
	GotoNextOutput = true;
	StillWriteOneOutput = (int)(argv[i][1]-'0');
      }
      if (strchr (argv[i], 's')) {
	Restart = true;
	i++;
	NbRestart = atoi(argv[i]);
	if ((NbRestart < 0)) {
	  printf ("Incorrect restart number\n");
	  PrintUsage (argv[0]);
	}
      }
      if (strchr (argv[i], 'o')) {
	OverridesOutputdir = true;
	i++;
	sprintf (NewOutputdir, "%s", argv[i]);
      } else {
	if (strchr (argv[i], 'f')) {
	  i++;
	  ScalingFactor = atof(argv[i]);
	  printf ("Scaling factor = %g\n", ScalingFactor);
	  if ((ScalingFactor <= 0)) {
	    printf ("Incorrect scaling factor\n");
	    PrintUsage (argv[0]);
	  }
	}
      }
    }
    else strcpy (ParameterFile, argv[i]);
  }

  if ( (StoreSigma || StoreEnergy) && !(Restart)) {
    printf ("You cannot use tabulated surface density\n");
    printf ("or surface internal energy in a non-restart run.\n");
    printf ("Aborted\n");
    exit (0);
  }
  if (ParameterFile[0] == 0) PrintUsage (argv[0]);


  ReadFile(ParameterFile);
  size_grid = (NRAD+1)*NSEC;

  if(!IsPow2(NRAD+1)) nrad2pot = NearestPowerOf2(NRAD+1);
  if(!IsPow2(NSEC)) nsec2pot = NearestPowerOf2(NSEC);

  if (verbose == YES) TellEverything();
  if (disable == YES) exit(0);
  printf("Allocating arrays...\n");

  dens = (float *) malloc(sizeof(float)*(size_grid));
  energy = (float *) malloc(sizeof(float)*size_grid);
  gas_label = (float *) malloc(sizeof(float)*size_grid);
  EnergyMed = (float *) malloc(sizeof(float)*NRAD);
  SigmaMed = (float *) malloc(sizeof(float)*NRAD);
  SigmaInf = (float *) malloc(sizeof(float)*NRAD);
  vt_int = (float *) malloc(sizeof(float)*NRAD);
  GLOBAL_bufarray = (float *)malloc(sizeof(float)*NRAD);
  vrad = (float *) malloc(sizeof(float)*size_grid);
  vtheta = (float *) malloc(sizeof(float)*size_grid);
  vradint = (float *) malloc(sizeof(float)*size_grid);
  pot = (float *) malloc(sizeof(float)*size_grid);
  vthetaint = (float *) malloc(sizeof(float)*size_grid);
  powRmed = (float *) malloc(sizeof(float)*NRAD);
  temperatureint = (float *)malloc(sizeof(float)*size_grid);
  densint = (float *)malloc(sizeof(float)*size_grid);
  vradnew = (float *)malloc(sizeof(float)*size_grid);
  vthetanew = (float *)malloc(sizeof(float)*size_grid);
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
  if ( Adiabaticc ) {
      InitGasEnergy (energy);
  }

  if ( SelfGravity ) {
    /* If SelfGravity = YES or Z, planets are initialized feeling disk
       potential. Only the surface density is required to calculate
       the radial self-gravity acceleration. The disk radial and
       azimutal velocities are not updated */

    //compute_selfgravity (gas_density, gas_v_rad, gas_v_theta, foostep, updatevelocities);
    //init_planetarysys_withSG (sys);
  }
  ListPlanets (sys);
  OmegaFrame1 = OMEGAFRAME;

  if (Corotating) OmegaFrame1 = GetPsysInfo (sys, FREQUENCY);
  /* Only gas velocities remain to be initialized */


  Initialization (dens, energy, vrad, vtheta, gas_label, sys);

  xpl = sys->x[0];
  ypl = sys->y[0];

  // aca voy

  mdcp = CircumPlanetaryMasshost(dens, xpl, ypl);

  float dt = 0.999;

  EmptyPlanetSystemFile (sys);
  PhysicalTimeInitial = PhysicalTime;
  MultiplyPolarGridbyConstanthost(dens);

  for (int i  = 0; i < size_grid; i++) {
    pot[i] = 0.00001*i;
  }

 for (int i = 0; i < NRAD; i++) {
   powRmed[i] = powf(Rmed[i],-2.5+SIGMASLOPE);
 }


  for (int i = 0; i <= NTOT; i++) {
    InnerOutputCounter++;

    if (InnerOutputCounter == 1) {
      InnerOutputCounter = 0;
      WriteBigPlanetSystemFile (sys, TimeStep);
      UpdateLog(force, dens, sys, TimeStep, PhysicalTime, dimfxy, i); // aca falta hacer computeForce
    }

    if (NINTERM * (TimeStep = (i / NINTERM)) == i)
    {
      /* Outputs are done here */
      TimeToWrite = YES;

      DeviceToHostcudaMemcpy(dens, energy, gas_label, temperature, vrad, vtheta); // Traigo los valores desde la GPU
      SendOutput (TimeStep, dens, vrad, vtheta, energy, gas_label);
      WritePlanetSystemFile (sys, TimeStep);

    }
    else TimeToWrite = NO;

    // algogas

    if (Adiabaticc)
    {
      ComputeSoundSpeedhost(dens, energy);
      /* it is necesary to update computation of soundspeed if one uses
        alphaviscosity in Fviscosity. It is not necesary in locally
        isothermal runs since cs is constant. It is computed here for
        the needs of ConditionCFL. */
    }
    if (IsDisk == YES)
    {

    }

    Substep1host(dens, vrad, vtheta, dt, i); // aca voy
    Substep2host(dens, energy, dt, i);
    ActualiseGasVrad(vrad, vradnew);
    ActualiseGasVtheta(vtheta, vthetanew);
    ApplyBoundaryCondition (dens, energy, vrad, vtheta, dt, i);

    if (Adiabaticc)
    {
      ComputeViscousTerms (vrad, vtheta, dens, 1, 1);
      Substep3host(dens, dt, i);
      ActualiseGasEnergy (energy, energynew);
    }
    Transport (dens, vrad, vtheta, energy, gas_label, dt, i);

    if (NINTERM * TimeStep == i) printf("step = %d\n",TimeStep );

  }

  FreePlanetary (sys);
  FreeForce (force);


  // FILE *f;
  // f = fopen("datos.txt","w");
  //
  // for (int i = 0; i < NRAD*NSEC; i++)
  // {
  //   fprintf(f, "%f\n",dens[i] );
  // }
  //
  // fclose(f);


  FreeArrays(dens, energy, gas_label);
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
  cudaFree(gas_label_d);
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
}

__host__ void FreeArrays(float *dens, float *energy, float *gas_label)
{
  free(dens);
  free(energy);
  free(gas_label);
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

}

__host__ void DeviceToHostcudaMemcpy(float *dens, float *energy, float *gas_label, float *temperature, float *vrad, float *vtheta)
{
  gpuErrchk(cudaMemcpy(dens, dens_d, size_grid*sizeof(float), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(energy, energy_d, size_grid*sizeof(float), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(temperature, temperature_d, size_grid*sizeof(float), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(gas_label, gas_label_d, size_grid*sizeof(float), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(vrad, vrad_d, size_grid*sizeof(float), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(vtheta, vtheta_d, size_grid*sizeof(float), cudaMemcpyDeviceToHost));

}

__host__ void Substep1cudamalloc()
{
  gpuErrchk(cudaMalloc((void**)&vradint_d, size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&pot_d, size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&vthetaint_d, size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&invdiffRmed_d, NRAD*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&invRinf_d,NRAD*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&powRmed_d,NRAD*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&invdiffRsup_d, NRAD*sizeof(float)));

  gpuErrchk(cudaMemcpy(vradint_d, vradint, size_grid*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(pot_d, pot, size_grid*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(vrad_d, vrad, size_grid*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(vthetaint_d, vthetaint, size_grid*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(vtheta_d, vtheta, size_grid*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(invdiffRmed_d, invdiffRmed, NRAD*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(invRinf_d, invRinf, NRAD*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(powRmed_d, powRmed, NRAD*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(invdiffRsup_d, invdiffRsup, NRAD*sizeof(float), cudaMemcpyHostToDevice));
}

__host__ void Substep2cudamalloc()
{

  gpuErrchk(cudaMalloc((void**)&temperatureint_d,size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&densint_d,size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&vradnew_d,size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&vthetanew_d,size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&energyint_d, size_grid*sizeof(float)));

  gpuErrchk(cudaMemcpy(temperatureint_d, temperatureint, size_grid*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(densint_d, densint, size_grid*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(vradnew_d, vradnew, size_grid*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(vthetanew_d, vthetanew, size_grid*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(energyint_d, energyint, size_grid*sizeof(float), cudaMemcpyHostToDevice));

}

__host__ void Substep3cudamalloc()
{
  gpuErrchk(cudaMalloc((void**)&qplus_d,size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&energynew_d,size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&EnergyMed_d,NRAD*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&SigmaMed_d,NRAD*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&CoolingTimeMed_d,NRAD*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&QplusMed_d,NRAD*sizeof(float)));


  gpuErrchk(cudaMemcpy(qplus_d, qplus, size_grid*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(energynew_d, energynew, size_grid*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(EnergyMed_d, EnergyMed, NRAD*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(SigmaMed_d, SigmaMed, NRAD*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(CoolingTimeMed_d, CoolingTimeMed, NRAD*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(QplusMed_d, QplusMed, NRAD*sizeof(float), cudaMemcpyHostToDevice));

}

__host__ float CircumPlanetaryMasshost(float *dens, float xpl, float ypl)
{

  dim3 dimGrid( nsec2pot/blocksize, nrad2pot/blocksize );
  dim3 dimBlock( blocksize, blocksize );
  float *mdcp0, *mdcp0_d;

  mdcp0 = (float *)malloc(size_grid*sizeof(float));

  gpuErrchk(cudaMalloc((void**)&Surf_d,NRAD*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&mdcp0_d,size_grid*sizeof(float)));

  gpuErrchk(cudaMemcpy(Surf_d, Surf, NRAD*sizeof(float), cudaMemcpyHostToDevice ));
  gpuErrchk(cudaMemcpy(mdcp0_d, mdcp0, size_grid*sizeof(float), cudaMemcpyHostToDevice));

  CircumPlanetaryMass<<<dimGrid, dimBlock>>> (dens_d, Surf_d, CellAbscissa_d, CellOrdinate_d, xpl, ypl, NRAD, NSEC, HillRadius, mdcp0_d);
  gpuErrchk(cudaDeviceSynchronize());

  // reduction mdcp
  mdcp = DeviceReduce(mdcp0_d, size_grid);
  cudaFree(mdcp0_d);
  free(mdcp0);

  return mdcp;
}

__host__ void MultiplyPolarGridbyConstanthost(float *dens)
{
  dim3 dimGrid( nsec2pot/blocksize, nrad2pot/blocksize );
  dim3 dimBlock( blocksize, blocksize );

  MultiplyPolarGridbyConstant<<<dimGrid, dimBlock>>>(dens_d, NRAD, NSEC, ScalingFactor);
  gpuErrchk(cudaDeviceSynchronize());
}

__host__ void Substep1host(float *dens, float *vrad, float *vtheta, float dt, int i)
{
  dim3 dimGrid( nsec2pot/blocksize, nrad2pot/blocksize );
  dim3 dimBlock( blocksize, blocksize );

  if(i == 0) Substep1cudamalloc();

  Substep1<<<dimGrid, dimBlock>>>(press_d, dens_d, vradint_d, invdiffRmed_d,pot_d,Rinf_d,
    invRinf_d, vrad_d, vthetaint_d, vtheta_d, Rmed_d,  dt, NRAD, NSEC, OmegaFrame1, ZMPlus,
    IMPOSEDDISKDRIFT, SIGMASLOPE, powRmed_d);
  gpuErrchk(cudaDeviceSynchronize());


    /*if (SelfGravity){
      selfgravityupdate = YES;
      compute_selfgravity(Rho, VradInt, VthetaInt, dt, selfgravityupdate);
    }*/

  ComputeViscousTerms (vradint, vthetaint, dens, i, 0);
  //UpdateVelocitiesWithViscosity(vradint, vthetaint, dens, dt);

  //if (!Evanescent) ApplySubKeplerianBoundary(VthetaInt);


}

__host__ void Substep2host(float *dens, float *energy, float dt, int i)
{
  dim3 dimGrid( nsec2pot/blocksize, nrad2pot/blocksize );
  dim3 dimBlock( blocksize, blocksize );

  if (i == 0) Substep2cudamalloc();

  Substep2<<<dimGrid, dimBlock>>>(dens_d, vradint_d, vthetaint_d, temperatureint_d, NRAD, NSEC, CVNR, invdiffRmed_d,
  invdiffRsup_d, densint_d, Adiabaticc, Rmed_d, dt, vradnew_d, vthetanew_d, energy_d, energyint_d);

  gpuErrchk(cudaDeviceSynchronize());
  //gpuErrchk(cudaMemcpy(densint, densint_d, size_grid*sizeof(float), cudaMemcpyDeviceToHost));
  //gpuErrchk(cudaMemcpy(temperatureint, temperatureint_d, size_grid*sizeof(float), cudaMemcpyDeviceToHost));
  //gpuErrchk(cudaMemcpy(vthetanew, vthetanew_d, size_grid*sizeof(float), cudaMemcpyDeviceToHost));
  //gpuErrchk(cudaMemcpy(vradnew, vradnew_d, size_grid*sizeof(float), cudaMemcpyDeviceToHost));
  //gpuErrchk(cudaMemcpy(energyint, energyint_d, size_grid*sizeof(float), cudaMemcpyDeviceToHost));
}

__host__ void Substep3host(float *dens, float dt, int p)
{

  dim3 dimGrid( nsec2pot/blocksize, nrad2pot/blocksize );
  dim3 dimBlock( blocksize, blocksize );

  for (int i = 0; i < NRAD; i++) viscosity_array[i] = FViscosity(Rmed[i]);
  gpuErrchk(cudaMemcpy(viscosity_array_d, viscosity_array, (NRAD+1)*sizeof(float), cudaMemcpyHostToDevice));

  if (p == 0) Substep3cudamalloc();

  Substep3<<<dimGrid, dimBlock>>>(dens_d, qplus_d, viscosity_array_d, Trr_d, Trp_d , Tpp_d, divergence_d,
     NRAD, NSEC, Rmed_d, Cooling, energynew_d, dt, EnergyMed_d, SigmaMed_d, CoolingTimeMed_d, energy_d,
     ADIABATICINDEX, QplusMed_d);
  gpuErrchk(cudaDeviceSynchronize());

}

__host__ void ActualiseGasVtheta(float *vtheta, float *vthetanew)
{
  gpuErrchk(cudaMemcpy(vtheta_d, vthetanew_d, size_grid*sizeof(float), cudaMemcpyDeviceToDevice));
  gpuErrchk(cudaDeviceSynchronize());

}

__host__ void ActualiseGasVrad(float *vrad, float *vradnew)
{

  gpuErrchk(cudaMemcpy(vrad_d, vradnew_d, size_grid*sizeof(float), cudaMemcpyDeviceToDevice));
  gpuErrchk(cudaDeviceSynchronize());

}

__host__ void ActualiseGasEnergy(float *energy, float *energynew)
{

  gpuErrchk(cudaMemcpy(energy_d, energynew_d, size_grid*sizeof(float), cudaMemcpyDeviceToDevice));
  gpuErrchk(cudaDeviceSynchronize());

}
