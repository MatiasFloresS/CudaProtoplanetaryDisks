// nvcc hello-world.cu -L /usr/local/cuda/lib -lcudart -o hello-world

#include "main.cuh"
#include "kernels.cuh"
//#include "readfiles.cuh"
#include "fondam.cuh"
#include "SourceEuler.cuh"
#include "Psys.cuh"
#include "Pframeforce.cuh"

using namespace std;

extern int NRAD, NSEC;
float *Rinf, *Rmed, *Rsup, *Surf, *invRinf, *invSurf, *invdiffSurf;
float *invdiffRsup, *invdiffRmed, *invRmed, *Radii;
float *SigmaMed, *SigmaInf, *dens, *EnergyMed, *energy;
extern int SelfGravity;
extern float OMEGAFRAME;
extern float OmegaFrame1;
int blocksize = 32;
int size_grid;
float OmegaFrame = 0.12871;
bool ZMPlus = false, verbose = false, Restart = false;
extern int Adiabaticc;
static int StillWriteOneOutput;
int NbRestart = 0;

float ScalingFactor = 1.0;

__host__ long NearestPowerOf2(long n)
{
  if(!n) return n; //(0 ==2^0)

  int x=1;
  while (x < n)
  {
    x<<=1;
  }
  return x;
}

__host__ bool isPow2(unsigned int x)
{
  return ((x&(x-1)==0));
}


__host__ int main(int argc, char *argv[])
{

  bool disable = false, TimeInfo = false, Profiling = false;
  bool Stockholm = false;
  char ParameterFile[256];

  PlanetarySystem *sys;

  float *gas_v_rad, *gas_v_theta, *gas_label;
  float *press, *rho, *vradint, *invdiffRmed, *pot, *invRinf, *vrad, *vthetaint, *vtheta;
  float *press_d,*rho_d,*vradint_d,*invdiffRmed_d,*pot_d, *invRinf_d, *Rinf_d, *vrad_d, *vthetaint_d, *vtheta_d, *Rmed_d;
  int nrad2pot, nsec2pot;


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
  size_grid = NRAD*NSEC;
  if (verbose == YES) TellEverything();
  if (disable == YES) exit(0);
  printf("Allocating arrays...\n");
  fflush(stdout);

  dens = (float *) malloc(sizeof(float)*size_grid);
  energy = (float *) malloc(sizeof(float)*size_grid);

  gas_v_rad = (float *) malloc(sizeof(float)*size_grid);
  gas_v_theta = (float *) malloc(sizeof(float)*size_grid);
  gas_label = (float *) malloc(sizeof(float)*size_grid);

  EnergyMed = (float *) malloc(sizeof(float)*(NRAD+1));
  SigmaMed = (float *) malloc(sizeof(float)*(NRAD+1));
  SigmaInf = (float *) malloc(sizeof(float)*(NRAD+1));
  printf("done.\n");

  FillPolar1DArray();

  // force = AllocateForce (dimfxy);

  char configplanet[100];
  strncpy(configplanet, PLANETCONFIG.c_str(), sizeof(configplanet));
  configplanet[sizeof(configplanet)-1]=0;

  /* Here planets are initialized feeling star potential but they do
     not feel disk potential  */

  sys = InitPlanetarySystem(configplanet);

  /* Gas density initialization */
  InitGasDensity ();

  /* If energy equation is taken into account, we initialize the gas
     thermal energy  */
  if ( Adiabaticc ) {
      InitGasEnergy ();
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
  float dt = 0.999;



  press = (float *) malloc(sizeof(float)*size_grid);
  rho = (float *) malloc(sizeof(float)*size_grid );
  vradint = (float *) malloc(sizeof(float)*size_grid);
  pot = (float *) malloc(sizeof(float)*size_grid);
  vrad = (float *) malloc(sizeof(float)*size_grid);
  vthetaint = (float *) malloc(sizeof(float)*size_grid);
  vtheta = (float *) malloc(sizeof(float)*size_grid);

  for (int i  = 0; i < size_grid; i++) {
    press[i] = i;
    rho[i] = 2*i;
    pot[i] = 0.001*i;
    vrad[i] = 0.212*i;
    vtheta[i] = 0.1;
  }


  if(!isPow2(NRAD)) nrad2pot = NearestPowerOf2(NRAD);
  if(!isPow2(NSEC)) nsec2pot = NearestPowerOf2(NSEC);

	cudaMalloc((void**)&press_d, size_grid*sizeof(float));
	cudaMalloc((void**)&rho_d, size_grid*sizeof(float) );
  cudaMalloc((void**)&vradint_d, size_grid*sizeof(float));
  cudaMalloc((void**)&pot_d, size_grid*sizeof(float));
  cudaMalloc((void**)&vrad_d, size_grid*sizeof(float));
  cudaMalloc((void**)&vthetaint_d, size_grid*sizeof(float));
  cudaMalloc((void**)&vtheta_d, size_grid*sizeof(float));
  cudaMalloc((void**)&invdiffRmed_d, NRAD*sizeof(float));
  cudaMalloc((void**)&invRinf_d,NRAD*sizeof(float));
  cudaMalloc((void**)&Rinf_d,NRAD*sizeof(float));
  cudaMalloc((void**)&Rmed_d,NRAD*sizeof(float));

	cudaMemcpy(press_d, press, size_grid*sizeof(float), cudaMemcpyHostToDevice );
	cudaMemcpy(rho_d, rho, size_grid*sizeof(float), cudaMemcpyHostToDevice );
  cudaMemcpy(vradint_d, vradint, size_grid*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(pot_d, pot, size_grid*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(vrad_d, vrad, size_grid*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(vthetaint_d, vthetaint, size_grid*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(vtheta_d, vtheta, size_grid*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(invdiffRmed_d, invdiffRmed, (NRAD+1)*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(invRinf_d, invRinf, (NRAD+1)*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(Rinf_d, Rinf, (NRAD+1)*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(Rmed_d, Rmed, (NRAD+1)*sizeof(float), cudaMemcpyHostToDevice);

	dim3 dimGrid( nsec2pot/blocksize, nrad2pot/blocksize );
	dim3 dimBlock( blocksize, blocksize );

	substep1<<<dimGrid, dimBlock>>>(press_d, rho_d, vradint_d, invdiffRmed_d,pot_d,Rinf_d,
    invRinf_d, vrad_d, vthetaint_d, vtheta_d, Rmed_d,  dt, NRAD, NSEC, OmegaFrame, ZMPlus,
    IMPOSEDDISKDRIFT, SIGMASLOPE);

	cudaMemcpy(vradint, vradint_d, size_grid*sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(vthetaint, vthetaint_d, size_grid*sizeof(float), cudaMemcpyDeviceToHost);


	cudaFree(press_d );
	cudaFree(rho_d );
  cudaFree(vradint_d);
  cudaFree(pot_d);
  cudaFree(invdiffRmed_d);
  cudaFree(invRinf_d);
  cudaFree(Rinf_d);
  cudaFree(vrad_d);
  cudaFree(vthetaint_d);
  cudaFree(vtheta_d);
  cudaFree(Rmed_d);

  /*if (SelfGravity){
    selfgravityupdate = YES;
    compute_selfgravity(Rho, VradInt, VthetaInt, dt, selfgravityupdate);
  }
  ComputeViscousTerms (VradInt, VthetaInt, Rho);*/
  UpdateVelocitiesWithViscosity(vradint, vthetaint, rho, dt);
/*
  if (!Evanescent) ApplySubKeplerianBoundary(VthetaInt);
*/
  FILE *f;
  f = fopen("datos.txt","w");

  for (int i = 0; i < size_grid; i++)
  {
    fprintf(f, "%f\n",vthetaint[i] );
  }

  fclose(f);
	return EXIT_SUCCESS;
}
