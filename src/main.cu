// nvcc hello-world.cu -L /usr/local/cuda/lib -lcudart -o hello-world

#include "main.cuh"
#include "kernels.cuh"
#include "fondam.cuh"
#include "SourceEuler.cuh"
#include "Psys.cuh"
#include "Pframeforce.cuh"
#include "Init.cuh"
#include "cuda_runtime.h"
#include "Output.cuh"
#include "Force.cuh"
#include "SideEuler.cuh"

using namespace std;

extern int NRAD, NSEC;
float *Rinf, *Rmed, *Rsup, *Surf, *invRinf, *invSurf, *invdiffSurf;
float *invdiffRsup, *invdiffRmed, *invRmed, *Radii;
float *SigmaMed, *SigmaInf, *dens, *EnergyMed, *energy;
extern int SelfGravity, Corotating, FREQUENCY;
extern float OMEGAFRAME, OmegaFrame1;
float *cosns, *sinns;
int nrad2pot, nsec2pot;
extern float *press;
extern float *CellAbscissa, *CellOrdinate;
extern float HillRadius;
extern float PhysicalTimeInitial, PhysicalTime, CVNR;
int blocksize = 32;
int size_grid;
float OmegaFrame = 0.12871;
bool ZMPlus = false, verbose = false, Restart = false;
extern int Adiabaticc;
static int StillWriteOneOutput;
int NbRestart = 0;
float mdcp;
float exces_mdcp = 0.0;
float ScalingFactor = 1.0;
int dimfxy=11;

int TimeStep = 0;
int static InnerOutputCounter=0;


__host__ int main(int argc, char *argv[])
{

  bool disable = false, TimeInfo = false, Profiling = false;
  bool Stockholm = false;
  char ParameterFile[256];

  PlanetarySystem *sys;
  Force *force;

  float *gas_v_rad, *gas_v_theta, *gas_label;
  float *rho, *vradint, *pot, *vrad, *vthetaint, *vtheta;
  float *press_d,*dens_d,*vradint_d,*invdiffRmed_d,*pot_d, *invRinf_d, *Rinf_d, *vrad_d, *vthetaint_d, *vtheta_d, *Rmed_d;
  float *powRmed, *powRmed_d, *temperatureint, *temperatureint_d, *invdiffRsup_d, *densint, *densint_d, *energyint_d;
  float *vradnew_d, *vthetanew_d, *energy_d, *vthetanew, *vradnew, *energyint;

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
  if(!isPow2(NRAD)) nrad2pot = NearestPowerOf2(NRAD);
  if(!isPow2(NSEC)) nsec2pot = NearestPowerOf2(NSEC);

  dim3 dimGrid( nsec2pot/blocksize, nrad2pot/blocksize );
  dim3 dimBlock( blocksize, blocksize );

  if (verbose == YES) TellEverything();
  if (disable == YES) exit(0);
  printf("Allocating arrays...\n");
  fflush(stdout);

  dens = (float *) malloc(sizeof(float)*size_grid);
  energy = (float *) malloc(sizeof(float)*size_grid);

  gas_v_rad = (float *) malloc(sizeof(float)*size_grid);
  gas_v_theta = (float *) malloc(sizeof(float)*size_grid);
  gas_label = (float *) malloc(sizeof(float)*size_grid);

  EnergyMed = (float *) malloc(sizeof(float)*(NRAD));
  SigmaMed = (float *) malloc(sizeof(float)*(NRAD));
  SigmaInf = (float *) malloc(sizeof(float)*(NRAD));
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

  if (Corotating) OmegaFrame1 = GetPsysInfo (sys, FREQUENCY);
  /* Only gas velocities remain to be initialized */

  Initialization (dens, gas_v_rad, gas_v_theta, energy, gas_label, sys);

  float *mdcp0, mdcp;
  float *Surf_d, *CellAbscissa_d, *CellOrdinate_d;
  float xpl, ypl;
  xpl = sys->x[0];
  ypl = sys->y[0];
  /* <-------------------------     CircumPlanetaryMass()       --------------------------> */

  mdcp0 = (float *)malloc(size_grid*sizeof(float));
  float *mdcp0_d;
  gpuErrchk(cudaMalloc((void**)&Surf_d,NRAD*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&dens_d,size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&CellAbscissa_d,size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&CellOrdinate_d,size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&mdcp0_d,size_grid*sizeof(float)));

  // gpuErrchk(cudaMalloc(&fieldsrc_d, ((NRAD+1)*NSEC)*sizeof(float)));


  gpuErrchk(cudaMemcpy(Surf_d, Surf, NRAD*sizeof(float), cudaMemcpyHostToDevice ));
  gpuErrchk(cudaMemcpy(dens_d, dens, size_grid*sizeof(float), cudaMemcpyHostToDevice ));
  gpuErrchk(cudaMemcpy(CellAbscissa_d, CellAbscissa, size_grid*sizeof(float), cudaMemcpyHostToDevice ));
  gpuErrchk(cudaMemcpy(CellOrdinate_d, CellOrdinate, size_grid*sizeof(float), cudaMemcpyHostToDevice ));
  gpuErrchk(cudaMemcpy(mdcp0_d, mdcp0, size_grid*sizeof(float), cudaMemcpyHostToDevice));

  CircumPlanetaryMass<<<dimGrid, dimBlock>>> (dens_d, Surf_d, CellAbscissa_d, CellOrdinate_d, xpl, ypl, NRAD, NSEC, HillRadius, mdcp0_d);
  gpuErrchk(cudaDeviceSynchronize());



	cudaFree(Surf_d );
  cudaFree(CellAbscissa_d);
  cudaFree(CellOrdinate_d);

  /* <-------------------------     CircumPlanetaryMass()       --------------------------> */



  // reduction mdcp
  mdcp = deviceReduce(mdcp0_d, size_grid);
  cudaFree(mdcp0_d);

  float dt = 0.999;

  EmptyPlanetSystemFile (sys);
  PhysicalTimeInitial = PhysicalTime;

  /* <-------------------------     MultiplyPolarGridbyConstant()       --------------------------> */

  float *fieldsrc_d, *fieldsrc;
  fieldsrc = (float *)malloc(((NRAD+1)*NSEC)*sizeof(float));
  gpuErrchk(cudaMalloc(&fieldsrc_d, ((NRAD+1)*NSEC)*sizeof(float)));
  gpuErrchk(cudaMemset(fieldsrc_d, 0, ((NRAD+1)*NSEC)*sizeof(float)));

  gpuErrchk(cudaMemcpy(fieldsrc_d, fieldsrc, ((NRAD+1)*NSEC)*sizeof(float), cudaMemcpyHostToDevice));

  MultiplyPolarGridbyConstant<<<dimGrid, dimBlock>>>(dens_d, fieldsrc_d, NRAD, NSEC, ScalingFactor);
  gpuErrchk(cudaDeviceSynchronize());

  gpuErrchk(cudaMemcpy(fieldsrc, fieldsrc_d, ((NRAD+1)*NSEC)*sizeof(float), cudaMemcpyDeviceToHost));
  cudaFree(fieldsrc_d);


  /* <-------------------------     MultiplyPolarGridbyConstant()       --------------------------> */

  for (int i = 0; i < 1; i++) {
    InnerOutputCounter++;

    if (InnerOutputCounter == 1) {
      InnerOutputCounter = 0;
      WriteBigPlanetSystemFile (sys, TimeStep);
      UpdateLog(force, sys, dens, energy, TimeStep, PhysicalTime, dimfxy);
    }

  }


  vradint = (float *) malloc(sizeof(float)*size_grid);
  pot = (float *) malloc(sizeof(float)*size_grid);
  vrad = (float *) malloc(sizeof(float)*size_grid);
  vthetaint = (float *) malloc(sizeof(float)*size_grid);
  vtheta = (float *) malloc(sizeof(float)*size_grid);
  powRmed = (float *) malloc(sizeof(float)*NRAD);
  temperatureint = (float *)malloc(sizeof(float)*size_grid);
  densint = (float *)malloc(sizeof(float)*size_grid);
  vradnew = (float *)malloc(sizeof(float)*size_grid);
  vthetanew = (float *)malloc(sizeof(float)*size_grid);
  energyint = (float *)malloc(sizeof(float)*size_grid);

  for (int i  = 0; i < size_grid; i++) {
    //rho[i] = 2*i;
    pot[i] = 0.00001*i;
    vrad[i] = 0.212*i;
    vtheta[i] = 0.1;
    densint[i] = 0.1;
  }

 for (int i = 0; i < NRAD; i++) {
   powRmed[i] = powf(Rmed[i],-2.5+SIGMASLOPE);
 }



    // cudaEvent_t start, stop;
    // float time;
    //
    //
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);
    // cudaEventRecord(start,0);

    // cuda function

    // cudaEventRecord(stop, 0);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&time, start, stop);
    // printf("CUDA execution time = %f ms\n",time);



  /* <-------------------------     substep1()       --------------------------> */


  gpuErrchk(cudaMalloc((void**)&press_d, size_grid*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&dens_d, size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&vradint_d, size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&pot_d, size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&vrad_d, size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&vthetaint_d, size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&vtheta_d, size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&invdiffRmed_d, NRAD*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&invRinf_d,NRAD*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&Rinf_d,NRAD*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&Rmed_d,NRAD*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&powRmed_d,NRAD*sizeof(float)));

	gpuErrchk(cudaMemcpy(press_d, press, size_grid*sizeof(float), cudaMemcpyHostToDevice ));
	gpuErrchk(cudaMemcpy(dens_d, dens, size_grid*sizeof(float), cudaMemcpyHostToDevice ));
  gpuErrchk(cudaMemcpy(vradint_d, vradint, size_grid*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(pot_d, pot, size_grid*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(vrad_d, vrad, size_grid*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(vthetaint_d, vthetaint, size_grid*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(vtheta_d, vtheta, size_grid*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(invdiffRmed_d, invdiffRmed, NRAD*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(invRinf_d, invRinf, NRAD*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(Rinf_d, Rinf, NRAD*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(Rmed_d, Rmed, NRAD*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(powRmed_d, powRmed, NRAD*sizeof(float), cudaMemcpyHostToDevice));

	substep1<<<dimGrid, dimBlock>>>(press_d, dens_d, vradint_d, invdiffRmed_d,pot_d,Rinf_d,
    invRinf_d, vrad_d, vthetaint_d, vtheta_d, Rmed_d,  dt, NRAD, NSEC, OmegaFrame, ZMPlus,
    IMPOSEDDISKDRIFT, SIGMASLOPE, powRmed_d);

  gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaMemcpy(vradint, vradint_d, size_grid*sizeof(float), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(vthetaint, vthetaint_d, size_grid*sizeof(float), cudaMemcpyDeviceToHost));


	cudaFree(press_d );
	cudaFree(pot_d);
  cudaFree(invRinf_d);
  cudaFree(Rinf_d);
  cudaFree(powRmed_d);


  /* <-------------------------     substep1()       --------------------------> */


  /* <-------------------------     substep2()       --------------------------> */

  gpuErrchk(cudaMalloc((void**)&temperatureint_d,size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&densint_d, size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&vradnew_d,size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&vthetanew_d,size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&energy_d,size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&energyint_d, size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&invdiffRsup_d, NRAD*sizeof(float)));


  gpuErrchk(cudaMemcpy(temperatureint_d, temperatureint, size_grid*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(densint_d, densint, size_grid*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(vradint_d, vradint, size_grid*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(vthetaint_d, vthetaint, size_grid*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(energy_d, energy, size_grid*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(invdiffRsup_d, invdiffRsup, NRAD*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(vradnew_d, vradnew, size_grid*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(vthetanew_d, vthetanew, size_grid*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(energyint_d, energyint, size_grid*sizeof(float), cudaMemcpyHostToDevice));


  substep2<<<dimGrid, dimBlock>>>(dens_d, vradint_d, vthetaint_d, temperatureint_d, NRAD, NSEC, CVNR, invdiffRmed_d,
  invdiffRsup_d, densint_d, Adiabaticc, Rmed_d, dt, vradnew_d, vthetanew_d, energy_d, energyint_d);

  gpuErrchk(cudaDeviceSynchronize());
  gpuErrchk(cudaMemcpy(densint, densint_d, size_grid*sizeof(float), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(temperatureint, temperatureint_d, size_grid*sizeof(float), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(vthetanew, vthetanew_d, size_grid*sizeof(float), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(vradnew, vradnew_d, size_grid*sizeof(float), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(energyint, energyint_d, size_grid*sizeof(float), cudaMemcpyDeviceToHost));

  cudaFree(dens_d);
  cudaFree(vradint_d);
  cudaFree(vthetaint_d);
  //cudaFree(vtheta_d);
  cudaFree(temperatureint_d);
  cudaFree(invdiffRmed_d);
  cudaFree(invdiffRsup_d);
  cudaFree(densint_d);
  cudaFree(Rmed_d);
  //cudaFree(vrad_d);
  //cudaFree(vradnew_d);
  //cudaFree(vthetanew_d);
  cudaFree(energy_d);
  cudaFree(energyint_d);


  /* <-------------------------     substep2()       --------------------------> */


  /* <-------------------------     ActualiseGas(Vrad, VradNew)       --------------------------> */

  gpuErrchk(cudaMemcpy(vthetanew_d, vtheta_d, size_grid*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(vtheta_d, vtheta, size_grid*sizeof(float), cudaMemcpyHostToDevice));

  gpuErrchk(cudaMemcpy(vtheta_d, vthetanew_d, size_grid*sizeof(float), cudaMemcpyDeviceToDevice));

  gpuErrchk(cudaDeviceSynchronize());
  gpuErrchk(cudaMemcpy(vtheta_d, vthetanew_d, size_grid*sizeof(float), cudaMemcpyDeviceToHost));

  /* <-------------------------     ActualiseGas(Vrad, VradNew)       --------------------------> */



  /* <-------------------------     ActualiseGas(Vtheta, VthetaNew)       --------------------------> */

  gpuErrchk(cudaMemcpy(vradnew_d, vradnew, size_grid*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(vrad_d, vrad, size_grid*sizeof(float), cudaMemcpyHostToDevice));

  gpuErrchk(cudaMemcpy(vrad_d, vradnew_d, size_grid*sizeof(float), cudaMemcpyDeviceToDevice));

  gpuErrchk(cudaDeviceSynchronize());
  gpuErrchk(cudaMemcpy(vrad, vrad_d, size_grid*sizeof(float), cudaMemcpyDeviceToHost));


  /* <-------------------------     ActualiseGas(Vtheta, VthetaNew)       --------------------------> */

  cudaFree(vrad_d);
  cudaFree(vradnew_d);
  cudaFree(vtheta_d);
  cudaFree(vthetanew_d);

  ApplyBoundaryCondition (vrad, vtheta, dens, energy, dt);

/* esto es parte de substep1

  if (SelfGravity){
    selfgravityupdate = YES;
    compute_selfgravity(Rho, VradInt, VthetaInt, dt, selfgravityupdate);
  }
  ComputeViscousTerms (VradInt, VthetaInt, Rho);
  UpdateVelocitiesWithViscosity(vradint, vthetaint, rho, dt);

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
