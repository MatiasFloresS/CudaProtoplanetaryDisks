#include "Main.cuh"
#include "Kernels.cuh"
#include "SourceEuler.cuh"
#include "Theo.cuh"
using namespace std;

extern int blocksize, nsec2pot, nrad2pot, NRAD, NSEC, YES, LogGrid, size_grid, SelfGravity, ViscosityAlpha, \
Adiabaticc, Cooling, size_grid2;

extern bool CentrifugalBalance;
extern string OUTPUTDIR;

extern float RMAX, RMIN, PI, MU, R,*invRmed, *invRinf, *invSurf, *invdiffRmed, *invdiffRsup, *Radii, \
*GLOBAL_bufarray, *invdiffSurf, *Rinf, *Rmed, *Rsup, *Surf, *cosns, *sinns, ADIABATICINDEX, \
FLARINGINDEX, *vt_int, OmegaFrame1, *SigmaInf, G, ASPECTRATIO, SIGMA0, SIGMASLOPE, IMPOSEDDISKDRIFT, \
*CoolingTimeMed, *QplusMed , *viscosity_array, *dens_d, *Rmed_d;

float *press, *CellAbscissa, *CellOrdinate, *AspectRatioRmed, *SoundSpeed, *temperature, *vtheta_d, \
*CellAbscissa_d, *CellOrdinate_d, *sinns_d, *cosns_d, *vt_cent, *Rinf_d, *SigmaInf_d, *vrad_d, *SoundSpeed_d, \
*energy_d, *AspectRatioRmed_d, *press_d, *temperature_d, *viscosity_array_d, *Kr_aux, *Kt_aux;

extern dim3 dimGrid2, dimBlock2;

double *Radii2;

__host__ void FillPolar1DArray()
{
  FILE *input, *output;
  int i,ii;
  float drrsep;
  float temporary;
  string InputName, OutputName;
  drrsep = (RMAX-RMIN)/NRAD;
  InputName = OUTPUTDIR +"radii.dat";
  OutputName = OUTPUTDIR +"used_rad.dat";

  Radii = (float *) malloc(sizeof(float)*(NRAD+1));
  Radii2 = (double *) malloc(sizeof(double)*(NRAD+1));
  vt_cent = (float *) malloc(sizeof(float)*NRAD);
  Rinf = (float *) malloc(sizeof(float)*(NRAD));
  Rmed = (float *) malloc(sizeof(float)*(NRAD));
  Rsup = (float *) malloc(sizeof(float)*(NRAD));
  Surf = (float *) malloc(sizeof(float)*(NRAD));
  invRinf = (float *) malloc(sizeof(float)*(NRAD));
  invSurf = (float *) malloc(sizeof(float)*(NRAD));
  invRmed = (float *) malloc(sizeof(float)*(NRAD));
  invdiffSurf = (float *) malloc(sizeof(float)*(NRAD));
  invdiffRsup = (float *) malloc(sizeof(float)*(NRAD));
  invdiffRmed = (float *) malloc(sizeof(float)*(NRAD));

  char inputcharname[100];
  strncpy(inputcharname, InputName.c_str(), sizeof(inputcharname));
  inputcharname[sizeof(inputcharname)-1]=0;

  double u, theta, base,algo, den_SGP_K, algo2;

  Kr_aux = (float *)malloc(sizeof(float)*2*size_grid);
  Kt_aux = (float *)malloc(sizeof(float)*2*size_grid);

  input = fopen (inputcharname, "r");
  if (input == NULL)
  {
    printf("Warning : no `radii.dat' file found. Using default.\n");
    if (LogGrid == YES)
    {
      for (i = 0; i <= NRAD; i++)
      {
        Radii2[i] = RMIN*exp((double)i/(double)NRAD*log(RMAX / RMIN));
        Radii[i] = (float) Radii2[i];
      }


      for (i = 0; i < 2*NRAD; i++)
      {
        if(i<NRAD) u = log(Radii2[i]/Radii2[0]);
        else u = -log(Radii2[2*NRAD-i]/Radii2[0]);



        for (int j = 0; j < NSEC; j++) {
          theta = 2.0*M_PI*(double)j  / (double)NSEC;
          base = 0.03*0.03 * exp(u) + 2.0* (cosh(u) - cos(theta));
          den_SGP_K = pow(base , -1.5);

          algo = 1.0 + 0.03*0.03 - cos(theta) * exp(-u);
          algo *= den_SGP_K;

          algo2 = sin(theta) * den_SGP_K;
          if ( i==10 && j == 10) printf("%g\n",algo2 );
          Kr_aux[i*NSEC+j] = (float) algo;
          Kt_aux[i*NSEC+j] = (float) algo2;
        }

      }


      // FILE *f;
      // f = fopen("Kr.raw", "w");
      //
      // for (int i = 0; i < NRAD; i++) {
      //   for (int j = 0; j < NSEC; j++) {
      //     fwrite((void *) &Kr_aux[i*NSEC+j], 1, sizeof(float), f);
      //   }
      // }
      // fclose(f);
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
    //Rmed[i] = 2.0/3.0*(Radii[i+1]*Radii[i+1]*Radii[i+1]-Radii[i]*Radii[i]*Radii[i]);
    //Rmed[i] = Rmed[i] / (Radii[i+1]*Radii[i+1]-Radii[i]*Radii[i]);
  }

  for (i = 0; i < NRAD; i++)
  {
    Rinf[i] = Radii[i];
    Rsup[i] = Radii[i+1];
    Rmed[i] = 2.0/3.0*(Rsup[i]*Rsup[i]*Rsup[i]-Rinf[i]*Rinf[i]*Rinf[i]);
    Rmed[i] = Rmed[i] / (Rsup[i]*Rsup[i]-Rinf[i]*Rinf[i]);
    Surf[i] = M_PI*(Rsup[i]*Rsup[i]-Rinf[i]*Rinf[i])/(float)NSEC;
    invRmed[i] = 1.0/Rmed[i];
    invSurf[i] = 1.0/Surf[i];
    invdiffRsup[i] = 1.0/(Rsup[i]-Rinf[i]);
    invRinf[i] = 1.0/Rinf[i];
  }

  Rinf[NRAD]=Radii[NRAD];
  for (i = 1; i < NRAD; i++) invdiffRmed[i] = 1.0/(Rmed[i]-Rmed[i-1]);

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

__host__ void InitEuler (float *dens, float *energy, float *vrad, float *vtheta)
{

  CellAbscissa = (float *)malloc(sizeof(float)*size_grid);
  CellOrdinate = (float *)malloc(sizeof(float)*size_grid);
  cosns = (float *)malloc(sizeof(float)*NSEC);
  sinns = (float *)malloc(sizeof(float)*NSEC);
  AspectRatioRmed = (float *)malloc(sizeof(float)*NRAD);
  press = (float *)malloc(sizeof(float)*size_grid);
  SoundSpeed = (float *)malloc(sizeof(float)*size_grid);
  temperature = (float *)malloc(sizeof(float)*size_grid);

  for (int i = 0; i < NSEC; i++)
  {
      cosns[i] = cos((2.0*CUDART_PI_F*i)/NSEC);
      sinns[i] = sin((2.0*CUDART_PI_F*i)/NSEC);
  }


  for (int i = 0; i < NRAD; i++) AspectRatioRmed[i] = AspectRatio(Rmed[i]);

  Computecudamalloc(dens, energy, vrad, vtheta);

  InitComputeAccel<<<dimGrid2, dimBlock2>>>(CellAbscissa_d, CellOrdinate_d, Rmed_d, cosns_d,
    sinns_d, NSEC, NRAD);
  gpuErrchk(cudaDeviceSynchronize());

  ComputeSoundSpeed<<<dimGrid2, dimBlock2>>>(SoundSpeed_d, dens_d, Rmed_d, energy_d, NSEC, NRAD,
    Adiabaticc, ADIABATICINDEX, FLARINGINDEX, AspectRatioRmed_d);
  gpuErrchk(cudaDeviceSynchronize());

  ComputePressureField<<<dimGrid2, dimBlock2>>>(SoundSpeed_d, dens_d, press_d, Adiabaticc, NRAD,
    NSEC, ADIABATICINDEX, energy_d);
  gpuErrchk(cudaDeviceSynchronize());

  ComputeTemperatureField<<<dimGrid2, dimBlock2>>>(dens_d, temperature_d, press_d, energy_d, MU, R,
    ADIABATICINDEX, Adiabaticc, NSEC, NRAD);
  gpuErrchk(cudaDeviceSynchronize());

  InitGasVelocitieshost(vrad, vtheta);

}


__host__ void InitGasVelocitieshost(float *vrad, float *vtheta)
{
  float t1, r, ri, vt_cent_d;
  int nrad2potlocal;

  CoolingTimeMed = (float *)malloc(sizeof(float)*size_grid);
  QplusMed = (float *)malloc(sizeof(float)*size_grid);
  viscosity_array = (float *)malloc(sizeof(float)*NRAD+1);

  /* Pressure is already initialized: cf initeuler in SourceEuler.c ...
    Initialization of azimutal velocity with exact centrifugal balance */

  if (CentrifugalBalance)
  {
    /* vt_int \equiv Romega = grad(P)/sigma + \partial(phi)/\partial(r) - acc_sg_radial
    ./bin/fargoGPU  -b in/template.par */

    gpuErrchk(cudaMemcpy(press, press_d, size_grid*sizeof(float), cudaMemcpyDeviceToHost));
    Make1Dprofilehost(press);

    /* global axisymmetric pressure field */
    for (int i = 1; i < NRAD; i++)
    {
      vt_int[i] = ( GLOBAL_bufarray[i] - GLOBAL_bufarray[i-1]) / \
      (.5*(Sigma(Rmed[i]) + Sigma(Rmed[i-1])))/(Rmed[i]-Rmed[i-1]);
    }
    /* Case of a disk with self-gravity */
    // if ( SelfGravity )

    for (int i = 1; i < NRAD; i++)
      vt_int[i] = sqrtf(vt_int[i]*Radii[i]) - Radii[i]*OmegaFrame1;

    t1 = vt_cent[0] = vt_int[1]+.75*(vt_int[1]-vt_int[2]);
    //r1 = ConstructSequence (vt_cent, vt_int, NRAD);
  }

  if (!CentrifugalBalance && SelfGravity) // init_azimutalvelocity_withSG (vtheta);

  if (ViscosityAlpha)
  {
    gpuErrchk(cudaMemcpy(SoundSpeed, SoundSpeed_d, size_grid*sizeof(float), cudaMemcpyDeviceToHost));
    Make1Dprofilehost(SoundSpeed);
  }


  if (Cooling)
  {
    FillCoolingTime();
    FillQplus();
  }

  for (int i = 0; i <= NRAD; i++)
  {
    if (i == NRAD) viscosity_array[i] = FViscosity(Rmed[NRAD-1]);
    else viscosity_array[i] = FViscosity(Rmed[i]);
  }

  if(!IsPow2(NRAD+1)) nrad2potlocal = NearestPowerOf2(NRAD+1);

  gpuErrchk(cudaMalloc((void**)&viscosity_array_d, (NRAD+1)*sizeof(float)));
  gpuErrchk(cudaMemcpy(viscosity_array_d, viscosity_array, (NRAD+1)*sizeof(float), cudaMemcpyHostToDevice));

  InitGasVelocities<<<dimGrid2, dimBlock2>>>(viscosity_array_d, NSEC, NRAD, SelfGravity, Rmed_d,
  G, ASPECTRATIO, FLARINGINDEX, SIGMASLOPE, CentrifugalBalance, vrad_d, vtheta_d, ViscosityAlpha,
  IMPOSEDDISKDRIFT, SIGMA0, SigmaInf_d, OmegaFrame1, Rinf_d); // falta vt_cent_d

  gpuErrchk(cudaDeviceSynchronize());
  gpuErrchk(cudaMemcpy(vrad, vrad_d, size_grid2*sizeof(float), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(vtheta, vtheta_d, size_grid2*sizeof(float), cudaMemcpyDeviceToHost));

  for (int i = 0; i < NSEC; i++) vrad[i] = vrad[i+NSEC*NRAD] = 0.0;

  gpuErrchk(cudaMemcpy(vrad_d, vrad, size_grid2*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(vtheta_d, vtheta, size_grid2*sizeof(float), cudaMemcpyHostToDevice));

}

__host__ void Computecudamalloc(float *dens, float *energy, float *vrad, float *vtheta)
{
  gpuErrchk(cudaMalloc((void**)&CellAbscissa_d, size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&CellOrdinate_d, size_grid*sizeof(float) ));
  gpuErrchk(cudaMalloc((void**)&cosns_d, NSEC*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&sinns_d, NSEC*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&SoundSpeed_d, size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&energy_d, size_grid*sizeof(float) ));
  gpuErrchk(cudaMalloc((void**)&AspectRatioRmed_d, NRAD*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&press_d, size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&temperature_d, size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&vrad_d, size_grid2*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&vtheta_d, size_grid2*sizeof(float)));
  //gpuErrchk(cudaMalloc((void**)&vt_cent_d, NRAD*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&SigmaInf_d, NRAD*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&Rinf_d, NRAD*sizeof(float)));

  gpuErrchk(cudaMemcpy(CellAbscissa_d, CellAbscissa, size_grid*sizeof(float), cudaMemcpyHostToDevice ));
  gpuErrchk(cudaMemcpy(CellOrdinate_d, CellOrdinate, size_grid*sizeof(float), cudaMemcpyHostToDevice ));
  gpuErrchk(cudaMemcpy(cosns_d, cosns, NSEC*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(sinns_d, sinns, NSEC*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(SoundSpeed_d, SoundSpeed, size_grid*sizeof(float), cudaMemcpyHostToDevice ));
  gpuErrchk(cudaMemcpy(energy_d, energy, size_grid*sizeof(float), cudaMemcpyHostToDevice ));
  gpuErrchk(cudaMemcpy(AspectRatioRmed_d, AspectRatioRmed, NRAD*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(press_d, press, size_grid*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(temperature_d, temperature, size_grid*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(vrad_d, vrad, size_grid2*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(vtheta_d, vtheta, size_grid2*sizeof(float), cudaMemcpyHostToDevice));
  //gpuErrchk(cudaMemcpy(vt_cent_d, vt_cent, NRAD*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(SigmaInf_d, SigmaInf, NRAD*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(Rinf_d, Rinf, NRAD*sizeof(float), cudaMemcpyHostToDevice));

}
