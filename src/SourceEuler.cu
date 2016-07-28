#include "main.cuh"
#include "kernels.cuh"
#include "SourceEuler.cuh"
#include "Theo.cuh"
using namespace std;

extern int blocksize, nsec2pot, nrad2pot;
extern int NRAD, NSEC, YES, LogGrid, size_grid, size_grid2;
extern float RMAX, RMIN, PI, MU, R;
extern float *invRmed, *invRinf, *invSurf, *invdiffRmed, *invdiffRsup;
extern float *invdiffSurf, *Rinf, *Rmed, *Rsup, *Surf, *cosns, *sinns;
extern float Adiabaticc, ADIABATICINDEX, FLARINGINDEX;
float *press, *CellAbscissa, *CellOrdinate, *AspectRatioRmed, *SoundSpeed, *temperature, *CoolingTimeMed,
*QplusMed;
extern string OUTPUTDIR;
extern bool CentrifugalBalance;
extern int SelfGravity, ViscosityAlpha;
float *GLOBAL_bufarray;
extern int Cooling;
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

  float Radii[129];
  char inputcharname[100];
  strncpy(inputcharname, InputName.c_str(), sizeof(inputcharname));
  inputcharname[sizeof(inputcharname)-1]=0;

  input = fopen (inputcharname, "r");
  if (input == NULL) {
    printf("Warning : no `radii.dat' file found. Using default.\n");
    if (LogGrid == YES) {
      for (i = 0; i <= NRAD; i++) {
	         Radii[i] = RMIN*expf(i*logf(RMAX/RMIN)/NRAD);
      }
    } else {
      for (i = 0; i <= NRAD; i++) {
	         Radii[i] = RMIN+drrsep*i;
      }
    }
  } else {
    printf("Reading 'radii.dat' file.\n");
    for (i = 0; i <= NRAD; i++) {
      fscanf (input, "%f", &temporary);
      Radii[i] = (float)temporary;
    }
  }

  for (i = 0; i < NRAD; i++) {
    Rmed[i] = 2.0/3.0*(Radii[i+1]*Radii[i+1]*Radii[i+1]-Radii[i]*Radii[i]*Radii[i]);
    Rmed[i] = Rmed[i] / (Radii[i+1]*Radii[i+1]-Radii[i]*Radii[i]);
  }

  for (i = 0; i < NRAD; i++) {
    Rinf[i] = Radii[i];
    Rsup[i] = Radii[i+1];
    Rmed[i] = 2.0/3.0*(Rsup[i]*Rsup[i]*Rsup[i]-Rinf[i]*Rinf[i]*Rinf[i]);
    Rmed[i] = Rmed[i] / (Rsup[i]*Rsup[i]-Rinf[i]*Rinf[i]);
    Surf[i] = PI*(Rsup[i]*Rsup[i]-Rinf[i]*Rinf[i])/(float)NSEC;
    invRmed[i] = 1.0/Rmed[i];
    invSurf[i] = 1.0/Surf[i];
    invdiffRsup[i] = 1.0/(Rsup[i]-Rinf[i]);
    invRinf[i] = 1.0/Rinf[i];
  }

  Rinf[NRAD]=Radii[NRAD];
  for (i = 1; i < NRAD; i++) {
    invdiffRmed[i] = 1.0/(Rmed[i]-Rmed[i-1]);
  }

  char outputcharname[100];
  strncpy(outputcharname, OutputName.c_str(), sizeof(outputcharname));
  outputcharname[sizeof(outputcharname)-1]=0;
  output = fopen (outputcharname, "w");
  if (output == NULL) {
    printf ("Can't write %s.\nProgram stopped.\n", outputcharname);
    exit (1);
  }
  for (i = 0; i <= NRAD; i++) {
    fprintf (output, "%f\n", Radii[i]);
  }
  fclose (output);
  if (input != NULL) fclose (input);
}

__host__ void InitEuler (float *vrad, float *vtheta, float *dens, float *energy)
{

  /* Init Transport */
  // float *RadMomP, *RadMomM, *ThetaMomP, *ThetaMomM, *Work, *QRStar, *ExtLabel;
  // float *VthetaRes, *Elongtions;  dicen que es (NRAD+3)(NSEC+1) + 5
  // float * TempShift, *dq;    todos NSEC*NRAD

  /* Init Viscosity */
  float *DivergenceVelocity, *DRR, *DRP, *DPP, *TAURR, *TAURP, *TAUPP;
  float *RhoStar, *RhoInt, *VradNew, *VradInt, *VthetaNew, *VthetaInt, *EnergyNew;
  float *EnergyInt, *TemperInt, *Potential, *Pressure, *Temperature, *Qplus;

  CellAbscissa = (float *)malloc(sizeof(float)*size_grid);
  CellOrdinate = (float *)malloc(sizeof(float)*size_grid);
  cosns = (float *)malloc(sizeof(float)*NSEC);
  sinns = (float *)malloc(sizeof(float)*NSEC);
  AspectRatioRmed = (float *)malloc(sizeof(float)*NRAD);
  press = (float *)malloc(sizeof(float)*size_grid);
  SoundSpeed = (float *)malloc(sizeof(float)*size_grid);
  temperature = (float *)malloc(sizeof(float)*size_grid);

  for (int i = 0; i < NSEC; i++) {
      cosns[i] = cos((2.0*CUDART_PI_F*i)/NSEC);
      sinns[i] = sin((2.0*CUDART_PI_F*i)/NSEC);
  }


  for (int i = 0; i < NRAD; i++) AspectRatioRmed[i] = AspectRatio(Rmed[i]);

  InitComputeAccelhost();
  ComputeSoundSpeedhost(dens, energy);
  ComputePressureFieldhost(dens, energy);
  ComputeTemperatureFieldhost(dens, energy);


 /* <---------     InitGasVelocities()        --------> */


 InitGasVelocitieshost(vrad, vtheta);


  /* <---------     InitGasVelocities()        --------> */

}


__host__ void InitComputeAccelhost()
{
  float *CellAbscissa_d, *CellOrdinate_d, *Rmed_d, *sinns_d, *cosns_d;
  dim3 dimGrid( nsec2pot/blocksize, nrad2pot/blocksize );
  dim3 dimBlock( blocksize, blocksize );

  gpuErrchk(cudaMalloc((void**)&CellAbscissa_d, size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&CellOrdinate_d, size_grid*sizeof(float) ));
  gpuErrchk(cudaMalloc((void**)&Rmed_d, NRAD*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&cosns_d, NSEC*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&sinns_d, NSEC*sizeof(float)));


  gpuErrchk(cudaMemcpy(CellAbscissa_d, CellAbscissa, size_grid*sizeof(float), cudaMemcpyHostToDevice ));
  gpuErrchk(cudaMemcpy(CellOrdinate_d, CellOrdinate, size_grid*sizeof(float), cudaMemcpyHostToDevice ));
  gpuErrchk(cudaMemcpy(Rmed_d, Rmed, NRAD*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(cosns_d, cosns, NSEC*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(sinns_d, sinns, NSEC*sizeof(float), cudaMemcpyHostToDevice));

  InitComputeAccel<<<dimGrid, dimBlock>>>(CellAbscissa_d, CellOrdinate_d, Rmed_d, cosns_d,
    sinns_d, NSEC, NRAD);
  gpuErrchk(cudaDeviceSynchronize());
  gpuErrchk(cudaMemcpy(CellAbscissa, CellAbscissa_d, size_grid*sizeof(float), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(CellOrdinate, CellOrdinate_d, size_grid*sizeof(float), cudaMemcpyDeviceToHost));

  cudaFree(CellAbscissa_d );
  cudaFree(CellOrdinate_d );
  cudaFree(Rmed_d);
  cudaFree(cosns_d);
  cudaFree(sinns_d);

}

__host__ void ComputeSoundSpeedhost(float *dens, float *energy)
{
  dim3 dimGrid( nsec2pot/blocksize, nrad2pot/blocksize );
  dim3 dimBlock( blocksize, blocksize );

  float *SoundSpeed_d, *energy_d, *Rmed_d, *AspectRatioRmed_d, *dens_d;

  gpuErrchk(cudaMalloc((void**)&SoundSpeed_d, size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&energy_d, size_grid*sizeof(float) ));
  gpuErrchk(cudaMalloc((void**)&Rmed_d, NRAD*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&AspectRatioRmed_d, NRAD*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&dens_d, size_grid2*sizeof(float)));

  gpuErrchk(cudaMemcpy(SoundSpeed_d, SoundSpeed, size_grid*sizeof(float), cudaMemcpyHostToDevice ));
  gpuErrchk(cudaMemcpy(energy_d, energy, size_grid*sizeof(float), cudaMemcpyHostToDevice ));
  gpuErrchk(cudaMemcpy(Rmed_d, Rmed, NRAD*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(AspectRatioRmed_d, AspectRatioRmed, NRAD*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(dens_d, dens, size_grid2*sizeof(float), cudaMemcpyHostToDevice));

  ComputeSoundSpeed<<<dimGrid, dimBlock>>>(SoundSpeed_d, dens_d, Rmed_d, energy_d, NSEC, NRAD,
     Adiabaticc, ADIABATICINDEX, FLARINGINDEX, AspectRatioRmed_d);
  gpuErrchk(cudaDeviceSynchronize());
  gpuErrchk(cudaMemcpy(SoundSpeed, SoundSpeed_d, size_grid*sizeof(float), cudaMemcpyDeviceToHost));

  cudaFree(SoundSpeed_d );
  cudaFree(dens_d );
  cudaFree(Rmed_d);
  cudaFree(energy_d);
  cudaFree(AspectRatioRmed_d);
}

__host__ void ComputePressureFieldhost(float *dens, float *energy)
{
  dim3 dimGrid( nsec2pot/blocksize, nrad2pot/blocksize );
  dim3 dimBlock( blocksize, blocksize );

  float *SoundSpeed_d, *energy_d, *press_d, *dens_d;

  gpuErrchk(cudaMalloc((void**)&SoundSpeed_d, size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&energy_d, size_grid*sizeof(float) ));
  gpuErrchk(cudaMalloc((void**)&press_d, size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&dens_d, size_grid2*sizeof(float)));

  gpuErrchk(cudaMemcpy(SoundSpeed_d, SoundSpeed, size_grid*sizeof(float), cudaMemcpyHostToDevice ));
  gpuErrchk(cudaMemcpy(energy_d, energy, size_grid*sizeof(float), cudaMemcpyHostToDevice ));
  gpuErrchk(cudaMemcpy(press_d, press, size_grid*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(dens_d, dens, size_grid2*sizeof(float), cudaMemcpyHostToDevice));

  ComputePressureField<<<dimGrid, dimBlock>>>(SoundSpeed_d, dens_d, press_d, Adiabaticc, NRAD, NSEC, ADIABATICINDEX, energy_d);
  gpuErrchk(cudaDeviceSynchronize());
  gpuErrchk(cudaMemcpy(press, press_d, size_grid*sizeof(float), cudaMemcpyDeviceToHost));

  cudaFree(SoundSpeed_d);
  cudaFree(dens_d);
  cudaFree(press_d);
  cudaFree(energy_d);

}

__host__ void ComputeTemperatureFieldhost(float *dens, float *energy)
{
  dim3 dimGrid( nsec2pot/blocksize, nrad2pot/blocksize );
  dim3 dimBlock( blocksize, blocksize );

  float *dens_d, *temperature_d, *press_d, *energy_d;

  gpuErrchk(cudaMalloc((void**)&dens_d, size_grid2*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&temperature_d, size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&press_d, size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&energy_d, size_grid*sizeof(float)));

  gpuErrchk(cudaMemcpy(dens_d, dens, size_grid2*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(temperature_d, temperature, size_grid*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(press_d, press, size_grid*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(energy_d, energy, size_grid*sizeof(float), cudaMemcpyHostToDevice));

  ComputeTemperatureField<<<dimGrid, dimBlock>>>(dens_d, temperature_d, press_d, energy_d, MU, R, ADIABATICINDEX, Adiabaticc, NSEC, NRAD);
  gpuErrchk(cudaDeviceSynchronize());
  gpuErrchk(cudaMemcpy(temperature, temperature_d, size_grid*sizeof(float), cudaMemcpyDeviceToHost));

  cudaFree(dens_d);
  cudaFree(temperature_d);
  cudaFree(press_d);
  cudaFree(energy_d);
}

__host__ void InitGasVelocitieshost(float *vrad, float *vtheta)
{
  if (CentrifugalBalance) {
    // ocurre algo , es cuando se coloca ./bin/fargoGPU  -b in/template.par
  }

  if (!CentrifugalBalance && SelfGravity) // init_azimutalvelocity_withSG (vtheta);

  if (ViscosityAlpha) make1Dprofile(SoundSpeed, GLOBAL_bufarray);

  CoolingTimeMed = (float *)malloc(sizeof(float)*size_grid);
  QplusMed = (float *)malloc(sizeof(float)*size_grid);

  if (!Cooling)
  {
    FillCoolingTime();
    FillQplus();
  }
}

__host__ void make1Dprofile(float *SoundSpeed, float *GLOBAL_bufarray)
{

  dim3 dimGrid( nrad2pot/blocksize, 1);
  dim3 dimBlock( blocksize, 1);

  GLOBAL_bufarray = (float *)malloc(sizeof(float)*NRAD);

  float *device_out2, *SoundSpeed_d, *GLOBAL_bufarray_d;

  gpuErrchk(cudaMalloc((void**)&SoundSpeed_d, size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&GLOBAL_bufarray_d, NRAD*sizeof(float)));

  gpuErrchk(cudaMalloc(&device_out2, sizeof(float)*NRAD));
  gpuErrchk(cudaMemset(device_out2, 1, sizeof(float)*NRAD));

  gpuErrchk(cudaMemcpy(SoundSpeed_d, SoundSpeed, size_grid*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(GLOBAL_bufarray_d, GLOBAL_bufarray, NRAD*sizeof(float), cudaMemcpyHostToDevice));

  make1Dprofile<<<dimGrid, dimBlock>>>(device_out2, SoundSpeed_d, GLOBAL_bufarray_d, NSEC, NRAD);
  gpuErrchk(cudaDeviceSynchronize());
  gpuErrchk(cudaMemcpy(GLOBAL_bufarray, GLOBAL_bufarray_d, NRAD*sizeof(float), cudaMemcpyDeviceToHost));

  cudaFree(GLOBAL_bufarray_d);
  cudaFree(device_out2);
  cudaFree(SoundSpeed_d);

  printf("%f\n", GLOBAL_bufarray[0]);
}
