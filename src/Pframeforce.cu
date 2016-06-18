#include "main.cuh"
#include "Theo.cuh"

extern int NRAD, NSEC;
extern float *SigmaMed, *EnergyMed;
extern float *dens, *energy;

__host__ void InitGasDensity ()
{
    FillSigma ();
    for (int i = 0; i < NRAD; i++) {
      for (int j = 0; j < NSEC; j++) {
        dens[j+i*NSEC] = SigmaMed[i];
      }
    }
}

__host__ void InitGasEnergy ()
{
    FillEnergy ();
    for (int i = 0; i < NRAD; i++) {
      for (int j = 0; j < NSEC; j++) {
        energy[j+i*NSEC] = EnergyMed[i];
      }
    }
}
