#include "Main.cuh"

extern int NRAD, NSEC;
extern float *SigmaMed, *EnergyMed;

__host__ void InitGasDensity (float *Dens)
{
  FillSigma ();
  for (int i = 0; i < NRAD; i++)
  {
    for (int j = 0; j < NSEC; j++)
    {
      Dens[j+i*NSEC] = SigmaMed[i];
    }
  }
}

__host__ void InitGasEnergy (float *energy)
{
  FillEnergy ();
  for (int i = 0; i < NRAD; i++)
  {
    for (int j = 0; j < NSEC; j++)
    {
      energy[j+i*NSEC] = EnergyMed[i];
    }
  }
}
