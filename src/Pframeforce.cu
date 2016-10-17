#include "Main.cuh"

extern int NRAD, NSEC, NO, Indirect_Term;
extern float *SigmaMed, *EnergyMed;
static Pair IndirectTerm;
extern Pair DiskOnPrimaryAcceleration;

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

__host__ void FillForcesArrays (PlanetarySystem *sys, float *Dens, float *energy)
{
  ComputeIndirectTerm ();
}


__host__ void ComputeIndirectTerm ()
{
  IndirectTerm.x = -DiskOnPrimaryAcceleration.x;
  IndirectTerm.x = -DiskOnPrimaryAcceleration.y;
  if (Indirect_Term == NO)
  {
    IndirectTerm.x = 0.0;
    IndirectTerm.y = 0.0;
  }
}
