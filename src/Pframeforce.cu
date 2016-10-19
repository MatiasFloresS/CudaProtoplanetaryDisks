#include "Main.cuh"

extern int NRAD, NSEC, NO, Indirect_Term, size_grid, RocheSmoothing, ForcedCircular;
extern float *SigmaMed, *EnergyMed, *Potential_d, G, *Rmed_d, ROCHESMOOTHING, MassTaper, *q0, \
*PlanetMasses, *q1;

static Pair IndirectTerm;
extern Pair DiskOnPrimaryAcceleration;

extern dim3 dimGrid2, dimBlock2;

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

__host__ void InitGasEnergy (float *Energy)
{
  FillEnergy ();
  for (int i = 0; i < NRAD; i++)
  {
    for (int j = 0; j < NSEC; j++)
    {
      Energy[j+i*NSEC] = EnergyMed[i];
    }
  }
}

__host__ void FillForcesArrays (PlanetarySystem *sys, float *Dens, float *Energy)
{
  int NbPlanets;
  float xplanet, yplanet, mplanet, PlanetDistance, InvPlanetDistance3, RRoche, smooth, smoothing;
  NbPlanets = sys->nb;

  /* Indirect term star on gas here */
  ComputeIndirectTerm ();

  gpuErrchk(cudaMemset(Potential_d, 0, size_grid*sizeof(float)));
  /* -- Gravitational potential from planet on gas -- */
  for (int k = 0; k < NbPlanets; k++)
  {
    xplanet = sys->x[k];
    yplanet = sys->y[k];
    mplanet = sys->mass[k]*MassTaper;
    PlanetDistance = sqrt(xplanet*xplanet+yplanet*yplanet);
    InvPlanetDistance3 = 1.0/PlanetDistance/PlanetDistance/PlanetDistance;
    RRoche = PlanetDistance*pow((1.0/3.0*mplanet),1.0/3.0);
    if (RocheSmoothing) smoothing = RRoche*ROCHESMOOTHING;
    else smoothing = Compute_smoothing (PlanetDistance);
    smooth = smoothing*smoothing;

    FillForcesArraysKernel<<<dimGrid2,dimBlock2>>>(Rmed_d, NSEC, NRAD, xplanet, yplanet, smooth,
      G, mplanet, Indirect_Term, InvPlanetDistance3, Potential_d, IndirectTerm);
    gpuErrchk(cudaDeviceSynchronize());
  }
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

__host__ void AdvanceSystemFromDisk (Force *force, float *Dens, float *Energy, PlanetarySystem *sys, float dt)
{
  int NbPlanets;
  float m, x, y, r, smoothing;
  Pair gamma;
  NbPlanets = sys->nb;

  for (int k = 0; k < NbPlanets; k++)
  {
    if (sys->FeelDisk[k])
    {
      m = sys->mass[k];
      x = sys->x[k];
      y = sys->y[k];
      r = sqrt(x*x + y*y);
      if (RocheSmoothing) smoothing = r*pow(m/3.,1./3.)*ROCHESMOOTHING;
      else smoothing = Compute_smoothing (r);
      gamma = ComputeAccel (force, Dens, x, y, smoothing, m);
      sys->vx[k] += dt * gamma.x;
      sys->vy[k] += dt * gamma.y;
      sys->vx[k] += dt * IndirectTerm.x;
      sys->vy[k] += dt * IndirectTerm.y;
    }
  }
}

__host__ void AdvanceSystemRK5 (PlanetarySystem *sys, float dt)
{
  int nb;
  bool *feelothers;
  nb = sys->nb;
  if (!ForcedCircular)
  {
    for (int k = 0; k < nb; k++)
    {
      q0[k] = sys->x[k];
      q0[k+nb] = sys->y[k];
      q0[k+2*nb] = sys->vx[k];
      q0[k+3*nb] = sys->vy[k];
      PlanetMasses[k] = sys->mass[k];
    }
    feelothers = sys->FeelOthers;
    RungeKutta (q0, dt, PlanetMasses, q1, nb, feelothers);
  }
}
