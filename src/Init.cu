#include "Main.cuh"

extern int NRAD, NSEC, size_grid;
extern float *Rmed, *Rmed_d, *Label_d;
extern dim3 dimGrid2, dimBlock2;

__host__ void Initialization (float *Dens, float *Vrad, float *Vtheta, float *Energy, float *Label, PlanetarySystem *sys)
{
  InitEuler (Vrad, Vtheta, Dens, Energy);
  InitLabel (Label, sys);
  WriteDim();
}

__host__ void InitLabel (float *Label, PlanetarySystem *sys)
{
  float xp,yp, rhill, rp;
  xp = sys->x[0];
  yp = sys->y[0];
  rp = sqrt(xp*xp+yp*yp);
  rhill = rp * pow(sys->mass[0]/3., 1./3);

  InitLabelKernel<<<dimGrid2, dimBlock2>>>(Label_d, xp, yp, rhill, Rmed_d, NRAD, NSEC);
  gpuErrchk(cudaDeviceSynchronize());
}
