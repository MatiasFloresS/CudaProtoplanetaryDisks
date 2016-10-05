#include "Main.cuh"


extern int NRAD, NSEC, nsec2pot, blocksize, nrad2pot, size_grid;
extern float *Rmed, *Rmed_d, *label_d;
extern dim3 dimGrid2, dimBlock2;

__host__ void Initialization (float *dens, float *energy, float *vrad, float *vtheta, float * label, PlanetarySystem *sys)
{
  InitEuler (dens, energy, vrad, vtheta);
  InitLabel (sys);
  WriteDim();
}

__host__ void InitLabel(PlanetarySystem *sys)
{
  float xp,yp, rhill, rp;
  xp = sys->x[0];
  yp = sys->y[0];
  rp = sqrt(xp*xp+yp*yp);
  rhill = rp * pow(sys->mass[0]/3., 1./3);

  InitLabelKernel<<<dimGrid2, dimBlock2>>>(label_d, xp, yp, rhill, Rmed_d, NRAD, NSEC);
  gpuErrchk(cudaDeviceSynchronize());
}
