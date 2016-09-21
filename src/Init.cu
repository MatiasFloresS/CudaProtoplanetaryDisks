#include "Main.cuh"
#include "Types.cuh"
#include "SourceEuler.cuh"
#include "Kernels.cuh"
#include "Output.cuh"
#include "Init.cuh"

extern int NRAD, NSEC, nsec2pot, blocksize, nrad2pot, size_grid;
extern float *Rmed, *Rmed_d, *label_d;
extern dim3 dimGrid2, dimBlock2;

__host__ void Initialization (float *dens, float *energy, float *vrad, float *vtheta, float * label, PlanetarySystem *sys)
{
  float rp, rhill, xp, yp;
  xp = sys->x[0];
  yp = sys->y[0];
  rp = sqrt(xp*xp+yp*yp);
  rhill = rp * pow(sys->mass[0]/3., 1./3);

  InitEuler (dens, energy, vrad, vtheta);
  InitLabel<<<dimGrid2, dimBlock2>>>(label_d, xp, yp, rhill, Rmed_d, NRAD, NSEC);
  gpuErrchk(cudaDeviceSynchronize());
  WriteDim();
}
