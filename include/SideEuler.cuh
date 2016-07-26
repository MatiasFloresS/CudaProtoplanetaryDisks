#include "main.cuh"

__host__ void ApplyBoundaryCondition (float *vrad, float *vtheta, float *dens, float *energy, float step);
__host__ void OpenBoundaryhost(float *vrad, float *dens, float *energy);
__host__ void NonReflectingBoundaryhost(float *vrad, float *dens, float *energy);
__host__ void ReduceMeanHost();
__host__ void ReduceCshost();
