#include <stdio.h>
#include <math.h>

const int nrad = 128;
const int nsec = 384;

__host__ void UpdateVelocitiesWithViscosity(float *RadialVelocity, float *AzimuthalVelocity, float *Rho, float DeltaT);
