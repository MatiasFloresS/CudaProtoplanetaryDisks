#include <stdio.h>
#include <math.h>
#include <cstdarg>
#include <iostream>

__host__ void UpdateVelocitiesWithViscosity(float *RadialVelocity, float *AzimuthalVelocity, float *Rho, float DeltaT);
__host__ void ReadFile(char *ParameterFile);
