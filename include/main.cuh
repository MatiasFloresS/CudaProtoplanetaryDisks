#include <stdio.h>
#include <math.h>
#include <cstdarg>
#include <iostream>
#include "string.h"

__host__ void ReadFile(char *ParameterFile);
__host__ void TellEverything();
__host__ void PrintUsage(char *argv);
__host__ void UpdateVelocitiesWithViscosity(float *RadialVelocity, float *AzimuthalVelocity, float *Rho, float DeltaT);
__host__ float CircumPlanetaryMasshost(float xpl, float ypl);
__host__ void MultiplyPolarGridbyConstanthost(float *dens);
__host__ void substep1host(float *vrad, float *vtheta, float *dens, float dt);
__host__ void substep2host(float *dens, float *energy, float dt);
__host__ void ActualiseGasVrad(float *vrad, float *vradnew);
__host__ void ActualiseGasVtheta(float *vtheta, float *vthetanew);
