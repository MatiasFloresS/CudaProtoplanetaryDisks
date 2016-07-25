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
__host__ void MultiplyPolarGridbyConstanthost();
__host__ void substep1host(float dt);
__host__ void substep2host(float dt);
__host__ void ActualiseGasVrad();
__host__ void ActualiseGasVtheta();
