#include <stdio.h>
#include <math.h>
#include <cstdarg>
#include <iostream>
#include "string.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

__host__ void ReadFile(char *ParameterFile);
__host__ void TellEverything();
__host__ void PrintUsage(char *argv);
__host__ void UpdateVelocitiesWithViscosity(float *RadialVelocity, float *AzimuthalVelocity, float *Rho, float DeltaT);
__host__ float CircumPlanetaryMasshost(float *dens, float xpl, float ypl);
__host__ void MultiplyPolarGridbyConstanthost(float *dens);
__host__ void Substep1host(float *dens, float *vrad, float *vtheta, float dt, int i);
__host__ void Substep2host(float *dens, float *energy, float dt, int i);
__host__ void Substep3host(float *dens, float dt, int p);
__host__ void ActualiseGasVrad(float *vrad, float *vradnew);
__host__ void ActualiseGasVtheta(float *vtheta, float *vthetanew);
__host__ float FViscosity(float r);
__host__ void FreeCuda();
__host__ void Substep1cudamalloc();
__host__ void Substep2cudamalloc();
__host__ void Substep3cudamalloc();
__host__ void ActualiseGasEnergy(float *energy, float *energynew);
__host__ void FreeArrays(float *dens, float *energy, float *gas_label);
__host__ void DeviceToHostcudaMemcpy(float *dens, float *energy, float *gas_label, float *temperature, float *vrad, float *vtheta);
__host__ void ComputeViscousTerms (float *vrad, float *vtheta, float *dens, int i, int option);
__host__ void Viscouscudamalloc();
