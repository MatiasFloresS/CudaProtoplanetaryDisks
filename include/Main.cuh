#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cstdarg>
#include <iostream>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cufft.h>
#include "math_constants.h"
#include <cufft.h>
#include "cuda_runtime.h"


#include "Types.cuh"
#include "Kernels.cuh"
#include "SourceEuler.cuh"
#include "Force.cuh"
#include "Psys.cuh"
#include "Pframeforce.cuh"
#include "Init.cuh"
#include "Output.cuh"
#include "SideEuler.cuh"
#include "TransportEuler.cuh"
#include "Readfiles.cuh"
#include "Theo.cuh"



__host__ void ReadFile(char *ParameterFile);
__host__ void TellEverything();
__host__ void PrintUsage(char *argv);
__host__ void UpdateVelocitiesWithViscosity(float *RadialVelocity, float *AzimuthalVelocity, float *Rho, float DeltaT);
__host__ float CircumPlanetaryMasshost(float *dens, float xpl, float ypl);
__host__ void MultiplyPolarGridbyConstanthost(float *dens);
__host__ void Substep1host(float *dens, float *vrad, float *vtheta, float dt, int i);
__host__ void Substep2host(float *dens, float *energy, float dt);
__host__ void Substep3host(float *dens, float dt);
__host__ void ActualiseGasVrad(float *vrad, float *vradnew);
__host__ void ActualiseGasVtheta(float *vtheta, float *vthetanew);
__host__ float FViscosity(float r);
__host__ void FreeCuda();
__host__ void Substep1cudamalloc(float *vrad, float *vtheta);
__host__ void ActualiseGasEnergy(float *energy, float *energynew);
__host__ void FreeArrays();
__host__ void DeviceToHostcudaMemcpy(float *dens, float *energy, float *label, float *temperature, float *vrad, float *vtheta);
__host__ void ComputeViscousTerms (float *vrad, float *vtheta, float *dens, int option);
__host__ void Viscouscudamalloc();
__host__ void CreateArrays();
__host__ void Init_planetarysys_withSG(PlanetarySystem *sys);
__host__ void cudamalloc(float *label, float *dens, float *vrad, float *vtheta);
__host__ void executeExeC2Cforward();
__host__ void executeExeC2Cbackward();
