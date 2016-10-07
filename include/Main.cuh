#include <stdio.h>
#include <math.h>
#include <cstdarg>
#include <iostream>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cufft.h>
#include "math_constants.h"
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

using namespace std;

__host__ void ReadVariables (char *ParameterFile);
__host__ void TellEverything ();
__host__ void PrintUsage (char *argv);
__host__ void UpdateVelocitiesWithViscosity (float *RadialVelocity, float *AzimuthalVelocity, float *Rho, float DeltaT);
__host__ void MultiplyPolarGridbyConstant (float *dens);
__host__ void FreeCuda ();
__host__ void FreeArrays ();
__host__ void DeviceToHostcudaMemcpy (float *dens, float *energy, float *label, float *temperature, float *vrad, float *vtheta);
__host__ void ComputeViscousTerms (float *vrad, float *vtheta, float *dens, int option);
__host__ void Viscouscudamalloc ();
__host__ void CreateArrays ();
__host__ void Init_planetarysys_withSG (PlanetarySystem *sys);
__host__ void Cudamalloc (float *label, float *dens, float *vrad, float *vtheta);
__host__ void ExecuteExeC2Cforward ();
__host__ void ExecuteExeC2Cbackward ();
__host__ void Compute_selfgravity (float *dens, float *vrad, float *vtheta, float DeltaT, bool SGUpdate);
__host__ void Fft ();
__host__ void Fftmul ();
__host__ void Sg_Acc ();
__host__ void Update_sgvelocity (float DeltaT);
__host__ float FViscosity (float r);
__host__ float CircumPlanetaryMass (float *dens, PlanetarySystem *sys);
