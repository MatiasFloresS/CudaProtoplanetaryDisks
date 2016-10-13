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
#include "Planet.cuh"
#include "Viscosity.cuh"

using namespace std;

__host__ void ReadVariables (char *ParameterFile);
__host__ void TellEverything ();
__host__ void PrintUsage (char *argv);
__host__ void UpdateVelocitiesWithViscosity (float *VradInt, float *VthetaInt, float *dens, float DeltaT);
__host__ void MultiplyPolarGridbyConstant (float *Dens);
__host__ void FreeCuda ();
__host__ void FreeArrays (float *Dens, float *Vrad, float *Vtheta, float *energy, float *label);
__host__ void DeviceToHostcudaMemcpy (float *Dens, float *energy, float *label, float *Temperature, float *Vrad, float *Vtheta);
__host__ void Viscouscudamalloc ();
__host__ void CreateArrays ();
__host__ void Init_planetarysys_withSG (PlanetarySystem *sys);
__host__ void Cudamalloc (float *label, float *Dens, float *Vrad, float *Vtheta);
__host__ void ExecuteExeC2Cforward ();
__host__ void ExecuteExeC2Cbackward ();
__host__ void Compute_selfgravity (float *Dens, float *Vrad, float *Vtheta, float DeltaT, bool SGUpdate);
__host__ void Fft ();
__host__ void Fftmul ();
__host__ void Sg_Acc ();
__host__ void Update_sgvelocity (float DeltaT);
__host__ float CircumPlanetaryMass (float *Dens, PlanetarySystem *sys);
