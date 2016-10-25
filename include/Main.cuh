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
#include "Interpret.cuh"
#include "Theo.cuh"
#include "Planet.cuh"
#include "Viscosity.cuh"
#include "RungeKutta.cuh"
#include "SgMain.cuh"
#include "LowTasks.cuh"

using namespace std;

__host__ void UpdateVelocitiesWithViscosity (float *VradInt, float *VthetaInt, float *dens, float DeltaT);
__host__ void MultiplyPolarGridbyConstant (float *Dens);
__host__ void FreeCuda ();
__host__ void FreeArrays (float *Dens, float *Vrad, float *Vtheta, float *energy, float *label);
__host__ void DeviceToHostcudaMemcpy (float *Dens, float *energy, float *label, float *Temperature, float *Vrad, float *Vtheta);
__host__ void Viscouscudamalloc ();
__host__ void CreateArrays ();
__host__ void Cudamalloc (float *label, float *Dens, float *Vrad, float *Vtheta);
