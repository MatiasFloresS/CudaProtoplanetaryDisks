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

#include "Fondam.cuh"
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

__host__ void UpdateVelocitiesWithViscosity (double *VradInt, double *VthetaInt, double *dens, double DeltaT);
__host__ void MultiplyPolarGridbyConstant (double *Dens);
__host__ void FreeCuda ();
__host__ void FreeArrays (double *Dens, double *Vrad, double *Vtheta, double *energy, double *label);
__host__ void DeviceToHostcudaMemcpy (double *Dens, double *energy, double *label, double *Temperature, double *Vrad, double *Vtheta);
__host__ void Viscouscudamalloc ();
__host__ void CreateArrays ();
__host__ void Cudamalloc (double *label, double *Dens, double *Vrad, double *Vtheta);
__host__ void binFile(double *array, int sizeArray, char *name);
