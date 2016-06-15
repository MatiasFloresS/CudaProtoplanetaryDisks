#include "main.cuh"
#include "types.cuh"
#include "SourceEuler.cuh"

__host__ void Initialization (float *dens,float *gas_v_rad,float * gas_v_theta,float * energy,float * gas_label, PlanetarySystem *sys)
{
  InitEuler (gas_v_rad, gas_v_theta, dens, energy);

}
