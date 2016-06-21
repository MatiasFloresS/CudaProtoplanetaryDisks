#include "main.cuh"
#include "types.cuh"
#include "SourceEuler.cuh"
#include "kernels.cuh"
#include "Output.cuh"

extern float *Rmed;
extern int NRAD, NSEC, nsec2pot, blocksize, nrad2pot, size_grid;

__host__ void Initialization (float *dens,float *gas_v_rad,float * gas_v_theta,float * energy,float * gas_label, PlanetarySystem *sys)
{
  float rp, rhill, xp, yp;
  xp = sys->x[0];
  yp = sys->y[0];
  rp = sqrt(xp*xp+yp*yp);
  rhill = rp * pow(sys->mass[0]/3., 1./3);

  dim3 dimGrid( nsec2pot/blocksize, nrad2pot/blocksize );
  dim3 dimBlock( blocksize, blocksize );

  float *gas_label_d, *Rmed_d;

  InitEuler (gas_v_rad, gas_v_theta, dens, energy);


    /* <---------     InitLabel()        --------> */

  gpuErrchk(cudaMalloc((void**)&gas_label_d, size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&Rmed_d,NRAD*sizeof(float)));

	gpuErrchk(cudaMemcpy(gas_label_d, gas_label, size_grid*sizeof(float), cudaMemcpyHostToDevice ));
  gpuErrchk(cudaMemcpy(Rmed_d, Rmed, NRAD*sizeof(float), cudaMemcpyHostToDevice));

  InitLabel<<<dimGrid, dimBlock>>>(gas_label_d, xp, yp, rhill, Rmed_d, NRAD, NSEC);
  gpuErrchk(cudaDeviceSynchronize());
  gpuErrchk(cudaMemcpy(gas_label, gas_label_d, size_grid*sizeof(float), cudaMemcpyDeviceToHost));

  cudaFree(gas_label_d);
  cudaFree(Rmed_d);

  /* <---------     InitLabel()        --------> */

  WriteDim();

}
