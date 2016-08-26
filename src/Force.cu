#include "main.cuh"
#include "types.cuh"
#include "Force.cuh"
#include "SourceEuler.cuh"
#include "kernels.cuh"
using namespace std;

extern string OUTPUTDIR;
extern float ROCHESMOOTHING, THICKNESSSMOOTHING, FLARINGINDEX, *CellAbscissa, *CellOrdinate, *Surf, G, *forcesxi, *forcesyi;
extern float *forcesxo, *forcesyo, *Rmed, *Rmed_d;
extern bool RocheSmoothing;
extern int size_grid, blocksize, NRAD, NSEC, nsec2pot, nrad2pot;

__host__ void UpdateLog (Force *fc, PlanetarySystem *sys, int outputnb, float time2, int dimfxy)
{
  FILE *out;
  float x, y, r, m, vx, vy, smoothing;
  float *globalforce;
  char filename[500];
  char filename2[500];
  int nb=sys->nb;
  string input;
  input = OUTPUTDIR +"tqwk";
  strncpy(filename, input.c_str(), sizeof(filename));
  filename[sizeof(filename)-1]=0;
  for (int i = 0; i < nb; i++) {
    x = sys->x[i];
    y = sys->y[i];
    vx = sys->vx[i];
    vy = sys->vy[i];
    r = sqrtf(x*x+y*y);
    m = sys->mass[i];
    if (RocheSmoothing) smoothing = r*powf(m/3.,1./3.)*ROCHESMOOTHING;
    else smoothing = compute_smoothing(r);
    //ComputeForce (fc, x, y, smoothing, m, dimfxy);
    globalforce = fc->GlobalForce;
    sprintf (filename2, "%s%d.dat", filename,i);
    out = fopen(filename2, "a");
    if (out == NULL){
      fprintf(stderr, "Can't open %s\n",filename2 );
      fprintf(stderr, "Aborted.\n");
    }
    fprintf(out, "%d\t%.15f\t%.15f\t%.15f\t%.15f\t%.15f\t%.15f\t%.15f\t%.15f\t%.15f\n", outputnb, \
      x*fc->fy_inner-y*fc->fx_inner, \
      x*fc->fy_outer-y*fc->fx_outer, \
      x*fc->fy_ex_inner-y*fc->fx_ex_inner, \
      x*fc->fy_ex_outer-y*fc->fx_ex_outer, \
      vx*fc->fx_inner+vy*fc->fy_inner , \
      vx*fc->fx_outer+vy*fc->fy_outer , \
      vx*fc->fx_ex_inner+vy*fc->fy_ex_inner , \
      vx*fc->fx_ex_outer+vy*fc->fy_ex_outer , time2);
    fclose (out);
  }
}

__host__ Force *AllocateForce (int dimfxy)
{
  Force *force;
  float *globalforce;
  force = (Force *)malloc(sizeof(Force));
  globalforce = (float *)malloc(sizeof(float)*4*dimfxy);
  for (int i = 0; i < 4*dimfxy; i++) globalforce[i] = 0.0;
  force->GlobalForce = globalforce;
  return force;
}

__host__ void ComputeForce (Force *fc, float *dens, float x, float y, float rsmoothing, float mass, int dimfxy)
{
  float *globalforce, *localforce, *fxi, *fxo, *fyi, *fyo;
  float a, rh;
  globalforce = fc->GlobalForce;
  localforce =(float *)malloc(sizeof(float)*4*dimfxy);
  fxi = (float *)malloc(sizeof(float)*dimfxy);
  fxo = (float *)malloc(sizeof(float)*dimfxy);
  fyi = (float *)malloc(sizeof(float)*dimfxy);
  fyo = (float *)malloc(sizeof(float)*dimfxy);

  a = sqrtf(x*x+y*y);
  rh = powf(mass/3., 1./3.)*a+1e-15;
  for (int k = 0; k < dimfxy; k++) {
      fxi[k] = 0.;
      fxo[k] = 0.;
      fyi[k] = 0.;
      fyo[k] = 0.;
  }
  for (int k = 0; k < 4*dimfxy; k++) {
    localforce[k] = 0.;
    globalforce[k] = 0.;
  }

  float *CellAbscissa_d, *CellOrdinate_d, *Surf_d, *forcesxi_d, *forcesyi_d, *forcesxo_d, *forcesyo_d, *dens_d;

  dim3 dimGrid( nsec2pot/blocksize, nrad2pot/blocksize );
  dim3 dimBlock( blocksize, blocksize );

  gpuErrchk(cudaMalloc((void**)&CellAbscissa_d, size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&CellOrdinate_d, size_grid*sizeof(float) ));
  gpuErrchk(cudaMalloc((void**)&Surf_d, NRAD*sizeof(float)));

  gpuErrchk(cudaMalloc(&forcesxi_d, size_grid*sizeof(float)));
  gpuErrchk(cudaMemset(forcesxi_d, 0, size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc(&forcesxo_d, size_grid*sizeof(float)));
  gpuErrchk(cudaMemset(forcesxo_d, 0, size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc(&forcesyi_d, size_grid*sizeof(float)));
  gpuErrchk(cudaMemset(forcesyi_d, 0, size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc(&forcesyo_d, size_grid*sizeof(float)));
  gpuErrchk(cudaMemset(forcesyo_d, 0, size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc(&dens_d, size_grid*sizeof(float)));

  gpuErrchk(cudaMemcpy(CellAbscissa_d, CellAbscissa, size_grid*sizeof(float), cudaMemcpyHostToDevice ));
  gpuErrchk(cudaMemcpy(CellOrdinate_d, CellOrdinate, size_grid*sizeof(float), cudaMemcpyHostToDevice ));
  gpuErrchk(cudaMemcpy(Surf_d, Surf, NRAD*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(forcesxi_d, forcesxi, size_grid*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(forcesyi_d, forcesyi, size_grid*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(forcesxo_d, forcesxo, size_grid*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(forcesyo_d, forcesyo, size_grid*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(dens_d, dens, size_grid*sizeof(float), cudaMemcpyHostToDevice));

  ComputeForceKernel<<<dimGrid, dimBlock>>>(CellAbscissa_d, CellOrdinate_d, Surf_d, dens_d, x, y, rsmoothing, forcesxi_d, forcesyi_d,
    forcesxo_d, forcesyo_d, NSEC, NRAD, G, a, Rmed_d);
  gpuErrchk(cudaDeviceSynchronize());
  gpuErrchk(cudaMemcpy(forcesxi, forcesxi_d, size_grid*sizeof(float), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(forcesyi, forcesyi_d, size_grid*sizeof(float), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(forcesxo, forcesxo_d, size_grid*sizeof(float), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(forcesyo, forcesyo_d, size_grid*sizeof(float), cudaMemcpyDeviceToHost));

  cudaFree(CellAbscissa_d );
  cudaFree(CellOrdinate_d );
  cudaFree(Surf_d);
  cudaFree(forcesxi_d);
  cudaFree(forcesyi_d);
  cudaFree(forcesxo_d);
  cudaFree(forcesyo_d);
  cudaFree(dens_d);

  //printf("%f %f %f %f\n",forcesxi[0], forcesxo[0],forcesyi[0], forcesyo[0]);
  // aca llamo al kernel
  fc->fx_inner = globalforce[0];
  fc->fx_ex_inner = globalforce[dimfxy-1];
  fc->fx_outer = globalforce[dimfxy];
  fc->fx_ex_outer = globalforce[2*dimfxy-1];
  fc->fy_inner = globalforce[2*dimfxy];
  fc->fy_ex_inner = globalforce[3*dimfxy-1];
  fc->fy_outer = globalforce[3*dimfxy];
  fc->fy_ex_outer = globalforce[4*dimfxy-1];
  fc->GlobalForce = globalforce;

  free(localforce);
  free(fxi);
  free(fxo);
  free(fyi);
  free(fyo);
}

__host__ float compute_smoothing(float r)
{
  float smooth;
  smooth = THICKNESSSMOOTHING * AspectRatio(r) * powf(r, 1.0+FLARINGINDEX);
  return smooth;
}

__host__ void FreeForce (Force *force)
{
  free (force->GlobalForce);
}
