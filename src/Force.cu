#include "main.cuh"
#include "types.cuh"
#include "Force.cuh"
#include "SourceEuler.cuh"
#include "kernels.cuh"
using namespace std;

extern string OUTPUTDIR;
extern float ROCHESMOOTHING, THICKNESSSMOOTHING, FLARINGINDEX, *CellAbscissa, *CellOrdinate, *Surf, G, *forcesx, *forcesy;
extern bool RocheSmoothing;
extern int size_grid, blocksize, NRAD, NSEC, nsec2pot, nrad2pot;

__host__ void UpdateLog (Force *fc, PlanetarySystem *sys, float *dens, float *energy, int outputnb, float time2, int dimfxy)
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
    ComputeForce (fc, dens, x, y, smoothing, m, dimfxy);
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

  float *CellAbscissa_d, *CellOrdinate_d, *Surf_d, *forcesx_d, *forcesy_d, *dens_d;

  dim3 dimGrid( nsec2pot/blocksize, nrad2pot/blocksize );
  dim3 dimBlock( blocksize, blocksize );

  gpuErrchk(cudaMalloc((void**)&CellAbscissa_d, size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&CellOrdinate_d, size_grid*sizeof(float) ));
  gpuErrchk(cudaMalloc((void**)&Surf_d, NRAD*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&forcesx_d, size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&forcesy_d, size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&dens_d, size_grid*sizeof(float)));

  gpuErrchk(cudaMemcpy(CellAbscissa_d, CellAbscissa, size_grid*sizeof(float), cudaMemcpyHostToDevice ));
  gpuErrchk(cudaMemcpy(CellOrdinate_d, CellOrdinate, size_grid*sizeof(float), cudaMemcpyHostToDevice ));
  gpuErrchk(cudaMemcpy(Surf_d, Surf, NRAD*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(forcesx_d, forcesx, size_grid*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(forcesy_d, forcesy, size_grid*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(dens_d, dens, size_grid*sizeof(float), cudaMemcpyHostToDevice));

  ComputeForceKernel<<<dimGrid, dimBlock>>>(CellAbscissa_d, CellOrdinate_d, Surf_d, dens_d, x, y, rsmoothing, forcesx_d, forcesy_d,
    NSEC, NRAD, G);
  gpuErrchk(cudaDeviceSynchronize());
  gpuErrchk(cudaMemcpy(forcesx, forcesx_d, size_grid*sizeof(float), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(forcesy, forcesy_d, size_grid*sizeof(float), cudaMemcpyDeviceToHost));

  cudaFree(CellAbscissa_d );
  cudaFree(CellOrdinate_d );
  cudaFree(Surf_d);
  cudaFree(forcesx_d);
  cudaFree(forcesy_d);
  cudaFree(dens_d);


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
