#include "Main.cuh"
#include "Types.cuh"
#include "Force.cuh"
#include "SourceEuler.cuh"
#include "Kernels.cuh"
using namespace std;

extern string OUTPUTDIR;
float *globalforce, *localforce;

extern float ROCHESMOOTHING, THICKNESSSMOOTHING, FLARINGINDEX, *CellAbscissa, *CellOrdinate, *Surf, G, \
*forcesxi, *forcesyi, *forcesxo, *forcesyo, *Rmed, *Rmed_d, *dens_d, *CellAbscissa_d, *CellOrdinate_d, *Surf_d,
*forcesxi_d, *forcesyi_d, *forcesxo_d, *forcesyo_d;

extern bool RocheSmoothing;
extern int size_grid, blocksize, NRAD, NSEC, nsec2pot, nrad2pot;
extern dim3 dimGrid2, dimBlock2;

__host__ void UpdateLog (Force *fc, float *dens, PlanetarySystem *sys, int outputnb, float time2, int dimfxy, int p)
{
  FILE *out;
  float x, y, r, m, vx, vy, smoothing, a, rh;
  float *globalforce;
  char filename[500];
  char filename2[500];
  int nb=sys->nb;
  string input;
  input = OUTPUTDIR +"tqwk";
  strncpy(filename, input.c_str(), sizeof(filename));
  filename[sizeof(filename)-1]=0;
  for (int i = 0; i < nb; i++)
  {
    x = sys->x[i];
    y = sys->y[i];
    vx = sys->vx[i];
    vy = sys->vy[i];
    r = sqrtf(x*x+y*y);
    m = sys->mass[i];
    a = sqrtf(x*x+y*y);
    rh = powf(m/3., 1./3.)*a+1e-15;

    if (RocheSmoothing) smoothing = r*powf(m/3.,1./3.)*ROCHESMOOTHING;
    else smoothing = Compute_smoothing(r);

    ComputeForce (fc, dens, x, y, smoothing, m, dimfxy, p, i, a, rh);

    globalforce = fc->GlobalForce;
    sprintf (filename2, "%s%d.dat", filename,i);
    out = fopen(filename2, "a");
    if (out == NULL)
    {
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

__host__ void ComputeForce (Force *fc, float *dens, float x, float y, float rsmoothing, float mass, int dimfxy, int p,
  int i, float a, float rh)
{
  globalforce = fc->GlobalForce;

  gpuErrchk(cudaMemset(forcesxi_d, 0, dimfxy*sizeof(float)));
  gpuErrchk(cudaMemset(forcesxo_d, 0, dimfxy*sizeof(float)));
  gpuErrchk(cudaMemset(forcesyi_d, 0, dimfxy*sizeof(float)));
  gpuErrchk(cudaMemset(forcesyo_d, 0, dimfxy*sizeof(float)));

  ComputeForceKernel<<<dimGrid2, dimBlock2>>>(CellAbscissa_d, CellOrdinate_d, Surf_d, dens_d, x, y, rsmoothing, forcesxi_d, forcesyi_d,
    forcesxo_d, forcesyo_d, NSEC, NRAD, G, a, Rmed_d, dimfxy, rh);
  gpuErrchk(cudaDeviceSynchronize());

  gpuErrchk(cudaMemcpy(forcesxi, forcesxi_d, dimfxy*sizeof(float), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(forcesyi, forcesyi_d, dimfxy*sizeof(float), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(forcesxo, forcesxo_d, dimfxy*sizeof(float), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(forcesyo, forcesyo_d, dimfxy*sizeof(float), cudaMemcpyDeviceToHost));

  for (int k = 0; k < dimfxy; k++)
  {
    globalforce[k]            = forcesxi[k];
    globalforce[k + dimfxy]   = forcesxo[k];
    globalforce[k + 2*dimfxy] = forcesyi[k];
    globalforce[k + 3*dimfxy] = forcesyo[k];
  }

  fc->fx_inner = globalforce[0];
  fc->fx_ex_inner = globalforce[dimfxy-1];
  fc->fx_outer = globalforce[dimfxy];
  fc->fx_ex_outer = globalforce[2*dimfxy-1];
  fc->fy_inner = globalforce[2*dimfxy];
  fc->fy_ex_inner = globalforce[3*dimfxy-1];
  fc->fy_outer = globalforce[3*dimfxy];
  fc->fy_ex_outer = globalforce[4*dimfxy-1];
  fc->GlobalForce = globalforce;

}

__host__ float Compute_smoothing(float r)
{
  float smooth;
  smooth = THICKNESSSMOOTHING * AspectRatio(r) * powf(r, 1.0+FLARINGINDEX);
  return smooth;
}

__host__ void FreeForce (Force *force)
{
  free (force->GlobalForce);
}
