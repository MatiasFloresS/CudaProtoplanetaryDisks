#include "main.cuh"
#include "types.cuh"
#include "Force.cuh"
#include "SourceEuler.cuh"
using namespace std;

extern string OUTPUTDIR;
extern float ROCHESMOOTHING, THICKNESSSMOOTHING, FLARINGINDEX;
extern bool RocheSmoothing;
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

__host__ void ComputeForce (Force *fc, float *dens, float x, float y, float smoothing, float mass, int dimfxy)
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

  /*ComputeForceKernel<<<dimGrid, dimBlock>>>(CellAbscissa_d, CellOrdinate_d, Surf_d, dens_d, x, rsmoothing, dimfxy,
     mass, a, fxi, fxo, fyi, fyo, Rmed_d)*/

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
}

__host__ float compute_smoothing(float r)
{
  float smooth;
  smooth = THICKNESSSMOOTHING * AspectRatio(r) * powf(r, 1.0+FLARINGINDEX);
  return smooth;
}
