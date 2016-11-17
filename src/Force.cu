#include "Main.cuh"

extern string OUTPUTDIR;

extern double ROCHESMOOTHING, THICKNESSSMOOTHING, FLARINGINDEX;

extern double *CellAbscissa, *CellOrdinate, *Surf, *forcesxi, *forcesyi, *forcesxo;
extern double *forcesyo;

extern double *Dens_d, *CellAbscissa_d, *CellOrdinate_d, *Surf_d;
extern double *fxi_d, *fxo_d, *fyi_d, *fyo_d;

extern double *Rmed, *Rmed_d;

extern int RocheSmoothing, size_grid, NRAD, NSEC, SelfGravity;

extern dim3 dimGrid2, dimBlock2;

__host__ void UpdateLog (Force *force, PlanetarySystem *sys, double *Dens, double *Energy, int TimeStep,
  double PhysicalTime, int dimfxy)
{
  FILE *out;
  double x, y, r, m, vx, vy, smoothing, a, rh;
  double *globalforce;
  char filename[500];
  char filename2[500];
  int i, nb;
  nb=sys->nb;
  string input;
  input = OUTPUTDIR +"tqwk";
  strncpy(filename, input.c_str(), sizeof(filename));
  filename[sizeof(filename)-1]=0;
  for (i = 0; i < nb; i++){
    x = sys->x[i];
    y = sys->y[i];
    vx = sys->vx[i];
    vy = sys->vy[i];
    r = sqrt(x*x+y*y);
    m = sys->mass[i];
    a = sqrt(x*x+y*y);
    rh = pow(m/3., 1./3.)*a+1e-15;

    if (RocheSmoothing) smoothing = r*pow(m/3.,1./3.)*ROCHESMOOTHING;
    else smoothing = Compute_smoothing(r);

    ComputeForce (force, Dens, x, y, smoothing, m, dimfxy, a, rh);

    globalforce = force->GlobalForce;
    sprintf (filename2, "%s%d.dat", filename,i);
    out = fopen(filename2, "a");
    if (out == NULL){
      fprintf(stderr, "Can't open %s\n",filename2 );
      fprintf(stderr, "Aborted.\n");
    }

    fprintf(out, "%d\t%.18g\t%.18g\t%.18g\t%.18g\t%.18g\t%.18g\t%.18g\t%.18g\t%.18g\n", TimeStep, \
      x*force->fy_inner-y*force->fx_inner, \
      x*force->fy_outer-y*force->fx_outer, \
      x*force->fy_ex_inner-y*force->fx_ex_inner, \
      x*force->fy_ex_outer-y*force->fx_ex_outer, \
      vx*force->fx_inner+vy*force->fy_inner , \
      vx*force->fx_outer+vy*force->fy_outer , \
      vx*force->fx_ex_inner+vy*force->fy_ex_inner , \
      vx*force->fx_ex_outer+vy*force->fy_ex_outer , PhysicalTime);
    fclose (out);

    if (!SelfGravity){
      for (int k = 0; k < dimfxy; k++) {
        sprintf( filename2, "%s%d_%d.dat", filename, i, k);
        out = fopen(filename2, "a");
        if (out == NULL){
          fprintf(stderr, "Can't open %s\n", filename2);
          fprintf(stderr, "Aborted.\n");
        }
        fprintf(out, "%d\t%.18g\t%.18g\t%.18g\t%.18g\t%.18g\n", TimeStep, \
          x*globalforce[2*dimfxy+k]-y*globalforce[k], \
          x*globalforce[3*dimfxy+k]-y*globalforce[dimfxy+k], \
          vx*globalforce[k]+vy*globalforce[2*dimfxy+k], \
          vx*globalforce[dimfxy+k]+vy*globalforce[3*dimfxy+k], PhysicalTime);
        fclose (out);
      }
    }
  }
}



__host__ Force *AllocateForce (int dimfxy)
{
  Force *force;
  double *globalforce;
  force = (Force *)malloc(sizeof(Force));
  globalforce = (double *)malloc(sizeof(double)*4*dimfxy);
  for (int i = 0; i < 4*dimfxy; i++)
    globalforce[i] = 0.0;
  force->GlobalForce = globalforce;
  return force;
}



__host__ void ComputeForce (Force *force, double *Dens, double x, double y, double rsmoothing,
  double mass, int dimfxy, double a, double rh)
{
  double *globalforce;
  int k;

  globalforce = force->GlobalForce;

  gpuErrchk(cudaMemset(fxi_d, 0, NRAD*NSEC*sizeof(double)));
  gpuErrchk(cudaMemset(fxo_d, 0, NRAD*NSEC*sizeof(double)));
  gpuErrchk(cudaMemset(fyi_d, 0, NRAD*NSEC*sizeof(double)));
  gpuErrchk(cudaMemset(fyo_d, 0, NRAD*NSEC*sizeof(double)));

  for (k = 0; k < dimfxy; k++) {
    ComputeForceKernel<<<dimGrid2, dimBlock2>>>(CellAbscissa_d, CellOrdinate_d, Surf_d, Dens_d, x, y, rsmoothing,
      NSEC, NRAD, a, Rmed_d, dimfxy, rh, fxi_d, fxo_d, fyi_d, fyo_d, k);
    gpuErrchk(cudaDeviceSynchronize());

    globalforce[k]            = DeviceReduce(fxi_d, NRAD*NSEC);
    globalforce[k + dimfxy]   = DeviceReduce(fxo_d, NRAD*NSEC);
    globalforce[k + 2*dimfxy] = DeviceReduce(fyi_d, NRAD*NSEC);
    globalforce[k + 3*dimfxy] = DeviceReduce(fyo_d, NRAD*NSEC);
  }

  force->fx_inner = globalforce[0];
  force->fx_ex_inner = globalforce[dimfxy-1];
  force->fx_outer = globalforce[dimfxy];
  force->fx_ex_outer = globalforce[2*dimfxy-1];
  force->fy_inner = globalforce[2*dimfxy];
  force->fy_ex_inner = globalforce[3*dimfxy-1];
  force->fy_outer = globalforce[3*dimfxy];
  force->fy_ex_outer = globalforce[4*dimfxy-1];
  force->GlobalForce = globalforce;
}

__host__ double Compute_smoothing(double r)
{
  double smooth;
  smooth = THICKNESSSMOOTHING * AspectRatioHost(r) * pow(r, 1.0+FLARINGINDEX);
  return smooth;
}

__host__ void FreeForce (Force *force)
{
  free (force->GlobalForce);
}
