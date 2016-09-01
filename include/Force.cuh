__host__ void UpdateLog (Force *fc, float *dens, PlanetarySystem *psys,  int outputnb, float time, int dimfxy, int p);
__host__ Force *AllocateForce (int dimfxy);
__host__ void ComputeForce (Force *fc, float *dens, float x, float y, float rsmoothing, float mass, int dimfxy, int p,
   int i, float a, float rh);
__host__ float compute_smoothing(float r);
__host__ void FreeForce (Force *force);
__host__ void Forcescudamalloc(int dimfxy);
