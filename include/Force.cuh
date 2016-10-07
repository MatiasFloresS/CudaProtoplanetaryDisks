__host__ void UpdateLog (Force *force, PlanetarySystem *sys, float *dens, float *energy, int TimeStep, float PhysicalTime,
   int dimfxy, int p);
__host__ Force *AllocateForce (int dimfxy);
__host__ void ComputeForce (Force *fc, float *dens, float x, float y, float rsmoothing, float mass, int dimfxy, int p,
   int i, float a, float rh);
__host__ float Compute_smoothing(float r);
__host__ void FreeForce (Force *force);
__host__ void Forcescudamalloc(int dimfxy);
