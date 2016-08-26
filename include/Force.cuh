__host__ void UpdateLog (Force *fc, PlanetarySystem *psys,  int outputnb, float time, int dimfxy);
__host__ Force *AllocateForce (int dimfxy);
__host__ void ComputeForce (Force *fc, float *dens, float x, float y, float rsmoothing, float mass, int dimfxy);
__host__ float compute_smoothing(float r);
__host__ void FreeForce (Force *force);
