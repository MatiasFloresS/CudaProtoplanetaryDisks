__host__ void UpdateLog (Force *fc, PlanetarySystem *psys, float *dens, float *energy, int outputnb, float time, int dimfxy);
__host__ Force *AllocateForce (int dimfxy);
__host__ void ComputeForce (Force *fc, float *dens, float x, float y, float smoothing, float mass, int dimfxy);
__host__ float compute_smoothing(float r);
