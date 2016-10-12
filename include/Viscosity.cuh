__host__ void InitViscosityDevice ();
__host__ void InitViscosity ();
__host__ void UpdateVelocitiesWithViscosity(float *vrad, float *vtheta, float *dens, float DeltaT);
__host__ float AspectRatio (float r);
__host__ void ComputeViscousTerms (float *vrad, float *vtheta, float *dens, int option);
__host__ float FViscosity (float r);
