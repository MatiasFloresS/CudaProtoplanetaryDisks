__host__ void InitViscosityDevice ();
__host__ void InitViscosity ();
__host__ void UpdateVelocitiesWithViscosity(float *Vrad, float *Vtheta, float *Dens, float DeltaT);
__host__ float AspectRatio (float r);
__host__ void ComputeViscousTerms (float *Vrad, float *Vtheta, float *Dens, int option);
__host__ float FViscosity (float r);
