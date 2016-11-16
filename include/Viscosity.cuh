__host__ void InitViscosityDevice ();
__host__ void InitViscosity ();
__host__ void UpdateVelocitiesWithViscosity(float *Vrad, float *Vtheta, float *Dens, float DeltaT);
__host__ void ComputeViscousTerms (float *Vrad, float *Vtheta, float *Dens);
__host__ float FViscosity (double r);
__host__ float AspectRatioHost(float r);
