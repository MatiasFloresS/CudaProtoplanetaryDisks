__host__ void InitViscosityDevice ();
__host__ void InitViscosity ();
__host__ void UpdateVelocitiesWithViscosity(double *Vrad, double *Vtheta, double *Dens, double DeltaT);
__host__ void ComputeViscousTerms (double *Vrad, double *Vtheta, double *Dens);
__host__ double FViscosity (double r);
__host__ double AspectRatioHost(double r);
