__host__ void InitGasDensity (double *Dens);
__host__ void InitGasEnergy (double *Energy);
__host__ void ComputeIndirectTerm ();
__host__ void FillForcesArrays (PlanetarySystem *sys, double *Dens, double *Energy);
__host__ void AdvanceSystemFromDisk (Force *force, double *Dens, double *Energy, PlanetarySystem *sys, double dt);
__host__ void AdvanceSystemRK5 (PlanetarySystem *sys, double dt);
__host__ void InitGasVelocities (double *Vrad, double *Vtheta);
__host__ void InitVelocities (double *Vrad, double* Vtheta);
