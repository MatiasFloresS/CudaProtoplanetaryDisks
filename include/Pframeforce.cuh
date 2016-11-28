__host__ void InitGasDensity (float *Dens);
__host__ void InitGasEnergy (float *Energy);
__host__ void ComputeIndirectTerm ();
__host__ void FillForcesArrays (PlanetarySystem *sys, float *Dens, float *Energy);
__host__ void AdvanceSystemFromDisk (Force *force, float *Dens, float *Energy, PlanetarySystem *sys, float dt);
__host__ void AdvanceSystemRK5 (PlanetarySystem *sys, double dt);
__host__ void InitGasVelocities (float *Vrad, float *Vtheta);
__host__ void InitVelocities (float *Vrad, float* Vtheta);
