__host__ void InitGasDensity (float *Dens);
__host__ void InitGasEnergy (float *energy);
__host__ void ComputeIndirectTerm ();
__host__ void FillForcesArrays (PlanetarySystem *sys, float *Dens, float *energy);
__host__ void AdvanceSystemFromDisk (Force *force, float *Dens, float *energy, PlanetarySystem *sys, float dt);
__host__ void AdvanceSystemRK5 (PlanetarySystem *sys, float dt);
