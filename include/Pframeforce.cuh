__host__ void InitGasDensity (float *Dens);
__host__ void InitGasEnergy (float *energy);
__host__ void ComputeIndirectTerm ();
__host__ void FillForcesArrays (PlanetarySystem *sys, float *Dens, float *energy);
