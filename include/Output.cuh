__host__ void WriteDim ();
__host__ void EmptyPlanetSystemFile(PlanetarySystem *sys);
__host__ void WriteBigPlanetSystemFile(PlanetarySystem *sys, int t);
__host__ void WriteBigPlanetFile (int TimeStep, int n);
__host__ void WritePlanetSystemFile (PlanetarySystem *sys, int TimeStep);
__host__ void WritePlanetFile (int TimeStep, int n);
__host__ void SendOutput (int TimeStep, float *dens, float *vrad, float *vtheta, float *energy, float *gas_label);
__host__ void WriteDiskPolar(float *array, char *inputname, int number);
