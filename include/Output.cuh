__host__ void WriteDim ();
__host__ void EmptyPlanetSystemFile (PlanetarySystem *sys);
__host__ void WriteBigPlanetSystemFile (PlanetarySystem *sys, int t);
__host__ void WriteBigPlanetFile (int TimeStep, int n);
__host__ void WritePlanetSystemFile (PlanetarySystem *sys, int TimeStep);
__host__ void WritePlanetFile (int TimeStep, int n);
__host__ void SendOutput (int TimeStep, double *Dens, double *Vrad, double *Vtheta, double *Energy, double *Label);
__host__ void WriteDiskPolar (double *array, char *inputname, int number);
