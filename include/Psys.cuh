__host__ int FindNumberOfPlanets (char *filename);
__host__ void ListPlanets (PlanetarySystem *sys);
__host__ PlanetarySystem *InitPlanetarySystem (char *filename);
__host__ PlanetarySystem *AllocPlanetSystem (int nb);
__host__ double GetPsysInfo (PlanetarySystem *sys, int action);
__host__ void FreePlanetary (PlanetarySystem *sys);
__host__ void RotatePsys (PlanetarySystem *sys, float angle);
