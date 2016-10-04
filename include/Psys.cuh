__host__ int FindNumberOfPlanets (char *filename);
__host__ void ListPlanets (PlanetarySystem *sys);
__host__ PlanetarySystem *InitPlanetarySystem(char *filename);
__host__ PlanetarySystem *AllocPlanetSystem(int nb);
__host__ float GetPsysInfo (PlanetarySystem *sys, int action);
__host__ void FreePlanetary (PlanetarySystem *sys);
