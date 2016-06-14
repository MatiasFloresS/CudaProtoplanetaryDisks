#include "types.cuh"

int FindNumberOfPlanets (char *filename);
void ListPlanets (PlanetarySystem *sys);
PlanetarySystem *InitPlanetarySystem(char *filename);
PlanetarySystem *AllocPlanetSystem(int nb);
