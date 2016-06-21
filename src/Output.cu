#include "main.cuh"
#include "types.cuh"
using namespace std;

extern int NSEC, NRAD, NTOT, NINTERM;
extern float RMAX;
extern string OUTPUTDIR;

__host__ void WriteDim ()
{
  char filename[200];
  string input;
  input = OUTPUTDIR +"dims.dat";
  strncpy(filename, input.c_str(), sizeof(filename));
  filename[sizeof(filename)-1]=0;
  FILE *dim;
  int ntotterm;
  ntotterm = NTOT / NINTERM;
  if ((dim = fopen (filename, "w")) == NULL)
  {
    fprintf(stderr, "Unable to open %s. Program stopped\n", filename);
    exit(1);
  }
  fprintf(dim, "%d\t%d\t\t%d\t%d\t%f\t%d\t%d\t%d\n",0,0,0,0,RMAX, ntotterm, NRAD, NSEC);
  fclose(dim);
}

__host__ void EmptyPlanetSystemFile(PlanetarySystem *sys)
{
  FILE *output;
  char name[256];
  char name2[256];
  int n = sys->nb;
  string input;
  input = OUTPUTDIR +"planet";
  strncpy(name, input.c_str(), sizeof(name));
  name[sizeof(name)-1] = 0;

  for (int i = 0; i < 3; i++) {
    sprintf (name2, "%s%d.dat", name,i);
    output = fopen (name2, "w");
    if (output == NULL) {
      fprintf(stderr, "Can't write %s file. Aborting\n", name2);
    }
    fclose(output);
  }
}
