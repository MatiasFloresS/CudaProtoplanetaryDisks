#include "main.cuh"
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
  printf("%f\n", RMAX);
if ((dim = fopen (filename, "w")) == NULL)
  {
    fprintf(stderr, "Unable to open %s. Program stopped\n", filename);
    exit(1);
  }
  fprintf(dim, "%d\t%d\t\t%d\t%d\t%f\t%d\t%d\t%d\n",0,0,0,0,RMAX, ntotterm, NRAD, NSEC);
  fclose(dim);
}
