#include "Main.cuh"

extern string OUTPUTDIR;

extern int NSEC, NRAD, NTOT, NINTERM, Write_Temperature, Write_DivV, Write_Qplus, Write_Energy, \
Write_Density, Write_Velocity, IsDisk, YES, AdvecteLabel;

extern float RMAX, *Temperature, mdcp, exces_mdcp, OmegaFrame1, PhysicalTime, LostMass, *Qplus, \
*DivergenceVelocity;
static float Xplanet, Yplanet, VXplanet, VYplanet, MplanetVirtual;

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

  for (int i = 0; i < n; i++)
  {
    sprintf (name2, "%s%d.dat", name,i);
    output = fopen (name2, "w");
    if (output == NULL)
    {
      fprintf(stderr, "Can't write %s file. Aborting\n", name2);
      exit(1);
    }
    fclose(output);
  }
}

__host__ void WriteBigPlanetSystemFile(PlanetarySystem *sys, int t)
{
  int n = sys->nb;
  for (int i = 0; i < n; i++)
  {
    Xplanet = sys->x[i];
    Yplanet = sys->y[i];
    VXplanet = sys->vx[i];
    VYplanet = sys->vy[i];
    MplanetVirtual = sys->mass[i];
    WriteBigPlanetFile (t, i);
  }
}

__host__ void WritePlanetSystemFile (PlanetarySystem *sys, int TimeStep)
{
  int n = sys->nb;
  for (int i = 0; i < n; i++)
  {
    Xplanet = sys->x[i];
    Yplanet = sys->y[i];
    VXplanet = sys->vx[i];
    VYplanet = sys->vy[i];
    MplanetVirtual = sys->mass[i];
    WritePlanetFile(TimeStep, i);
  }
}

__host__ void WriteBigPlanetFile (int TimeStep, int n)
{
  FILE *output;
  char name[256];
  char name2[256];
  string input;
  input = OUTPUTDIR +"bigplanet";
  strncpy(name, input.c_str(), sizeof(name));
  name[sizeof(name)-1] = 0;
  sprintf (name2, "%s%d.dat", name,n);
  output = fopen (name2, "a");
  if (output == NULL)
  {
    fprintf(stderr, "Can't write 'bigplanet.dat' file. Aborting.\n");
    exit(1);
  }
  fprintf (output, "%d\t%.15f\t%.15f\t%.15f\t%.15f\t%.15f\t%.15f\t%.15f\t%.15f\t%.15f\t%.15f\n",
  TimeStep, Xplanet, Yplanet, VXplanet, VYplanet, MplanetVirtual, LostMass, PhysicalTime, OmegaFrame1, mdcp, exces_mdcp);
  fclose(output);
}

__host__ void WritePlanetFile (int TimeStep, int n)
{
  FILE *output;
  char name[256];
  char name2[256];
  string input;
  input = OUTPUTDIR + "planet";
  printf("Updating 'planet%d.dat'...",n);
  strncpy(name, input.c_str(), sizeof(name));
  name[sizeof(name)-1] = 0;
  sprintf (name2, "%s%d.dat", name, n);
  output = fopen (name2, "a");
  if (output == NULL)
  {
    fprintf(stderr, "Can't write 'planet%d,dat' file. Aborting.\n", n);
    exit(1);
  }
  fprintf (output, "%d\t%.15f\t%.15f\t%.15f\t%.15f\t%.15f\t%.15f\t%.15f\t%.15f\t%.15f\t%.15f\n",
  TimeStep, Xplanet, Yplanet, VXplanet, VYplanet, MplanetVirtual, LostMass, PhysicalTime, OmegaFrame1, mdcp, exces_mdcp);
  fclose(output);
  printf("done\n");

}

__host__ void SendOutput (int index, float *Dens, float *Vrad, float *Vtheta, float *Energy, float *Label)
{
  printf("\n*** OUTPUT %d ***\n", index);
  if (IsDisk == YES)
  {
    if (Write_Density == YES) WriteDiskPolar(Dens, "gasdensity", index);
    if (Write_Velocity == YES)
    {
      WriteDiskPolar(Vrad, "gasvrad", index);
      WriteDiskPolar(Vtheta, "gasvtheta", index);
    }
    if (Write_Energy == YES) WriteDiskPolar(Energy, "gasenergy", index);
    if (Write_Temperature == YES) WriteDiskPolar(Temperature, "gastemperature", index);
    if (Write_DivV == YES) WriteDiskPolar(DivergenceVelocity, "gasdivergence", index);
    if (Write_Qplus == YES) WriteDiskPolar(Qplus, "gasqplus", index);
    if (AdvecteLabel == YES) WriteDiskPolar(Label, "gaslabel", index);
  }
}

__host__ void WriteDiskPolar(float *array, char *inputname, int number)
{
  FILE *dump;
  char name[256];
  char name2[256];
  char nameinput[256];

  string input, input2, url;
  input = OUTPUTDIR + inputname + "/" + inputname;
  input2 = inputname;
  url = OUTPUTDIR + inputname;

  strncpy(name, input.c_str(), sizeof(name));
  strncpy(nameinput, input2.c_str(), sizeof(nameinput));
  name[sizeof(name)-1] = 0;
  nameinput[sizeof(nameinput)-1] = 0;
  sprintf (name2, "%s%d.dat", name, number);

  // mkdir ("/some/directory") ... etc

  struct stat st = {0};
  if (stat(url.c_str(), &st) == -1) mkdir(url.c_str(), 0700);

  dump = fopen(name2, "w");
  if (dump == NULL)
  {
    fprintf(stderr, "Unable to open '%s'\n", name2);
    exit(1);
  }
  printf("Writting '%s%d.dat'...\n", nameinput, number);
  fwrite(array, sizeof(float), NRAD*NSEC, dump);
  fclose(dump);
  printf("done\n");
}
