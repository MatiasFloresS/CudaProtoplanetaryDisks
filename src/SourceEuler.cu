#include "main.cuh"

using namespace std;

extern float RMAX, RMIN, PI;
extern int NRAD, NSEC;

extern float *invRmed, *invRinf, *invSurf, *invdiffRmed, *invdiffRsup;
extern float *invdiffSurf, *Rinf, *Rmed, *Rsup, *Surf;
extern string OUTPUTDIR;
extern int YES, LogGrid;

__host__ void FillPolar1DArray()
{
  FILE *input, *output;
  int i,ii;
  float drrsep;
  float temporary;
  string InputName, OutputName;
  drrsep = (RMAX-RMIN)/NRAD;
  InputName = OUTPUTDIR +"radii.dat";
  OutputName = OUTPUTDIR +"used_rad.dat";

  Rinf = (float *) malloc(sizeof(float)*(NRAD+1));
  Rmed = (float *) malloc(sizeof(float)*(NRAD+1));
  Rsup = (float *) malloc(sizeof(float)*(NRAD+1));
  Surf = (float *) malloc(sizeof(float)*(NRAD+1));
  invRinf = (float *) malloc(sizeof(float)*(NRAD+1));
  invSurf = (float *) malloc(sizeof(float)*(NRAD+1));
  invRmed = (float *) malloc(sizeof(float)*(NRAD+1));
  invdiffSurf = (float *) malloc(sizeof(float)*(NRAD+1));
  invdiffRsup = (float *) malloc(sizeof(float)*(NRAD+1));
  invdiffRmed = (float *) malloc(sizeof(float)*(NRAD+1));

  float Radii[129];
  float GlobalRmed[128];
  char inputcharname[100];
  strncpy(inputcharname, InputName.c_str(), sizeof(inputcharname));
  inputcharname[sizeof(inputcharname)-1]=0;

  input = fopen (inputcharname, "r");
  if (input == NULL) {
    printf("Warning : no `radii.dat' file found. Using default.\n");
    if (LogGrid == YES) {
      for (i = 0; i <= NRAD; i++) {
	         Radii[i] = RMIN*expf(i*logf(RMAX/RMIN)/NRAD);
      }
    } else {
      for (i = 0; i <= NRAD; i++) {
	         Radii[i] = RMIN+drrsep*i;
      }
    }
  } else {
    printf("Reading 'radii.dat' file.\n");
    for (i = 0; i <= NRAD; i++) {
      fscanf (input, "%f", &temporary);
      Radii[i] = (float)temporary;
    }
  }

  for (i = 0; i < NRAD; i++) {
    Rmed[i] = 2.0/3.0*(Radii[i+1]*Radii[i+1]*Radii[i+1]-Radii[i]*Radii[i]*Radii[i]);
    Rmed[i] = Rmed[i] / (Radii[i+1]*Radii[i+1]-Radii[i]*Radii[i]);
  }

  for (i = 0; i < NRAD; i++) {
    Rinf[i] = Radii[i];
    Rsup[i] = Radii[i+1];
    Rmed[i] = 2.0/3.0*(Rsup[i]*Rsup[i]*Rsup[i]-Rinf[i]*Rinf[i]*Rinf[i]);
    Rmed[i] = Rmed[i] / (Rsup[i]*Rsup[i]-Rinf[i]*Rinf[i]);
    Surf[i] = PI*(Rsup[i]*Rsup[i]-Rinf[i]*Rinf[i])/(float)NSEC;
    invRmed[i] = 1.0/Rmed[i];
    invSurf[i] = 1.0/Surf[i];
    invdiffRsup[i] = 1.0/(Rsup[i]-Rinf[i]);
    invRinf[i] = 1.0/Rinf[i];
  }

  Rinf[NRAD]=Radii[NRAD];
  for (i = 1; i < NRAD; i++) {
    invdiffRmed[i] = 1.0/(Rmed[i]-Rmed[i-1]);
  }

  char outputcharname[100];
  strncpy(outputcharname, OutputName.c_str(), sizeof(outputcharname));
  outputcharname[sizeof(outputcharname)-1]=0;
  output = fopen (outputcharname, "w");
  if (output == NULL) {
    printf ("Can't write %s.\nProgram stopped.\n", outputcharname);
    exit (1);
  }
  for (i = 0; i <= NRAD; i++) {
    fprintf (output, "%f\n", Radii[i]);
  }
  fclose (output);
  if (input != NULL) fclose (input);
}
