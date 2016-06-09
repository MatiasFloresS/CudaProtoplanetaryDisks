#include "main.cuh"

using namespace std;
extern float DT;
__host__ void PrintUsage (char *execname)
{
  printf("Usage : %s [-abcdeimnptvz] [-(0-9)] [-s number] [-f scaling] parameters file\n", execname);
  printf("\n-a : Monitor mass and angular momentum at each timestep\n");
  printf("-b : Adjust azimuthal velocity to impose strict centrifugal balance at t=0\n");
  printf("-c : Sloppy CFL condition (checked at each DT, not at each timestep)\n");
  printf("-d : Print some debugging information on 'stdout' at each timestep\n");
  printf("-e : Activate EU test problem torque file output\n");
  printf("-f : Scale density array by 'scaling'. Useful to increase/decrease\n");
  printf("     disk surface density after a restart, for instance.            \n");
  printf("-i : tabulate Sigma profile as given by restart files\n");
  printf("-m : Merge output files from different CPUs\n");
  printf("-n : Disable simulation. The program just reads parameters file\n");
  printf("-o : Overrides output directory of input file.\n");
  printf("-p : Give profiling information at each time step\n");
  printf("-s : Restart simulation, taking #'number' files as initial conditions\n");
  printf("-t : Monitor CPU time usage at each time step\n");
  printf("-v : Verbose mode. Tells everything about parameters file\n");
  printf("-z : fake sequential built when evaluating sums on HD meshes\n");
  printf("-(0-9) : only write initial (or restart) HD meshes,\n");
  printf("     proceed to the next nth output and exit\n");
  printf("     This option must stand alone on one switch (-va -4 is legal, -v4a is not)\n");
  exit(1);

}


__host__ void ReadFile(char *ParameterFile)
{
    //printf("file : %s\n",ParameterFile);
    char nm[300], s[350], stringval[290];
    char *s1;
    int success, valuei;
    float valuef, temp;
    FILE *input;
    input = fopen(ParameterFile, "r");
    if (input == NULL)
    {
      printf("Unable to read %s . Program stopped. \n", ParameterFile);
      exit(1);
    }
    while (fgets(s, 349, input) != NULL) {
      success = sscanf(s, "%s ", nm);
      if ((nm[0] != '#') && (success == 1)) {  /* # begins a comment line */
        s1 = s + strlen(nm);
        sscanf(s1 + strspn(s1, "\t :=>_"), "%f", &temp);
        sscanf(s1 + strspn(s1, "\t :=>_"), "%289s ", stringval);
        valuef = (float) temp;
        valuei = (int) temp;
        for (int i = 0; i < strlen(nm); i++) {
  	nm[i] = (char) toupper(nm[i]);
        }
      }
    }
    fclose(input);

    for (int i = 0; i < strlen(nm); i++) {
        printf("%c\n", nm[i]);
    }
}
/*
void ReadVariables(filename)
     char *filename;
{
  char            nm[300], s[350],stringval[290];
  char           *s1;
  float           temp;
  real            valuer;
  int             i, found, valuei, success, type;
  int            *ptri;
  real           *ptrr;
  FILE           *input;

  while (fgets(s, 349, input) != NULL) {
    success = sscanf(s, "%s ", nm);
    if ((nm[0] != '#') && (success == 1)) {  /* # begins a comment line *//*
      s1 = s + strlen(nm);
      sscanf(s1 + strspn(s1, "\t :=>_"), "%f", &temp);
      sscanf(s1 + strspn(s1, "\t :=>_"), "%289s ", stringval);
      valuer = (real) temp;
      valuei = (int) temp;
      for (i = 0; i < strlen(nm); i++) {
	nm[i] = (char) toupper(nm[i]);
      }
      found = NO;
      for (i = 0; i < VariableIndex; i++) {
	if (strcmp(nm, VariableSet[i].name) == 0) {
	  if (VariableSet[i].read == YES) {
	    mastererr("Warning : %s defined more than once.\n", nm);
	  }
	  found = YES;
	  VariableSet[i].read = YES;
	  ptri = (int *) (VariableSet[i].variable);
	  ptrr = (real *) (VariableSet[i].variable);
	  if (VariableSet[i].type == INT) {
	    *ptri = valuei;
	  } else if (VariableSet[i].type == REAL) {
	    *ptrr = valuer;
	  } else if (VariableSet[i].type == STRING) {
	    strcpy (VariableSet[i].variable, stringval);
	  }
	}
      }
      if (found == NO) {
	mastererr("Warning : variable %s defined but non-existent in code.\n", nm);
      }
    }
  }

  found = NO;
  for (i = 0; i < VariableIndex; i++) {
    if ((VariableSet[i].read == NO) && (VariableSet[i].necessary == YES)) {
      if (found == NO) {
	mastererr("Fatal error : undefined mandatory variable(s):\n");
	found = YES;
      }
      mastererr("%s\n", VariableSet[i].name);
    }
    if (found == YES)
      prs_exit(1);

  }
  found = NO;
  for (i = 0; i < VariableIndex; i++) {
    if (VariableSet[i].read == NO) {
      if (found == NO) {
	mastererr("Secondary variables omitted :\n");
	found = YES;
      }
      if ((type = VariableSet[i].type) == REAL)
	mastererr("%s ;\t Default Value : %.5g\n", VariableSet[i].name, *((real *) VariableSet[i].variable));
      if (type == INT)
	mastererr("%s ;\t Default Value : %d\n", VariableSet[i].name, *((int *) VariableSet[i].variable));
      if (type == STRING)
	mastererr("%s ;\t Default Value : %s\n", VariableSet[i].name, VariableSet[i].variable);
    }
  }
  if ((*ADVLABEL == 'y') || (*ADVLABEL == 'Y')) AdvecteLabel = YES;
  if ((*OUTERSOURCEMASS == 'y') || (*OUTERSOURCEMASS == 'Y')) OuterSourceMass = YES;
  if ((*TRANSPORT == 's') || (*TRANSPORT == 'S')) FastTransport = NO;
  if ((*OPENINNERBOUNDARY == 'O') || (*OPENINNERBOUNDARY == 'o')) OpenInner = YES;
  if ((*OPENINNERBOUNDARY == 'N') || (*OPENINNERBOUNDARY == 'n')) NonReflecting = YES;
   if ((*OPENINNERBOUNDARY == 'E') || (*OPENINNERBOUNDARY == 'e')) Evanescent = YES;
  if ((*GRIDSPACING == 'L') || (*GRIDSPACING == 'l')) LogGrid = YES;
  if ((*DISK == 'N') || (*DISK == 'n')) IsDisk = NO;
  if ((*FRAME == 'C') || (*FRAME == 'c')) Corotating = YES;
  if ((*FRAME == 'G') || (*FRAME == 'g')) {
    Corotating = YES;
    GuidingCenter = YES;
  }
  if ((*WRITEVELOCITY == 'N') || (*WRITEVELOCITY == 'n')) Write_Velocity = NO;
  if ((*WRITEDENSITY == 'N') || (*WRITEDENSITY == 'n')) Write_Density = NO;
  if ((*WRITEENERGY == 'Y') || (*WRITEENERGY == 'y')) Write_Energy = YES;
  if ((*WRITETEMPERATURE == 'Y') || (*WRITETEMPERATURE == 'y')) Write_Temperature = YES;
  if ((*WRITEDIVV == 'Y') || (*WRITEDIVV == 'y')) Write_DivV = YES;
  if ((*WRITEQPLUS == 'Y') || (*WRITEQPLUS == 'y')) Write_Qplus = YES;
  if ((*INDIRECTTERM == 'N') || (*INDIRECTTERM == 'n')) Indirect_Term = NO;
  if ((*SELFGRAVITY == 'Y') || (*SELFGRAVITY == 'y')) SelfGravity = YES;
  if ((*SELFGRAVITY == 'Z') || (*SELFGRAVITY == 'z')) {
    SelfGravity = YES;
    SGZeroMode = YES;
  }
  if ((*ZMPLUS == 'Y') || (*ZMPLUS == 'y')) ZMPlus = YES;
  if ( (ZMPlus) && (!SGZeroMode) ) {
    masterprint ("This is not very meaningfull to involve the anisotropic pressure model (ZMPlus=Yes) without taking into account the axisymmetric component of the disk self-gravity. I decided to put ZMPlus = No. Please check again!");
    ZMPlus = NO;
  }
  if ((*ADIABATIC == 'Y') || (*ADIABATIC == 'y')) {
    Adiabatic = YES;
    Write_Temperature = YES;
  }
  if ((*COOLING == 'Y') || (*COOLING == 'y')) Cooling = YES;
  if ( (Adiabatic) && (ADIABATICINDEX == 1) ) {
    masterprint ("You cannot have Adiabatic = YES and AdiabatcIndex = 1. I decided to put Adiabatic = No, to simulate a locally isothermal equation of state. Please check that it what you really wanted to do!\n");
    Adiabatic = NO;
  }
  if ((*WRITEENERGY == 'N') || (*WRITEENERGY == 'n')) Write_Energy = NO;
  if ((*EXCLUDEHILL == 'Y') || (*EXCLUDEHILL == 'y')) ExcludeHill = YES;
  if ((*CICPLANET == 'Y') || (*CICPLANET == 'y')) CICPlanet = YES;
  if ((*FORCEDCIRCULAR == 'Y') || (*FORCEDCIRCULAR == 'y')) ForcedCircular = YES;
  if ((ALPHAVISCOSITY != 0.0) && (VISCOSITY != 0.0)) {
    mastererr ("You cannot use at the same time\n");
    mastererr ("VISCOSITY and ALPHAVISCOSITY.\n");
    mastererr ("Edit the parameter file so as to remove\n");
    mastererr ("one of these variables and run again.\n");
    prs_exit (1);
  }
  if (ALPHAVISCOSITY != 0.0) {
    ViscosityAlpha = YES;
    masterprint ("Viscosity is of alpha type\n");
  }
  if ((THICKNESSSMOOTHING != 0.0) && (ROCHESMOOTHING != 0.0)) {
    mastererr ("You cannot use at the same time\n");
    mastererr ("`ThicknessSmoothing' and `RocheSmoothing'.\n");
    mastererr ("Edit the parameter file so as to remove\n");
    mastererr ("one of these variables and run again.\n");
    prs_exit (1);
  }
  if ((THICKNESSSMOOTHING <= 0.0) && (ROCHESMOOTHING <= 0.0)) {
    mastererr ("A non-vanishing potential smoothing length is required.\n");
    mastererr ("Please use either of the following variables:\n");
    mastererr ("`ThicknessSmoothing' *or* `RocheSmoothing'.\n");
    mastererr ("before launching the run again.\n");
    prs_exit (1);
  }
  if (ROCHESMOOTHING != 0.0) {
    RocheSmoothing = YES;
    masterprint ("Planet potential smoothing scales with their Hill sphere.\n");
  }
  if (OverridesOutputdir == YES) {
    sprintf (OUTPUTDIR, "%s", NewOutputdir);
  }
  /* Add a trailing slash to OUTPUTDIR if needed *//*
  if (*(OUTPUTDIR+strlen(OUTPUTDIR)-1) != '/')
    strcat (OUTPUTDIR, "/");
}*/
