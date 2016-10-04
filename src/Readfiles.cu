#include "Main.cuh"

using namespace std;

extern char NewOutputdir[1024];
extern int NINTERM, NTOT, NRAD, NSEC, YES, NO, OverridesOutputdir;

extern float RMIN, RMAX, THICKNESSSMOOTHING, ROCHESMOOTHING, ASPECTRATIO, VISCOSITY, ALPHAVISCOSITY, \
SIGMASLOPE, RELEASERADIUS, RELEASEDATE, OMEGAFRAME, MASSTAPER, DT, SIGMA0, TRANSITIONRATIO, TRANSITIONWIDTH, \
LAMBDADOUBLING, ADIABATICINDEX, COOLINGTIME0, G, PI, IMPOSEDDISKDRIFT, FLARINGINDEX, ECCENTRICITY, CAVITYRADIUS, \
CAVITYRATIO, CAVITYWIDTH, TRANSITIONRADIUS;

extern string DISK, FRAME, OUTERSOURCEMASS, WRITEDENSITY, WRITEVELOCITY, WRITEENERGY, WRITETEMPERATURE, \
WRITEDIVV, WRITEQPLUS, INDIRECTTERM, OUTPUTDIR ,INNERBOUNDARY, LABELADVECTION, TRANSPORT, PLANETCONFIG, \
SELFGRAVITY, CICPLANET, FORCEDCIRCULAR, ZMPLUS, ADIABATIC, COOLING, EXCLUDEHILL, RADIALSPACING;

int ExcludeHill, ViscosityAlpha, RocheSmoothing, OpenInner, AdvecteLabel, LogGrid, FastTransport = YES, \
GuidingCenter = NO, Indirect_Term = YES, IsDisk = YES, NonReflecting = NO, Corotating = NO, \
OuterSourceMass = NO, Evanescent = NO, Write_Density = YES, Write_Velocity = YES, Write_Energy = NO, \
Write_Temperature = NO, Write_DivV = NO, Write_Qplus = NO, SelfGravity = NO, SGZeroMode = NO, ZMPluss = NO, \
Adiabaticc = NO, Cooling = NO, CICPlanet = NO, ForcedCircular = NO;

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

__host__ void TellEverything()
{

  float temp, nbfileoutput;
  printf ("\nDisc properties:\n");
  printf ("----------------\n");
  printf ("Inner Radius          : %g\n", RMIN);
  printf ("Outer Radius          : %g\n", RMAX);
  printf ("G : %g\n", G);
  printf ("Aspect Ratio          : %g\n", ASPECTRATIO);
  printf ("VKep at inner edge    : %.3g\n", sqrt(G*1.0*(1.-0.0)/RMIN));
  printf ("VKep at outer edge    : %.3g\n", sqrt(G*1.0/RMAX));
  temp=2.0*PI*SIGMA0/(2.0-SIGMASLOPE)*(pow(RMAX,2.0-SIGMASLOPE) - pow(RMIN,2.0-SIGMASLOPE));	/* correct this and what follows... */
  printf ("Initial Disk Mass             : %g\n", temp);
  temp=2.0*PI*SIGMA0/(2.0-SIGMASLOPE)*(1.0 - pow(RMIN,2.0-SIGMASLOPE));
  printf ("Initial Mass inner to r=1.0  : %g \n", temp);
  temp=2.0*PI*SIGMA0/(2.0-SIGMASLOPE)*(pow(RMAX,2.0-SIGMASLOPE) - 1.0);
  printf ("Initial Mass outer to r=1.0  : %g \n", temp);
  printf ("Travelling time for acoustic density waves :\n");
  temp = 2.0/3.0/ASPECTRATIO*(pow(RMAX,1.5)-pow(RMIN,1.5));
  printf (" * From Rmin to Rmax  : %.2g = %.2f orbits ~ %.1f outputs\n", temp, TellNbOrbits(temp), TellNbOutputs(temp));
  temp = 2.0/3.0/ASPECTRATIO*(pow(RMAX,1.5)-pow(1.0,1.5));
  printf (" * From r=1.0 to Rmax: %.2g = %.2f orbits ~ %.1f outputs\n", temp, TellNbOrbits(temp), TellNbOutputs(temp));
  temp = 2.0/3.0/ASPECTRATIO*(pow(1.0,1.5)-pow(RMIN,1.5));
  printf (" * From r=1.0 to Rmin: %.2g = %.2f orbits ~ %.1f outputs\n", temp, TellNbOrbits(temp), TellNbOutputs(temp));
  temp = 2.0*PI*sqrt(RMIN*RMIN*RMIN/G/1.0);
  printf ("Orbital time at Rmin  : %.3g ~ %.2f outputs\n", temp, TellNbOutputs(temp));
  temp = 2.0*PI*sqrt(RMAX*RMAX*RMAX/G/1.0);
  printf ("Orbital time at Rmax  : %.3g ~ %.2f outputs\n", temp, TellNbOutputs(temp));
  printf ("Sound speed :\n");
  printf (" * At unit radius     : %.3g\n", ASPECTRATIO*sqrt(G*1.0));
  printf (" * At outer edge      : %.3g\n", ASPECTRATIO*sqrt(G*1.0/RMAX));
  printf (" * At inner edge      : %.3g\n", ASPECTRATIO*sqrt(G*1.0/RMIN));
  printf ("\nGrid properties:\n");
  printf ("----------------\n");
  printf ("Number of rings       : %d\n", NRAD);
  printf ("Number of sectors     : %d\n", NSEC);
  printf ("Total cells           : %d\n", NRAD*NSEC);
  printf ("\nOutputs properties:\n");
  printf ("-------------------\n");
  printf ("Time increment between outputs : %.3f = %.3f orbits\n", NINTERM*DT, TellNbOrbits(NINTERM*DT));
  printf ("At each output #i, the following files are written:\n");
  printf ("gasdens[i].dat : %d bytes\n",(int)(NRAD*NSEC*sizeof(float)));
  printf ("gasvrad[i].dat : %d bytes\n",(int)(NRAD*NSEC*sizeof(float)));
  printf ("gasvtheta[i].dat : %d bytes\n",(int)(NRAD*NSEC*sizeof(float)));
  if (Adiabaticc == YES)
    printf ("gasTemperature[i].dat : %d bytes\n",(int)(NRAD*NSEC*sizeof(float)));
  if (AdvecteLabel == YES)
    printf ("gaslabel[i].dat : %d bytes\n",(int)(NRAD*NSEC*sizeof(float)));
  printf ("There will be in total %d outputs\n", NTOT/NINTERM);
  printf ("(which correspond to an elapsed time = %.3f or to %.2f orbits)\n", NTOT*DT, TellNbOrbits(NTOT*DT));
  nbfileoutput = 3.0;
  if (Adiabaticc == YES)
    nbfileoutput += 1.0;
  if (AdvecteLabel == YES)
    nbfileoutput += 1.0;
  temp =nbfileoutput*NRAD*NSEC*sizeof(float);
  temp *= (float)NTOT/(float)NINTERM;
  temp /= 1024.0*1024.0;
  printf ("So the code will produce ~%.2f Mbytes of data\n", temp);
  printf ("Check (eg by issuing a 'df' command) that you have enough disk space,\n");
  printf ("otherwise you will get a system full and the code will stop.\n");
  fflush (stdout);

}

__host__ float TellNbOrbits (float time)
{
  return time/2.0/PI*sqrt(G*1.0/1.0/1.0/1.0);
}

__host__ float TellNbOutputs (float time)
{
  return (time/DT/NINTERM);
}

__host__ void ReadFile(char *ParameterFile)
{

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
    while (fgets(s, 349, input) != NULL)
    {
      success = sscanf(s, "%s ", nm);
      if ((nm[0] != '#') && (success == 1))
      {  /* # begins a comment line */
        s1 = s + strlen(nm);
        sscanf(s1 + strspn(s1, "\t :=>_"), "%f", &temp);
        sscanf(s1 + strspn(s1, "\t :=>_"), "%289s ", stringval);
        valuef = (float) temp;
        valuei = (int) temp;
        for (int i = 0; i < strlen(nm); i++) nm[i] = (char) toupper(nm[i]);
        SearchVariable(nm, stringval);
      }
    }
    fclose(input);
    // for(int i = 0; i < RADIALSPACING.size(); i++) {
      // std::cout  << RADIALSPACING[i];
    // }


    if (LABELADVECTION.compare("YES") == 0) AdvecteLabel = YES;
    if (OUTERSOURCEMASS.compare("YES") == 0) OuterSourceMass = YES;
    if (TRANSPORT.compare("STANDARD") == 0) FastTransport = NO;
    if (INNERBOUNDARY.compare("OPEN") == 0) OpenInner = YES;
    if (INNERBOUNDARY.compare("NONREFLECTING") == 0) NonReflecting = YES;
    if (INNERBOUNDARY.compare("EVANESCENT") == 0) Evanescent = YES;
    if (RADIALSPACING.compare("Logarithm") == 0) LogGrid = YES;
    if (DISK.compare("NO") == 0) IsDisk = NO;
    if (FRAME.compare("COROTATING") == 0) Corotating = YES;
    if (FRAME.compare("Guiding-Center") == 0)
    {
      GuidingCenter = YES;
      Corotating = YES;
    }
    if (WRITEVELOCITY.compare("NO") == 0) Write_Velocity = NO;
    if (WRITEDENSITY.compare("NO") == 0) Write_Density = NO;
    if (WRITEENERGY.compare("YES") == 0) Write_Energy = YES;
    if (WRITETEMPERATURE.compare("YES") == 0) Write_Temperature = YES;
    if (WRITEDIVV.compare("YES") == 0) Write_DivV = YES;
    if (WRITEQPLUS.compare("YES") == 0) Write_Qplus = YES;
    if (INDIRECTTERM.compare("NO") == 0) Indirect_Term = NO;
    if (SELFGRAVITY.compare("YES") == 0) SelfGravity = YES;
    if (SELFGRAVITY.compare("Z") == 0)
    {
      SelfGravity = YES;
      SGZeroMode = YES;
    }
    if (ZMPLUS.compare("YES") == 0) ZMPluss = YES;
    if ((ZMPluss) && (!SGZeroMode))
    {
      printf("This is not very meaningfull to involve the anisotropic pressure model (ZMPluss=Yes) \
      without taking into account the axisymmetric component of the disk self-gravity. I decided to \
      put ZMPluss = No. Please check again!");
      ZMPluss = NO;
    }

    if (ADIABATIC.compare("YES") == 0)
    {
      Adiabaticc = YES;
      Write_Temperature = YES;
    }
    if (COOLING.compare("YES") == 0) Cooling = YES;
    if ((Adiabaticc) && (ADIABATICINDEX == 1))
    {
      printf("You cannot have Adiabatic = YES and AdiabatcIndex = 1. I decided to put Adiabatic = No,\
       to simulate a locally isothermal equation of state. Please check that it what you really wanted to do!\n");
      Adiabaticc = NO;
    }
    if (WRITEENERGY.compare("NO") == 0) Write_Energy = NO;
    if (EXCLUDEHILL.compare("YES") == 0) ExcludeHill = YES;
    if (CICPLANET.compare("YES") == 0) CICPlanet = YES;
    if (FORCEDCIRCULAR.compare("YES") == 0) ForcedCircular = YES;
    if ((ALPHAVISCOSITY != 0.0) && (VISCOSITY != 0.0))
    {
      printf("You cannot use at the same time\n");
      printf("VISCOSITY and ALPHAVISCOSITY.\n");
      printf("Edit the parameter file so as to remove\n");
      printf("one of these variables and run again.\n");
      exit(1);
    }
    if (ALPHAVISCOSITY != 0.0)
    {
      ViscosityAlpha = YES;
      printf("Viscosity is of alpha type\n");
    }
    if ((THICKNESSSMOOTHING != 0.0) && (ROCHESMOOTHING != 0.0))
    {
      printf("You cannot use at the same time\n");
      printf("ThicknessSmoothing and RocheSmoothing.\n");
      printf("Edit the parameter file so as to remove\n");
      printf("one of these variables and run again.\n");
      exit(1);
    }
    if ((THICKNESSSMOOTHING <= 0.0) && (ROCHESMOOTHING <= 0.0))
    {
      printf("A non-vanishing potential smoothing length is required.\n");
      printf("Please use either of the following variables:\n");
      printf("ThicknessSmoothing *or* RocheSmoothing.\n");
      printf("before launching the run again.\n");
      exit(1);
    }
    if (ROCHESMOOTHING != 0.0)
    {
      RocheSmoothing = YES;
      printf("Planet potential smoothing scales with their Hill sphere.\n");
    }
    if (OverridesOutputdir == YES) OUTPUTDIR = NewOutputdir;

    /* Add a trailing slash to OUTPUTDIR if needed *//*
    if (*(OUTPUTDIR+strlen(OUTPUTDIR)-1) != '/')
      strcat (OUTPUTDIR, "/");
  }*/
}

__host__ void SearchVariable(char *name, char *stringval)
{
  float temp;
  char *stringval2;
  stringval2 = stringval;
  sscanf (stringval2, "%f", &temp);

  if (strcmp(name, "DT") == 0) DT = temp;
  if (strcmp(name, "SIGMA0") == 0) SIGMA0 = temp;
  if (strcmp(name, "NINTERM") == 0) NINTERM = (int) (temp);
  if (strcmp(name, "NTOT") == 0) NTOT = (int) (temp);
  if (strcmp(name, "OUTPUTDIR") == 0) OUTPUTDIR = stringval;
  if (strcmp(name, "INNERBOUNDARY") == 0) INNERBOUNDARY = stringval;
  if (strcmp(name, "LABELADVECTION") == 0) LABELADVECTION = stringval;
  if (strcmp(name, "TRANSPORT") == 0) TRANSPORT = stringval;
  if (strcmp(name, "PLANETCONFIG") == 0) PLANETCONFIG = stringval;
  if (strcmp(name, "MASSTAPER") == 0) MASSTAPER = (float) (temp);
  if (strcmp(name, "RADIALSPACING") == 0) RADIALSPACING = stringval;
  if (strcmp(name, "NRAD") == 0) NRAD = (int) (temp);
  if (strcmp(name, "NSEC") == 0) NSEC = (int) (temp);
  if (strcmp(name, "RMIN") == 0) RMIN = (float) (temp);
  if (strcmp(name, "RMAX") == 0) RMAX = (float) (temp);
  if (strcmp(name, "THICKNESSSMOOTHING") == 0) THICKNESSSMOOTHING = (float) (temp);
  if (strcmp(name, "ROCHESMOOTHING") == 0) ROCHESMOOTHING = (float) (temp);
  if (strcmp(name, "ASPECTRATIO") == 0) ASPECTRATIO = (float) (temp);
  if (strcmp(name, "VISCOSITY") == 0) VISCOSITY = (float) (temp);
  if (strcmp(name, "ALPHAVISCOSITY") == 0) ALPHAVISCOSITY = (float) (temp);
  if (strcmp(name, "SIGMASLOPE") == 0) SIGMASLOPE = (float) (temp);
  if (strcmp(name, "RELEASERADIUS") == 0) RELEASERADIUS = (float) (temp);
  if (strcmp(name, "RELEASEDATE") == 0) RELEASEDATE = (float) (temp);
  if (strcmp(name, "OMEGAFRAME") == 0) OMEGAFRAME = (float) (temp);
  if (strcmp(name, "DISK") == 0) DISK = stringval;
  if (strcmp(name, "FRAME") == 0) FRAME = stringval;
  if (strcmp(name, "OUTERSOURCEMASS") == 0) OUTERSOURCEMASS = stringval;
  if (strcmp(name, "WRITEDENSITY") == 0) WRITEDENSITY = stringval;
  if (strcmp(name, "WRITEVELOCITY") == 0) WRITEVELOCITY = stringval;
  if (strcmp(name, "WRITEENERGY") == 0) WRITEENERGY = stringval;
  if (strcmp(name, "WRITETEMPERATURE") == 0) WRITETEMPERATURE = stringval;
  if (strcmp(name, "WRITEDIVV") == 0) WRITEDIVV = temp;
  if (strcmp(name, "WRITEQPLUS") == 0) WRITEQPLUS = stringval;
  if (strcmp(name, "INDIRECTTERM") == 0) INDIRECTTERM = stringval;
  if (strcmp(name, "EXCLUDEHILL") == 0) EXCLUDEHILL = stringval;
  if (strcmp(name, "IMPOSEDDISKDRIFT") == 0) IMPOSEDDISKDRIFT = (float) (temp);
  if (strcmp(name, "FLARINGINDEX") == 0) FLARINGINDEX = (float) (temp);
  if (strcmp(name, "ECCENTRICITY") == 0) ECCENTRICITY = (float) (temp);
  if (strcmp(name, "CAVITYRADIUS") == 0) CAVITYRADIUS = (float) (temp);
  if (strcmp(name, "CAVITYRATIO") == 0) CAVITYRATIO = (float) (temp);
  if (strcmp(name, "CAVITYWIDTH") == 0) CAVITYWIDTH = (float) (temp);
  if (strcmp(name, "TRANSITIONRADIUS") == 0) TRANSITIONRADIUS = (float) (temp);
  if (strcmp(name, "TRANSITIONRATIO") == 0) TRANSITIONRATIO = (float) (temp);
  if (strcmp(name, "TRANSITIONWIDTH") == 0) TRANSITIONWIDTH = (float) (temp);
  if (strcmp(name, "LAMBDADOUBLING") == 0) LAMBDADOUBLING = (float) (temp);
  if (strcmp(name, "SELFGRAVITY") == 0) SELFGRAVITY = stringval;
  if (strcmp(name, "CICPLANET") == 0) CICPLANET = stringval;
  if (strcmp(name, "FORCEDCIRCULAR") == 0) FORCEDCIRCULAR = stringval;
  if (strcmp(name, "ZMPLUS") == 0) ZMPLUS = stringval;
  if (strcmp(name, "ADIABATIC") == 0) ADIABATIC = stringval;
  if (strcmp(name, "ADIABATICINDEX") == 0) ADIABATICINDEX = (float) (temp);
  if (strcmp(name, "COOLING") == 0) COOLING = stringval;
  if (strcmp(name, "COOLINGTIME0") == 0) COOLINGTIME0 = (float) (temp);
}
