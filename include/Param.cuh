int SloppyCFL, debug, CentrifugalBalance, Merge, MonitorIntegral, GotoNextOutput;
int FakeSequential, StoreSigma, StoreEnergy, OnlyInit;
int OverridesOutputdir;
char NewOutputdir[1024];


/* <------- Init Variables -------> */

double DT = 1.0;
double SIGMA0 = 173.0;
int NINTERM = 10;
int NTOT = 1501;
string OUTPUTDIR = "~masset";
string INNERBOUNDARY = "WALL";
string LABELADVECTION = "NO";
string TRANSPORT = "FAST";
string PLANETCONFIG = "Systems/SolarSystem.cfg";
double MASSTAPER = 0.0000001;
string RADIALSPACING = "ARITHMETIC";
int NRAD = 64;
int NSEC = 64;
double RMIN = 1.0;
double RMAX = 1.0;
double THICKNESSSMOOTHING = 0.0;
double ROCHESMOOTHING = 0.0;
double ASPECTRATIO = 0.05;
double VISCOSITY = 0.0;
double ALPHAVISCOSITY = 0.0;
double SIGMASLOPE = 0.0;
double RELEASERADIUS = 0.0;
double RELEASEDATE = 0.0;
double OMEGAFRAME = 0.0;

string DISK = "YES";
string FRAME = "FIXED";
string OUTERSOURCEMASS = "NO";
string WRITEDENSITY = "YES";
string WRITEVELOCITY = "YES";
string WRITEENERGY = "NO";
string WRITETEMPERATURE = "NO";
string WRITEDIVV = "NO";
string WRITEQPLUS = "NO";
string INDIRECTTERM = "YES";
string EXCLUDEHILL = "NO";
double IMPOSEDDISKDRIFT = 0.0;
double FLARINGINDEX = 0.0;
double ECCENTRICITY = 0.0;
double CAVITYRADIUS = 0.0;
double CAVITYRATIO = 1.0;
double CAVITYWIDTH = 1.0;
double TRANSITIONRADIUS = 0.0;
double TRANSITIONRATIO = 1.0;
double TRANSITIONWIDTH = 1.0;
double LAMBDADOUBLING = 0.0;
string SELFGRAVITY = "NO";
string CICPLANET = "NO";
string FORCEDCIRCULAR = "NO";
string ZMPLUS = "NO";
string ADIABATIC = "NO";
double ADIABATICINDEX =  1.4;
string COOLING =  "NO";
double COOLINGTIME0 =6.28;

/* <------- end Init Variables -------> */
