using namespace std;

float G	  = 1.0;
float PI4 =	12.56637061435917295376;
float PI	= 3.14159265358979323844;
float MU  = 1.0;  /* Mean molecular weight */
float R   = 1.0; /* Universal Gas Constant in code units */

float CVNR = 1.41;

int GET       = 0;
int MARK      = 1;
int FREQUENCY = 2;

bool SloppyCFL, debug, CentrifugalBalance, Merge, MonitorIntegral, GotoNextOutput;
bool FakeSequential, StoreSigma, StoreEnergy, OnlyInit;
int OverridesOutputdir;
char NewOutputdir[1024];

int YES = 1;
int NO = 0;

/* <------- Init Variables -------> */

float DT = 1.0;
float SIGMA0 = 173.0;
int NINTERM = 10;
int NTOT = 1501;
string OUTPUTDIR = "~masset";
string INNERBOUNDARY = "WALL";
string LABELADVECTION = "NO";
string TRANSPORT = "FAST";
string PLANETCONFIG = "Systems/SolarSystem.cfg";
float MASSTAPER = 0.0000001;
string RADIALSPACING = "ARITHMETIC";
int NRAD = 64;
int NSEC = 64;
float RMIN = 1.0;
float RMAX = 1.0;
float THICKNESSSMOOTHING = 0.0;
float ROCHESMOOTHING = 0.0;
float ASPECTRATIO = 0.05;
float VISCOSITY = 0.0;
float ALPHAVISCOSITY = 0.0;
float SIGMASLOPE = 0.0;
float RELEASERADIUS = 0.0;
float RELEASEDATE = 0.0;
float OMEGAFRAME = 0.0;
float OmegaFrame1;
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
float IMPOSEDDISKDRIFT = 0.0;
float FLARINGINDEX = 0.0;
float ECCENTRICITY = 0.0;
float CAVITYRADIUS = 0.0;
float CAVITYRATIO = 1.0;
float CAVITYWIDTH = 1.0;
float TRANSITIONRADIUS = 0.0;
float TRANSITIONRATIO = 1.0;
float TRANSITIONWIDTH = 1.0;
float LAMBDADOUBLING = 0.0;
string SELFGRAVITY = "NO";
string CICPLANET = "NO";
string FORCEDCIRCULAR = "NO";
string ZMPLUS = "NO";
string ADIABATIC = "NO";
float ADIABATICINDEX =  1.4;
string COOLING =  "NO";
float COOLINGTIME0 =6.28;

/* <------- end Init Variables -------> */
