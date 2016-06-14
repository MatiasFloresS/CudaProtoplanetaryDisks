#include "main.cuh"
#include "Psys.cuh"

extern int YES, NO;
extern float *Rmed,  G,  *Radii;
extern bool CICPlanet, ForcedCircular;
extern float ECCENTRICITY;
float HillRadius;
PlanetarySystem *InitPlanetarySystem (char *filename)
{
  int nb, i=0, j;
  PlanetarySystem *sys;
  FILE *input;
  char s[512], nm[512], test1[512], test2[512], *s1;
  float mass, dist, accret;
  bool feeldis, feelothers;
  //   extern boolean CICPlanet, ForcedCircular;


  nb = FindNumberOfPlanets(filename);
  printf("%d planet(s) found.\n",nb);
  sys = AllocPlanetSystem (nb);
  input = fopen (filename, "r");
  sys->nb = nb;
  while (fgets(s, 510, input) != NULL) {
    sscanf(s, "%s ", nm);
    if (isalpha(s[0])) {
      s1 = s + strlen(nm);
      sscanf(s1 + strspn(s1, "\t :=>_"), "%f %f %f %s %s", &dist, &mass, &accret, test1, test2);
      if ( CICPlanet ) {
        // initialization puts centered-in-cell planets (with excentricity = 0 only)
        j = 0;
        while ( Rmed[j] < dist ) j++;
        dist = Radii[j+1];
      }
      sys->mass[i] = (float)mass;
      feeldis = feelothers = YES;
      if (tolower(*test1) == 'n') feeldis = NO;
      /*
        if ( (feeldis == YES) && (ForcedCircular == YES) ) {
        masterprint ("Careful: there is a contradiction between FeelDisk = Yes in your planet configuration file, and ForcedCircular = Yes in your parameter file. I decided to put FeelDisk = No. Please check this is what you really want to mean.");
        feeldis = NO;
      }    */
      if (tolower(*test2) == 'n') feelothers = NO;
      sys->x[i] = (float)dist*(1.0+ECCENTRICITY);
      sys->y[i] = 0.0;
      sys->vy[i] = (float)sqrtf(G*(1.0+mass)/dist)*			\
  	   sqrtf( (1.0-ECCENTRICITY)/(1.0+ECCENTRICITY) );
      sys->vx[i] = -0.0000000001*sys->vy[i];
      sys->acc[i] = accret;
      sys->FeelDisk[i] = feeldis;
      sys->FeelOthers[i] = feelothers;
      i++;
    }
  }
  HillRadius = sys->x[0] * powf( sys->mass[0]/3., 1./3. );
  // printf("%f\n",sys->x[0] );
  // printf("%f\n",sys->mass[0] );
  // printf("%f\n", HillRadius);
  return sys;
}


int FindNumberOfPlanets (char *filename)
{
  FILE *input;
  char s[512];
  int cont=0;
  input = fopen (filename, "r");
  if (input == NULL) {
    fprintf(stderr, "Error : can't find '%s'.\n",filename);
    exit(1);
  }
  while (fgets(s, 510, input) != NULL) {
    if (isalpha(s[0])) cont++;
  }
  fclose (input);
  return cont;
}

PlanetarySystem *AllocPlanetSystem (int nb)
{
  float *mass, *x, *y, *vx, *vy, *acc;
  bool *feeldisk, *feelothers;
  PlanetarySystem *sys;
  sys = (PlanetarySystem *)malloc (sizeof(PlanetarySystem));
  if (sys == NULL) {
    fprintf(stderr, "Not enough memory.\n");
    exit(1);
  }
  x    = (float *)malloc (sizeof(float)*(nb+1));
  y    = (float *)malloc (sizeof(float)*(nb+1));
  vy   = (float *)malloc (sizeof(float)*(nb+1));
  vx   = (float *)malloc (sizeof(float)*(nb+1));
  mass = (float *)malloc (sizeof(float)*(nb+1));
  acc  = (float *)malloc (sizeof(float)*(nb+1));
  if ((x == NULL) || (y == NULL) || (vx == NULL) || (vy == NULL) || (acc == NULL) || (mass == NULL)) {
    fprintf (stderr, "Not enough memory.\n");
    exit (1);
  }

  feeldisk   = (bool *)malloc (sizeof(float)*(nb+1));
  feelothers = (bool *)malloc (sizeof(float)*(nb+1));
  if ((feeldisk == NULL) || (feelothers == NULL)) {
    fprintf (stderr, "Not enough memory.\n");
    exit (1);
  }
  sys->x = x;
  sys->y = y;
  sys->vx= vx;
  sys->vy= vy;
  sys->acc=acc;
  sys->mass = mass;
  sys->FeelDisk = feeldisk;
  sys->FeelOthers = feelothers;

  for (int i = 0; i < nb; i++) {
    x[i] = y[i] = vx[i] = vy[i] = mass[i] = acc[i] = 0.0;
    feeldisk[i] = feelothers[i] = YES;
  }
  return sys;
}

void ListPlanets (PlanetarySystem *sys)
{
  int nb;
  int i;
  nb = sys->nb;
  for (i = 0; i < nb; i++) {
    printf ("Planet number %d\n", i);
    printf ("---------------\n");
    printf ("x = %f\ty = %f\n", sys->x[i],sys->y[i]);
    printf ("vx = %f\tvy = %f\n", sys->vx[i],sys->vy[i]);
    if (sys->acc[i] == 0.0)
      printf ("Non-accreting.\n");
    else
      printf ("accretion time = %f\n", 1.0/(sys->acc[i]));
    if (sys->FeelDisk[i] == YES) {
      printf ("Feels the disk potential\n");
    } else {
      printf ("Doesn't feel the disk potential\n");
    }
    if (sys->FeelOthers[i] == YES) {
      printf ("Feels the other planets potential\n");
    } else {
      printf ("Doesn't feel the other planets potential\n");
    }
    printf ("\n");
  }
}
