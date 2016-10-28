#include "Main.cuh"

extern float *Rmed, *Radii, ECCENTRICITY;
static float Xplanet, Yplanet;
extern int GuidingCenter;
float HillRadius;



__host__ int FindNumberOfPlanets (char *filename)
{
  FILE *input;
  char s[512];
  int Counter=0;
  input = fopen (filename, "r");
  if (input == NULL){
    fprintf(stderr, "Error : can't find '%s'.\n",filename);
    exit(1);
  }
  while (fgets(s, 510, input) != NULL){
    if (isalpha(s[0]))
      Counter++;
  }
  fclose (input);
  return Counter;
}



__host__ PlanetarySystem *AllocPlanetSystem (int nb)
{
  double *mass, *x, *y, *vx, *vy, *acc;
  int *feeldisk, *feelothers;
  int i;
  PlanetarySystem *sys;
  sys = (PlanetarySystem *)malloc(sizeof(PlanetarySystem));
  if (sys == NULL){
    fprintf(stderr, "Not enough memory.\n");
    exit(1);
  }
  x    = (double *)malloc(sizeof(double)*(nb+1));
  y    = (double *)malloc(sizeof(double)*(nb+1));
  vy   = (double *)malloc(sizeof(double)*(nb+1));
  vx   = (double *)malloc(sizeof(double)*(nb+1));
  mass = (double *)malloc(sizeof(double)*(nb+1));
  acc  = (double *)malloc(sizeof(double)*(nb+1));
  if ((x == NULL) || (y == NULL) || (vx == NULL) || (vy == NULL) || (acc == NULL) || (mass == NULL)){
    fprintf (stderr, "Not enough memory.\n");
    exit (1);
  }

  feeldisk   = (int *)malloc(sizeof(int)*(nb+1));
  feelothers = (int *)malloc(sizeof(int)*(nb+1));
  if ((feeldisk == NULL) || (feelothers == NULL)){
    fprintf (stderr, "Not enough memory.\n");
    exit (1);
  }
  sys->x = x;
  sys->y = y;
  sys->vx = vx;
  sys->vy = vy;
  sys->acc = acc;
  sys->mass = mass;
  sys->FeelDisk = feeldisk;
  sys->FeelOthers = feelothers;

  for (i = 0; i < nb; i++){
    x[i] = y[i] = vx[i] = vy[i] = mass[i] = acc[i] = 0.0;
    feeldisk[i] = feelothers[i] = YES;
  }
  return sys;
}



__host__ void FreePlanetary (PlanetarySystem *sys)
{
  free (sys->x);
  free (sys->vx);
  free (sys->y);
  free (sys->vy);
  free (sys->mass);
  free (sys->acc);
  free (sys->FeelOthers);
  free (sys->FeelDisk);
  free (sys);
}



__host__ PlanetarySystem *InitPlanetarySystem (char *filename)
{
  extern int CICPlanet, ForcedCircular;
  FILE *input;
  char s[512], nm[512], test1[512], test2[512], *s1;
  PlanetarySystem *sys;
  int i=0, j, nb;
  float mass, dist, accret;
  int feeldis, feelothers;
  nb = FindNumberOfPlanets(filename);
  printf("%d planet(s) found.\n",nb);
  sys = AllocPlanetSystem (nb);
  input = fopen (filename, "r");
  sys->nb = nb;
  while (fgets(s, 510, input) != NULL){
    sscanf(s, "%s ", nm);
    if (isalpha(s[0])){
      s1 = s + strlen(nm);
      sscanf(s1 + strspn(s1, "\t :=>_"), "%f %f %f %s %s", &dist, &mass, &accret, test1, test2);
      if (CICPlanet){
        // initialization puts centered-in-cell planets (with excentricity = 0 only)
        j = 0;
        while (Rmed[j] < dist) j++;
        dist = Radii[j+1];
      }
      sys->mass[i] = (double)mass;
      feeldis = feelothers = YES;
      if (tolower(*test1) == 'n') feeldis = NO;
      /*
        if ((feeldis == YES) && (ForcedCircular == YES)) {
        masterprint ("Careful: there is a contradiction between FeelDisk = Yes in your planet configuration file, and ForcedCircular = Yes in your parameter file. I decided to put FeelDisk = No. Please check this is what you really want to mean.");
        feeldis = NO;
      }
      */
      if (tolower(*test2) == 'n') feelothers = NO;
      sys->x[i] = (double)dist*(1.0+ECCENTRICITY);
      sys->y[i] = 0.0;
      sys->vy[i] = sqrt(G*(1.0+mass)/dist)*			\
  	   sqrt((1.0-ECCENTRICITY)/(1.0+ECCENTRICITY));

      sys->vx[i] = -0.0000000001*sys->vy[i];
      sys->acc[i] = accret;
      sys->FeelDisk[i] = feeldis;
      sys->FeelOthers[i] = feelothers;
      i++;
    }
  }
  HillRadius = sys->x[0] * pow( sys->mass[0]/3., 1./3.);
  return sys;
}



__host__ void ListPlanets (PlanetarySystem *sys)
{
  int nb;
  int i;
  nb = sys->nb;
  for (i = 0; i < nb; i++){
    printf ("Planet number %d\n", i);
    printf ("---------------\n");
    printf ("x = %f\ty = %f\n", sys->x[i],sys->y[i]);
    printf ("vx = %f\tvy = %f\n", sys->vx[i],sys->vy[i]);
    if (sys->acc[i] == 0.0)
      printf ("Non-accreting.\n");
    else
      printf ("accretion time = %f\n", 1.0/(sys->acc[i]));
    if (sys->FeelDisk[i] == YES)
      printf ("Feels the disk potential\n");
    else
      printf ("Doesn't feel the disk potential\n");
    if (sys->FeelOthers[i] == YES)
      printf ("Feels the other planets potential\n");
    else
      printf ("Doesn't feel the other planets potential\n");
    printf ("\n");
  }
}



__host__ float GetPsysInfo (PlanetarySystem *sys, int action)
{

  double d1, d2, cross;
  double x,y, vx, vy, m, h, d, Ax, Ay, e, a, E, M;
  double xc, yc, vxc, vyc, omega;
  double arg, PerihelionPA;
  xc = x = sys->x[0];
  yc = y = sys->y[0];
  vxc = vx= sys->vx[0];
  vyc = vy= sys->vy[0];
  m = sys->mass[0]+1.;
  h = x*vy-y*vx;
  d = sqrt(x*x+y*y);

  Ax = x*vy*vy-y*vx*vy -G*m*x/d;
  Ay = y*vx*vx-x*vx*vy -G*m*y/d;

  e = sqrt(Ax*Ax+Ay*Ay)/m;
  a = h*h/G/m/(1.-e*e);
  if (e == 0.0) arg = 1.0;
  else arg = (1.0-d/a)/e;

  if (fabs(arg) >= 1.0) E = M_PI*(1.-arg/fabs(arg))/2.;
  else E = acos((1.0-d/a)/e);

  if ((x*y*(vy*vy-vx*vx)+vx*vy*(x*x-y*y)) < 0) E= -E;
  M = E-e*sin(E);
  PerihelionPA=atan2(Ay,Ax);
  omega = sqrt(m/a/a/a);

  if (GuidingCenter){
    xc = a*cos(M+PerihelionPA);
    yc = a*sin(M+PerihelionPA);
    vxc = -a*omega*sin(M+PerihelionPA);
    vyc =  a*omega*cos(M+PerihelionPA);
  }

  if (e < 1e-8){
    xc = x;
    yc = y;
    vxc = vx;
    vyc = vy;
  }

  switch (action){
    case 1:
      Xplanet = xc;
      Yplanet = yc;
      return 0.;
      break;
    case 0:
      x = xc;
      y = yc;
      vx = vxc;
      vy = vyc;
      d2 = sqrt(x*x+y*y);
      d1 = sqrt(Xplanet*Xplanet+Yplanet*Yplanet);
      cross = Xplanet*y-x*Yplanet;
      Xplanet = x;
      Yplanet = y;
      return asin(cross/(d1*d2));
      break;
    case 2:
      return omega;
      break;
  }
  return 0.0;
}



__host__ void RotatePsys (PlanetarySystem *sys, float angle) /* Rotate by angle '-angle' */
{
  int i, nb;
  double sint, cost, xt, yt;
  nb = sys->nb;
  sint = sin(angle);
  cost = cos(angle);
  for (i = 0; i < nb; i++){
    xt = sys->x[i];
    yt = sys->y[i];
    sys->x[i] = xt*cost+yt*sint;
    sys->y[i] = -xt*sint+yt*cost;
    xt = sys->vx[i];
    yt = sys->vy[i];
    sys->vx[i] = xt*cost+yt*sint;
    sys->vy[i] = -xt*sint+yt*cost;
  }
}
