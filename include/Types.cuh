
struct planetary_system {
  int nb;			/* Number of planets */
  double *mass;			/* their masses */
  double *x;			/* their coordinates */
  double *y;
  double *vx;			/* their velocities */
  double *vy;
  double *acc;			/* Their accretion times^-1 */
  char **name;			/* their names */
  int *FeelDisk;		/* will "migrate" ? */
  int *FeelOthers;		/* will feel other planets ? */
};

typedef struct planetary_system PlanetarySystem;

struct force {
  double fx_inner;
  double fy_inner;
  double fx_ex_inner;
  double fy_ex_inner;
  double fx_outer;
  double fy_outer;
  double fx_ex_outer;
  double fy_ex_outer;
  double *GlobalForce;
};

typedef struct force Force;


struct pair {
  double x;
  double y;
};

typedef struct pair Pair;


#define   YES   1
#define   NO    0

#define   GET         0
#define   MARK        1
#define   FREQUENCY   2
