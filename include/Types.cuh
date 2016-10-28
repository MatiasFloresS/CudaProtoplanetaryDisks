
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
  float fx_inner;
  float fy_inner;
  float fx_ex_inner;
  float fy_ex_inner;
  float fx_outer;
  float fy_outer;
  float fx_ex_outer;
  float fy_ex_outer;
  float *GlobalForce;
};

typedef struct force Force;


struct pair {
  float x;
  float y;
};

typedef struct pair Pair;


#define   YES   1
#define   NO    0

#define   GET         0
#define   MARK        1
#define   FREQUENCY   2
