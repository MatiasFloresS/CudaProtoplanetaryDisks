
typedef int  boolean;

struct planetary_system {
  int nb;			/* Number of planets */
  float *mass;			/* their masses */
  float *x;			/* their coordinates */
  float *y;
  float *vx;			/* their velocities */
  float *vy;
  float *acc;			/* Their accretion times^-1 */
  char **name;			/* their names */
  bool *FeelDisk;		/* will "migrate" ? */
  bool *FeelOthers;		/* will feel other planets ? */
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
