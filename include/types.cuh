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
