__host__ void RungeKutta (double *q0, double timestep, double *PlanetMasses, double *q1, int nb, int *feelothers);
__host__ void TranslatePlanetRK5 (double *qold, double c1, double c2, double c3, double c4, double c5, double *qnew, int nb);
__host__ void DerivMotionRK5 (double *q_init, double *PlanetMasses, double *deriv, int nb, double dt, int *feelothers);
