__host__ void RungeKutta (float *q0, float timestep, float *PlanetMasses, float *q1, int nb, boolean *feelothers);
__host__ void TranslatePlanetRK5 (float *qold, float c1, float c2, float c3, float c4, float c5, float *qnew, int nb);
__host__ void DerivMotionRK5 (float *q_init, float *PlanetMasses, float *deriv, int nb, float dt, boolean *feelothers);
