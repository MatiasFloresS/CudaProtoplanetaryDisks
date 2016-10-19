#include "Main.cuh"

static float k1[100], k2[100], k3[100], k4[100], k5[100], k6[100];
static float Dist[100];

extern float G;
extern int Indirect_Term, YES;

__host__ void RungeKutta (float *q0, float timestep, float *PlanetMasses, float *q1, int nb, boolean *feelothers)
{
  for (int i = 0; i < 4*nb; i++)
  {
    k1[i] = k2[i] = k3[i] = k4[i] = k5[i] = k6[i] = 0;
  }

  DerivMotionRK5 (q0, PlanetMasses, k1, nb, timestep, feelothers);
  TranslatePlanetRK5 (q0, 0.2, 0.0, 0.0, 0.0, 0.0, q1, nb);
  DerivMotionRK5 (q1, PlanetMasses, k2, nb, timestep, feelothers);
  TranslatePlanetRK5 (q0, 0.075, 0.225, 0.0, 0.0, 0.0, q1, nb);
  DerivMotionRK5 (q1, PlanetMasses, k3, nb, timestep, feelothers);
  TranslatePlanetRK5 (q0, 0.3, -0.9, 1.2, 0.0, 0.0, q1, nb);
  DerivMotionRK5 (q1, PlanetMasses, k4, nb, timestep, feelothers);
  TranslatePlanetRK5 (q0, -11.0/54.0, 2.5, -70.0/27.0, 35.0/27.0, 0.0, q1, nb);
  DerivMotionRK5 (q1, PlanetMasses, k5, nb, timestep, feelothers);
  TranslatePlanetRK5 (q0, 1631.0/55296.0, 175.0/521.0, 575.0/13824.0, 44275.0/110592.0, 253.0/4096.0, q1, nb);
  DerivMotionRK5 (q1, PlanetMasses, k6, nb, timestep, feelothers);
  for (int i = 0; i < 4*nb; i++)
  {
    q1[i] = q0[i]+37.0/378.0*k1[i]+250.0/621.0*k3[i]+125.0/594.0*k4[i]+512.0/1771.0*k6[i];
  }
}

__host__ void DerivMotionRK5 (float *q_init, float *PlanetMasses, float *deriv, int nb, float dt, boolean *feelothers)
{
  float *x, *y, *vx, *vy, dist;
  float *derivx, *derivy, *derivvx, *derivvy;
  int i,j;

  x = q_init;
  y = x+nb;
  vx = y+nb;
  vy = vx+nb;
  derivx = deriv;
  derivy = derivx+nb;
  derivvx = derivy+nb;
  derivvy = derivvx+nb;

  for (i = 0; i < nb; i++)
    Dist[i] = sqrt(x[i]*x[i]+y[i]*y[i]);

  for (i = 0; i < nb; i++)
  {
    derivx[i] = vx[i];
    derivy[i] = vy[i];
    derivvx[i] = -G*1.0/Dist[i]/Dist[i]/Dist[i]*x[i];
    derivvy[i] = -G*1.0/Dist[i]/Dist[i]/Dist[i]*y[i];
    for (j = 0; j < nb; j++)
    {
      if (Indirect_Term)
      {
        derivvx[i] -= G*PlanetMasses[j]/Dist[j]/Dist[j]/Dist[j]*x[j];
        derivvy[i] -= G*PlanetMasses[j]/Dist[j]/Dist[j]/Dist[j]*y[j];
      }
      if ((j != i) && (feelothers[i] == YES))
      {
        dist = (x[i]-x[j])*(x[i]-x[j])+(y[i]-y[j])*(y[i]-y[j]);
        dist = sqrt(dist);
        derivvx[i] += G*PlanetMasses[j]/Dist[j]/Dist[j]/Dist[j]*(x[j]-x[i]);
        derivvy[i] += G*PlanetMasses[j]/Dist[j]/Dist[j]/Dist[j]*(y[j]-y[i]);
      }
    }
  }
  for (i = 0; i < 4*nb; i++)
    deriv[i] *= dt;
}

__host__ void TranslatePlanetRK5 (float *qold, float c1, float c2, float c3, float c4, float c5, float *qnew, int nb)
{
  for (int i = 0; i < 4*nb; i++)
  {
    qnew[i] = qold[i]+c1*k1[i]+c2*k2[i]+c3*k3[i]+c4*k4[i]+c5*k5[i];
  }
}
