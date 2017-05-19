#include "Main.cuh"

extern int NRAD, NSEC;
extern float *Rinf, *Rsup;

__host__ void AccreteOntoPlanets (float *Dens, float *Vrad, float *Vtheta, float dt, PlanetarySystem *sys)
{

  double RRoche, Rplanet, distance;
  int i_min, i_max, j_min, j_max, angle;
  double facc, facc1, facc2, frac1, frac2; /* We adopt the same notations as W. Kley*/
  double Xplanet, Yplanet, Mplanet, VXplanet, VYplanet, PxPlanet, PyPlanet;
  double dMplanet, dPxPlanet, dPyPlanet;

  for (int k = 0; k < sys->nb; k++) {
    if (sys->acc[k] > 1e-10){ // case acc positive
      dMplanet = dPxPlanet = dPyPlanet = 0.0;
      /* Hereafter: initialization of W. Kley's parameters */
      facc = dt*(sys->acc[k]);
      facc1 = 1.0/3.0*facc;
      facc2 = 2.0/3.0*facc;
      frac1 = 0.75;
      frac2 = 0.45;
      /* W. Kley's parameters initialization finished */
      Xplanet = sys->x[k];
      Yplanet = sys->y[k];
      VXplanet = sys->vx[k];
      VYplanet = sys->vy[k];
      Mplanet = sys->mass[k];
      Rplanet = sqrt(Xplanet*Xplanet+Yplanet*Yplanet);
      RRoche = pow((1.0/3.0*Mplanet), (1.0/3.0))*Rplanet;
      /* Central mass is 1.0 */
      i_min = 0;
      i_max = NRAD-1;
      while ((Rsup[i_min] < Rplanet-RRoche) && (i_min < NRAD)) i_min++;
      while ((Rinf[i_max] > Rplanet+RRoche) && (i_max > 0)) i_max--;
      angle = atan2(Yplanet, Xplanet);
      j_min = (int)((double)NSEC/2.0/PI*(angle - 2.0*RRoche/Rplanet));
      j_max = (int)((double)NSEC/2.0/PI*(angle + 2.0*RRoche/Rplanet));
      PxPlanet = Mplanet*VXplanet;
      PyPlanet = Mplanet*VYplanet;

      //AccreteOntoPlanetsKernel<<<>>>

    }
  }
}
