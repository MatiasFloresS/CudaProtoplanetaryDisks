#include "Main.cuh"

extern int NRAD, NSEC, NO, size_grid, RocheSmoothing;
extern int ForcedCircular;
extern int Indirect_Term, SelfGravity, Cooling, CentrifugalBalance;

extern float *SigmaMed, *EnergyMed, *Potential_d, G, *Rmed_d, ROCHESMOOTHING, MassTaper, *q0, \
*PlanetMasses, *q1, *Pressure, *Pressure_d, *SoundSpeed, *SoundSpeed_d, *Rmed, *viscosity_array;

extern float *Radii, ViscosityAlpha, *GLOBAL_bufarray, OmegaFrame, *vt_int, *SG_Accr, *SG_Accr_d, *vt_cent;
extern float ASPECTRATIO, FLARINGINDEX, *viscosity_array_d, *vt_cent_d, *Vrad_d, *Vtheta_d;
extern float SIGMASLOPE, SIGMA0, *SigmaInf_d, *Rinf_d, IMPOSEDDISKDRIFT, PhysicalTime, RELEASEDATE, RELEASERADIUS;

extern Pair DiskOnPrimaryAcceleration;
static Pair IndirectTerm;

extern dim3 dimGrid2, dimBlock2;



__host__ void InitGasDensity (float *Dens)
{
  int i, j;
  FillSigma ();
  for (i = 0; i < NRAD; i++){
    for (j = 0; j < NSEC; j++){
      Dens[j+i*NSEC] = SigmaMed[i];
    }
  }
}

__host__ void InitGasEnergy (float *Energy)
{
  FillEnergy ();
  for (int i = 0; i < NRAD; i++){
    for (int j = 0; j < NSEC; j++){
      Energy[i*NSEC + j] = EnergyMed[i];
    }
  }
}

__host__ void FillForcesArrays (PlanetarySystem *sys, float *Dens, float *Energy)
{
  int NbPlanets, k;
  float xplanet, yplanet, mplanet, PlanetDistance, InvPlanetDistance3, RRoche, smooth, smoothing;
  NbPlanets = sys->nb;

  /* Indirect term star on gas here */
  ComputeIndirectTerm ();

  gpuErrchk(cudaMemset(Potential_d, 0, size_grid*sizeof(float)));
  /* -- Gravitational potential from planet on gas -- */
  for (k = 0; k < NbPlanets; k++){
    xplanet = sys->x[k];
    yplanet = sys->y[k];
    mplanet = sys->mass[k]*MassTaper;
    PlanetDistance = sqrt(xplanet*xplanet+yplanet*yplanet);
    InvPlanetDistance3 = 1.0/PlanetDistance/PlanetDistance/PlanetDistance;
    RRoche = PlanetDistance*pow((1.0/3.0*mplanet),1.0/3.0);
    if (RocheSmoothing) smoothing = RRoche*ROCHESMOOTHING;
    else smoothing = Compute_smoothing (PlanetDistance);
    smooth = smoothing*smoothing;

    FillForcesArraysKernel<<<dimGrid2,dimBlock2>>>(Rmed_d, NSEC, NRAD, xplanet, yplanet, smooth,
      G, mplanet, Indirect_Term, InvPlanetDistance3, Potential_d, IndirectTerm);
    gpuErrchk(cudaDeviceSynchronize());
  }
}


__host__ void ComputeIndirectTerm ()
{
  IndirectTerm.x = -DiskOnPrimaryAcceleration.x;
  IndirectTerm.x = -DiskOnPrimaryAcceleration.y;
  if (!Indirect_Term){
    IndirectTerm.x = 0.0;
    IndirectTerm.y = 0.0;
  }
}

__host__ void AdvanceSystemFromDisk (Force *force, float *Dens, float *Energy, PlanetarySystem *sys, float dt)
{
  int NbPlanets, k;
  float m, x, y, r, smoothing;
  Pair gamma;
  NbPlanets = sys->nb;

  for (k = 0; k < NbPlanets; k++){
    if (sys->FeelDisk[k]){
      m = sys->mass[k];
      x = sys->x[k];
      y = sys->y[k];
      r = sqrt(x*x + y*y);
      if (RocheSmoothing) smoothing = r*pow(m/3.,1./3.)*ROCHESMOOTHING;
      else smoothing = Compute_smoothing (r);
      gamma = ComputeAccel (force, Dens, x, y, smoothing, m);
      sys->vx[k] += dt * gamma.x;
      sys->vy[k] += dt * gamma.y;
      sys->vx[k] += dt * IndirectTerm.x;
      sys->vy[k] += dt * IndirectTerm.y;
    }
  }
}

__host__ void AdvanceSystemRK5 (PlanetarySystem *sys, float dt)
{
  int nb, i , k;
  int *feelothers;
  nb = sys->nb;
  float dtheta, omega, rdot, x, y, r, new_r, vx, vy, theta, denom;

  if (!ForcedCircular){
    for (k = 0; k < nb; k++){
      q0[k] = sys->x[k];
      q0[k+nb] = sys->y[k];
      q0[k+2*nb] = sys->vx[k];
      q0[k+3*nb] = sys->vy[k];
      PlanetMasses[k] = sys->mass[k];
    }
    feelothers = sys->FeelOthers;
    RungeKutta (q0, dt, PlanetMasses, q1, nb, feelothers);
  }
  for (i = 1-(PhysicalTime >= RELEASEDATE); i < sys->nb; i++) {
    if (!ForcedCircular){
      sys->x[i] = q1[i];
      sys->y[i] = q1[i+nb];
      sys->vx[i] = q1[i+2*nb];
      sys->vy[i] = q1[i+3*nb];
    }
    else {
      x = sys->x[i];
      y = sys->y[i];
      theta = atan2(y,x);
      vx = sys->vx[i];
      vy = sys->vy[i];
      r = sqrt(x*x +y*y);
      omega = (-y*vx + x*vy)/r/r;
      dtheta = omega*dt;
      sys->x[i] = r*cos(theta+dtheta);
      sys->y[i] = r*sin(theta+dtheta);
      sys->vx[i] = vx*cos(theta+dtheta) - vy*sin(dtheta+theta);
      sys->vy[i] = vx*sin(theta+dtheta) + vx*cos(dtheta+theta);
    }
  }
  if (PhysicalTime < RELEASEDATE){
    x = sys->x[0];
    y = sys->y[0];
    r = sqrt(x*x+y*y);
    theta = atan2(y,x);
    rdot = (RELEASERADIUS-r)/(RELEASEDATE-PhysicalTime);
    omega = sqrt((1.+sys->mass[0])/r/r/r);
    new_r = r + rdot*dt;
    denom = r-new_r;
    if (denom != 0.0){
      dtheta = 2.*dt*r*omega/denom*(sqrt(r/new_r)-1.);
    }
    else {
      dtheta = omega*dt;
    }
    vx = rdot;
    vy = new_r*sqrt((1.+sys->mass[0])/new_r/new_r/new_r);
    sys->x[0] = new_r*cos(dtheta+theta);
    sys->y[0] = new_r*sin(dtheta+theta);
    sys->vx[0] = vx*cos(dtheta+theta) - vy*sin(dtheta+theta);
    sys->vy[0] = vx*sin(dtheta+theta) + vx*cos(dtheta+theta);
  }

}

__host__ void InitGasVelocities (float *Vrad, float *Vtheta)
{
  float r1, t1, r2, t2, r, ri;
  int i;

  /* Pressure is already initialized: cf initeuler in SourceEuler.c ...
    Initialization of azimutal velocity with exact centrifugal balance */

  if (CentrifugalBalance){
    /* vt_int \equiv Romega = grad(P)/sigma + \partial(phi)/\partial(r) - acc_sg_radial
    ./bin/fargoGPU  -b in/template.par */

    gpuErrchk(cudaMemcpy(Pressure, Pressure_d, size_grid*sizeof(float), cudaMemcpyDeviceToHost));
    Make1Dprofile(1);

    /* global axisymmetric pressure field */
    for (i = 1; i < NRAD; i++){
      vt_int[i] = ( GLOBAL_bufarray[i] - GLOBAL_bufarray[i-1]) / \
      (.5*(Sigma(Rmed[i]) + Sigma(Rmed[i-1])))/(Rmed[i]-Rmed[i-1]);
    }
    /* Case of a disk with self-gravity */
    if ( SelfGravity ) // Better test with CL rigid!
    {
      gpuErrchk(cudaMemcpy(SG_Accr, SG_Accr_d, size_grid*sizeof(float), cudaMemcpyDeviceToHost));
      Make1Dprofile(1);

      for (i = 1; i < NRAD; i++)
        vt_int[i] -= ((Radii[i] - Rmed[i-1]) * GLOBAL_bufarray[i] + \
          (Rmed[i] - Radii[i])* GLOBAL_bufarray[i-1]) / (Rmed[i]-Rmed[i-1]);
    }

    for (i = 1; i < NRAD; i++)
      vt_int[i] = sqrt(vt_int[i]*Radii[i]) - Radii[i]*OmegaFrame;

    t1 = vt_cent[0] = vt_int[1]+.75*(vt_int[1]-vt_int[2]);
    r1 = ConstructSequence (vt_cent, vt_int, NRAD);
    vt_cent[0] += .25*(vt_int[1]-vt_int[2]);
    t2 = vt_cent[0];
    r2 = ConstructSequence (vt_cent, vt_int, NRAD);
    t1 = t1-r1/(r2-r1)*(t2-t1);
    vt_cent[0] = t1;
    ConstructSequence (vt_cent, vt_int, NRAD);
    vt_cent[NRAD] = vt_cent[NRAD-1];
  }

  if (!CentrifugalBalance && SelfGravity)
    Init_azimutalvelocity_withSG (Vtheta);

  if (ViscosityAlpha){
    gpuErrchk(cudaMemcpy(SoundSpeed, SoundSpeed_d, size_grid*sizeof(float), cudaMemcpyDeviceToHost));
    Make1Dprofile(1);
  }

  /* We calculate here the cooling time radial profile (Theo.cu) */
  if (Cooling){
    FillCoolingTime();
    /* To fill qplus, one requires to calculate viscosity, hence cs if one uses
    alpha-viscosity */
    FillQplus();
  }

  for (i = 0; i <= NRAD; i++){
    if (i == NRAD) viscosity_array[i] = FViscosity(Rmed[NRAD-1]);
    else viscosity_array[i] = FViscosity(Rmed[i]);
  }

  InitVelocities(Vrad, Vtheta);
}


__host__ void InitVelocities (float *Vrad, float *Vtheta)
{
  gpuErrchk(cudaMemcpy(viscosity_array_d, viscosity_array, (NRAD+1)*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(vt_cent_d, vt_cent,     (NRAD+1)*sizeof(float), cudaMemcpyHostToDevice));

  InitGasVelocitiesKernel<<<dimGrid2, dimBlock2>>>(viscosity_array_d, NSEC, NRAD, SelfGravity, Rmed_d,
  G, ASPECTRATIO, FLARINGINDEX, SIGMASLOPE, CentrifugalBalance, Vrad_d, Vtheta_d, ViscosityAlpha,
  IMPOSEDDISKDRIFT, SIGMA0, SigmaInf_d, OmegaFrame, Rinf_d, vt_cent_d);
  gpuErrchk(cudaDeviceSynchronize());

}
