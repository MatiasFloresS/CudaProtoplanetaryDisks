#include "Main.cuh"

extern int NRAD;

extern float *Rmed, *Rinf, CAVITYRATIO, CAVITYRADIUS, SIGMASLOPE, SIGMA0, *QplusMed, *SigmaMed, *SigmaInf, \
ADIABATICINDEX, *EnergyMed, ScalingFactor, *CoolingTimeMed, *QplusMed, R, MU, ASPECTRATIO, FLARINGINDEX,   \
COOLINGTIME0;


/* Surface density */
__host__ float Sigma(float r)
{
  float cavity = 1.0;
  if (r < CAVITYRADIUS) cavity = 1.0/CAVITYRATIO;
  /* This is *not* a steady state */
  /* profile, if a cavity is defined. It first needs */
  /* to relax towards steady state, on a viscous time scale */
  return cavity*ScalingFactor*SIGMA0*pow(r,-SIGMASLOPE);
}



__host__ void FillSigma ()
{
  int i;
  for (i = 0; i < NRAD; i++){
    SigmaMed[i] = Sigma(Rmed[i]);
    SigmaInf[i] = Sigma(Rinf[i]);
  }
}



/* Thermal energy */
__host__ float Energy(float r)
{
  float energy0;
  if (ADIABATICINDEX == 1.0){
    fprintf(stderr, "The adiabatic index must differ from unity to initialized \
    the gas internal energy. I must exit.\n");
    exit(1);
  }
  else
    energy0 = R/MU/(ADIABATICINDEX-1.0)*SIGMA0*pow(ASPECTRATIO,2.0)*pow(r,-SIGMASLOPE-1.0+2.0*FLARINGINDEX);

  return energy0;
}



__host__ void FillEnergy ()
{
  int i;
  for (i = 0; i < NRAD; i++)
    EnergyMed[i] = Energy(Rmed[i]);
}



__host__ float CoolingTime(float r)
{
  float ct0;
  ct0 = COOLINGTIME0*pow(r,2.0+2.0*FLARINGINDEX);
  return ct0;
}

__host__ void FillCoolingTime()
{
  for (int i = 0; i < NRAD; i++) CoolingTimeMed[i] = CoolingTime(Rmed[i]);
}

__host__ void FillQplus()
{
  for (int i = 0; i < NRAD; i++) QplusMed[i] = Qplusinit(Rmed[i]);
}

__host__ float Qplusinit(float r)
{
  float qp0, viscosity;
  viscosity = FViscosity(r);
  qp0 = 2.25*viscosity*SIGMA0*pow(r,-SIGMASLOPE-3.0);
  return qp0;
}
