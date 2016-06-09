#include "main.cuh"
#include "kernels.cuh"
extern int nsec;


__host__ void UpdateVelocitiesWithViscosity(float *RadialVelocity, float *AzimuthalVelocity, float *Rho, float DeltaT)
{

  printf("nsec: %d\n", nsec);
  /*UpdateVelocities<<<dimGrid, dimBlock>>>(vt_d,vr_d,invRmed_d,Rmed_d, Rsup_d, Rinf_d,
    invdiffRmed_d, invdiffSup_d,  rho_d, invRinf_d, Trr_d, Tpp_d, DeltaT, nrad, nsec);

    float *vt, float *vr, float *invRmed, float *Rmed, float *Rsup,
      float *Rinf, float *invdiffRmed, float *invdiffRsup, float *rho, float *invRinf, float *Trr,
      float *Trp, float *Tpp, float DeltaT, int nrad, int nsec*/
}
