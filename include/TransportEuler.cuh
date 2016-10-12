__host__ void Transport (float *dens, float *vrad, float *vtheta, float *energy, float *label, float dt);
__host__ void OneWindRad (float *dens, float *vrad, float *energy, float dt);
__host__ void ComputeLRMomenta ();
__host__ void ComputeExtQty ();
__host__ void ActualiseGasDens (float *densint, float *dens);
__host__ void ComputeStarRad(float *dens, float *vrad, float *Rhostar, float dt);
__host__ void VanLeerRadial (float *vrad, float *Qbase, float dt);
__host__ void InitTransportDevice ();
__host__ void InitTransport ();
