__host__ void Transport (float *Dens, float *Vrad, float *Vtheta, float *energy, float *label, float dt);
__host__ void OneWindRad (float *Dens, float *Vrad, float *energy, float dt);
__host__ void ComputeLRMomenta ();
__host__ void ComputeExtQty ();
__host__ void ActualiseGasDens (float *DensInt, float *Dens);
__host__ void ComputeStarRad(float *Dens, float *Vrad, float *RhoStar, float dt);
__host__ void VanLeerRadial (float *Vrad, float *Qbase, float dt);
__host__ void InitTransportDevice ();
__host__ void InitTransport ();
