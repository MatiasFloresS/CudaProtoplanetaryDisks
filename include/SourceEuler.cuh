__host__ void FillPolar1DArray();
__host__ void InitEuler (float *dens, float *energy, float *vrad, float *vtheta);
__host__ float AspectRatio(float r);
__host__ void InitGasVelocitieshost(float *vrad, float *vtheta);
__host__ void Computecudamalloc(float *dens, float *energy, float *vrad, float *vtheta);
__host__ float ConstructSequence(float *u, float *v, int n);
__host__ void init_azimutalvelocity_withSG(float *vtheta);
