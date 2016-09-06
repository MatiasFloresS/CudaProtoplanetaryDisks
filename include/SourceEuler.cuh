__host__ void FillPolar1DArray();
__host__ void InitEuler (float *dens, float *energy, float *vrad, float *vtheta);
__host__ float AspectRatio(float r);
__host__ void InitGasVelocitieshost(float *vrad, float *vtheta);
__host__ void Computecudamalloc(float *dens, float *energy, float *vrad, float *vtheta);
