__host__ void ApplyBoundaryCondition (float *dens, float *energy, float *vrad, float *vtheta, float step);
__host__ void NonReflectingBoundaryhost(float *dens, float *energy, float *vrad);
__host__ void ReduceMeanHost(float *dens, float *energy);
__host__ void ReduceCshost();
__host__ void MinusMeanHost(float *dens, float *energy);
__host__ void EvanescentBoundary (float *vrad, float *vtheta, float step);
__host__ void Cscudamalloc();
__host__ void Meancudamalloc();
