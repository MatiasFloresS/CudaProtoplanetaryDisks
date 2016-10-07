__host__ void ApplyBoundaryCondition (float *dens, float *energy, float *vrad, float *vtheta, float step);
__host__ void NonReflectingBoundary (float *dens, float *energy, float *vrad);
__host__ void ReduceMean (float *dens, float *energy);
__host__ void ReduceCs ();
__host__ void MinusMean (float *dens, float *energy);
__host__ void EvanescentBoundary (float *vrad, float *vtheta, float step);
__host__ void Cscudamalloc ();
__host__ void Meancudamalloc ();
__host__ void OpenBoundary ();
