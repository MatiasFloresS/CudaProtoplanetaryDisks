__host__ void ExecuteExeC2Cforward ();
__host__ void ExecuteExeC2Cbackward ();
__host__ void compute_selfgravity (double *Dens, double DeltaT, int SGUpdate, int initialization);
__host__ void compute_FFT ();
__host__ void compute_Convolution ();
__host__ void compute_sgacc ();
__host__ void update_sgvelocity (double DeltaT);
__host__ void compute_kernel ();
__host__ void Init_planetarysys_withSG (PlanetarySystem *sys);
