__host__ void FillPolar1DArray();
__host__ void InitEuler (float *gas_v_rad, float *gas_v_theta, float *dens, float *energy);
__host__ float AspectRatio(float r);
__host__ void InitComputeAccelhost();
__host__ void ComputeSoundSpeedhost(float *dens, float *energy);
__host__ void ComputePressureFieldhost(float *dens, float *energy);
__host__ void ComputeTemperatureFieldhost(float *dens, float *energy);
