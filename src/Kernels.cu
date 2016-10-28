#include "Main.cuh"

extern int blocksize2, size_grid, NRAD, NSEC;

extern float *GLOBAL_bufarray;
extern float *gridfield_d, *GLOBAL_bufarray_d, *axifield_d, *SG_Accr_d, *GLOBAL_AxiSGAccr_d;

extern float ASPECTRATIO, TRANSITIONWIDTH, TRANSITIONRATIO, TRANSITIONRADIUS;
extern float PhysicalTime, PhysicalTimeInitial, LAMBDADOUBLING;

extern dim3 dimGrid, dimBlock, dimGrid4;

__global__ void Substep1Kernel (float *Pressure, float *Dens, float *VradInt, float *invdiffRmed, float *Potential,
   float *Rinf, float *invRinf, float *Vrad, float *VthetaInt, float *Vtheta, float *Rmed, float dt,
   int nrad, int nsec, float OmegaFrame, int ZMPlus, float IMPOSEDDISKDRIFT, float SIGMASLOPE,
   float *powRmed)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = threadIdx.y + blockDim.y*blockIdx.y;
  float gradp, gradphi, vt2;

  // i=1->nrad , j=0->nsec
  if (i > 0 && i<nrad && j<nsec){
    gradp = 2.0*(Pressure[i*nsec + j] - Pressure[(i-1)*nsec + j])/(Dens[i*nsec + j] + Dens[(i-1)*nsec + j])*invdiffRmed[i];
    gradphi = (Potential[i*nsec + j]-Potential[(i-1)*nsec + j])*invdiffRmed[i];
    vt2 = Vtheta[i*nsec + j] + Vtheta[(i-1)*nsec + j] + Vtheta[i*nsec + (j+1)%nsec] + Vtheta[(i-1)*nsec + (j+1)%nsec];
    vt2 = vt2/4.0+Rinf[i]*OmegaFrame;
    vt2 = vt2*vt2;
    VradInt[i*nsec + j] = dt*(-gradp - gradphi + vt2*invRinf[i]);
    if ( i == 5 && j == 5)
    {
      printf("dt %g\n", dt);
      printf("%g\n", gradp);
      printf("%g\n", gradphi);
      printf("%g\n", vt2);
      printf("%g\n", invRinf[i]);
      printf("%g\n", VradInt[i*nsec + j]);
    }
  }


  // i=0->nrad ,   j=0->nsec
  if (i<nrad && j<nsec){
    gradp = 2.0*(Pressure[i*nsec + j] - Pressure[i*nsec + ((j-1)+nsec)%nsec])/(Dens[i*nsec +j] +Dens[i*nsec + ((j-1)+nsec)%nsec]) \
    *1.0/(2.0*M_PI/nsec*Rmed[i]);

    if (ZMPlus) gradp *= 1; //gradp *= SG_aniso_coeff;  Definir mas adelante SG_aniso_coeff

    gradphi = (Potential[i*nsec+ j] - Potential[i*nsec + ((j-1)+nsec)%nsec])*1.0/(2.0*M_PI/nsec*Rmed[i]);
    VthetaInt[i*nsec + j] = Vtheta[i*nsec + j]- dt*(gradp+gradphi);
    VthetaInt[i*nsec + j] += dt*IMPOSEDDISKDRIFT*0.5*powRmed[i];
  }
}

__global__ void Substep3Kernel (float *Dens, float *Qplus, float *viscosity_array, float *TAURR, float *TAURP,float *TAUPP,
  float *DivergenceVelocity, int nrad, int nsec, float *Rmed, int Cooling, float *EnergyNew, float dt, float *EnergyMed,
  float *SigmaMed, float *CoolingTimeMed, float *Energy, float ADIABATICINDEX, float *QplusMed)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = threadIdx.y + blockDim.y*blockIdx.y;
  float den, num;

  i+= 1;

  if (i<nrad && j<nsec){
    if (viscosity_array[i] != 0.0){
      Qplus[i*nsec + j] = 0.5/viscosity_array[i]/Dens[i*nsec + j]*(TAURR[i*nsec + j]*TAURR[i*nsec + j] + \
        TAURP[i*nsec + j]* TAURP[i*nsec + j] + TAUPP[i*nsec + j]*TAUPP[i*nsec + j]);

      Qplus[i*nsec + j] += (2.0/9.0)*viscosity_array[i]*Dens[i*nsec + j]*DivergenceVelocity[i*nsec + j]* \
        DivergenceVelocity[i*nsec + j];
    }
    else Qplus[i*nsec + j] = 0.0;

  }
  __syncthreads();
  i-= 1;
  if (i==0 && j<nsec){
    /* We calculate the heating source term Qplus for i=0 */

    if (viscosity_array[i] != 0) {
      /* power-law extrapolation */
      Qplus[i*nsec + j] = Qplus[(i+1)*nsec + j]*expf(logf(Qplus[(i+1)*nsec + j]/Qplus[(i+2)*nsec + j]) * \
        logf(Rmed[i]/Rmed[i+1]) / logf(Rmed[i+1]/Rmed[i+2]));
    }
    else Qplus[i*nsec + j] = 0.0;
  }

  __syncthreads();

  if (i<nrad && j<nsec){
    if (!Cooling){
      num = dt*Qplus[i*nsec + j] + Energy[i*nsec + j];
      den = 1.0+(ADIABATICINDEX-1.0)*dt*DivergenceVelocity[i*nsec + j];
      EnergyNew[i*nsec + j] = num/den;
    }
    else{
      num = EnergyMed[i]*dt*Dens[i*nsec + j]/SigmaMed[i] + CoolingTimeMed[i]*Energy[i*nsec + j] + \
        dt*CoolingTimeMed[i]*(Qplus[i*nsec + j]-QplusMed[i]*Dens[i*nsec + j]/SigmaMed[i]);

      den = dt + CoolingTimeMed[i] + (ADIABATICINDEX-1.0)*dt*CoolingTimeMed[i]*DivergenceVelocity[i*nsec + j];
      EnergyNew[i*nsec + j] = num/den;
    }
  }
}

__global__ void UpdateVelocitiesKernel (float *VthetaInt, float *VradInt, float *invRmed, float *Rmed, float *Rsup,
  float *Rinf, float *invdiffRmed, float *invdiffRsup, float *Dens, float *invRinf, float *TAURR, float *TAURP,
  float *TAUPP, float DeltaT, int nrad, int nsec)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = threadIdx.y + blockDim.y*blockIdx.y;

  i+=1;
  /* Now we can update velocities
     with the viscous source term
     of Navier-Stokes equation */

  /* vtheta first */
  if (i<nrad-1 && j<nsec){
    VthetaInt[i*nsec +j] += DeltaT*invRmed[i]*((Rsup[i]*TAURP[(i+1)*nsec+ j]-Rinf[i]*TAURP[i*nsec +j])*invdiffRsup[i] + \
    (TAUPP[i*nsec +j]-TAUPP[i*nsec + ((j-1)+nsec)%nsec])*1.0/(2.0*M_PI/nsec) + 0.5*(TAURP[i*nsec + j] + TAURP[(i+1)*nsec +j]))/ \
    (0.5*(Dens[i*nsec +j]+Dens[i*nsec + ((j-1)+nsec)%nsec]));
  }

  /* now vrad */
  if (i<nrad && j<nsec){
    VradInt[i*nsec +j] += DeltaT*invRinf[i]*((Rmed[i]*TAURR[i*nsec +j]- Rmed[i-1]*TAURR[(i-1)*nsec + j])*invdiffRmed[i] + \
    (TAURP[i*nsec + (j+1)%nsec] -TAURP[i*nsec + j])*1.0/(2.0*M_PI/nsec) - 0.5*(TAURP[i*nsec +j] + TAURP[(i-1)*nsec + j]))/ \
    (0.5*(Dens[i*nsec +j] + Dens[(i-1)*nsec + j]));

  }
}

__global__ void InitComputeAccelKernel (float *CellAbscissa, float *CellOrdinate, float *Rmed, int nsec, int nrad)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = threadIdx.y + blockDim.y*blockIdx.y;

  if (i<nrad && j<nsec){
    CellAbscissa[i*nsec+j] = Rmed[i] * cosf((2.0*M_PI*(float)j)/(float)nsec);
    CellOrdinate[i*nsec+j] = Rmed[i] * sinf((2.0*M_PI*(float)j)/(float)nsec);
  }
}

__global__ void ComputeSoundSpeedKernel (float *SoundSpeed, float *Dens, float *Rmed, float *Energy, int nsec, int nrad,
  int Adiabatic, float ADIABATICINDEX, float FLARINGINDEX, float ASPECTRATIO, float TRANSITIONWIDTH,
  float TRANSITIONRADIUS, float TRANSITIONRATIO, float PhysicalTime, float PhysicalTimeInitial, float LAMBDADOUBLING)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = threadIdx.y + blockDim.y*blockIdx.y;

  float AspectRatio;
  if (i<nrad && j<nsec){
    if (!Adiabatic){
      AspectRatio = AspectRatioDevice(Rmed[i], ASPECTRATIO, TRANSITIONWIDTH, TRANSITIONRADIUS, TRANSITIONRATIO,
        PhysicalTime, PhysicalTimeInitial, LAMBDADOUBLING);
      SoundSpeed[i*nsec + j] = AspectRatio*sqrtf(G*1.0/Rmed[i])*powf(Rmed[i], FLARINGINDEX);
    }
    else SoundSpeed[i*nsec + j] = sqrtf(ADIABATICINDEX*(ADIABATICINDEX-1.0)*Energy[i*nsec + j]/Dens[i*nsec + j]);
  }
}

__global__ void ComputePressureFieldKernel (float *SoundSpeed, float *Dens, float *Pressure, int Adiabatic, int nrad,
  int nsec, float ADIABATICINDEX, float *Energy) /* LISTO */
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = threadIdx.y + blockDim.y*blockIdx.y;

  if (i<nrad && j<nsec){
    if (!Adiabatic) Pressure[i*nsec + j] = Dens[i*nsec + j]*SoundSpeed[i*nsec + j]*SoundSpeed[i*nsec + j];
    /* Since SoundSpeed is not update from initialization, cs remains axisymmetric*/
    else Pressure[i*nsec + j] = (ADIABATICINDEX-1.0)*Energy[i*nsec + j];
  }
}


__global__ void ComputeTemperatureFieldKernel (float *Dens, float *Temperature, float *Pressure, float *Energy,
  float ADIABATICINDEX, int Adiabatic, int nsec, int nrad) /* LISTO */
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = threadIdx.y + blockDim.y*blockIdx.y;

  if (i<nrad && j<nsec){
    if (!Adiabatic) Temperature[i*nsec + j] = MU/R*Pressure[i*nsec + j]/Dens[i*nsec + j];
    else Temperature[i*nsec + j] = MU/R*(ADIABATICINDEX-1.0)*Energy[i*nsec + j]/Dens[i*nsec + j];
  }
}


/* LISTO */
__global__ void InitLabelKernel (float *Label, float xp, float yp, float rhill, float *Rmed, int nrad, int nsec)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = threadIdx.y + blockDim.y*blockIdx.y;

  if (i<nrad && j<nsec){
    float distance, angle, x, y;
    angle = (float)j / (float)nsec*2.0*M_PI;
    x = Rmed[i] * cosf(angle);
    y = Rmed[i] * sinf(angle);
    distance = sqrtf((x - xp) * (x - xp) + (y - yp)*(y -yp));

    if (distance < rhill) Label[i*nsec + j] = 1.0;
    else Label[i*nsec + j] = 0.0;

  }
}

__global__ void CircumPlanetaryMassKernel (float *Dens, float *Surf, float *CellAbscissa, float *CellOrdinate,
  float xpl, float ypl, int nrad, int nsec, float HillRadius, float *mdcp0) /* LISTA */
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = threadIdx.y + blockDim.y*blockIdx.y;

  float dist;

  if (i<nrad && j<nsec){
    dist = sqrtf((CellAbscissa[i*nsec + j]-xpl)*(CellAbscissa[i*nsec + j]-xpl) + (CellOrdinate[i*nsec + j]-ypl) * \
    (CellOrdinate[i*nsec + j]-ypl));
    if (dist < HillRadius) mdcp0[i*nsec + j] =  Surf[i]* Dens[i*nsec + j];
    else mdcp0[i*nsec + j] = 0.0;
  }
}

template <bool nIsPow2>
__global__ void DeviceReduceKernel (float *g_idata, float *g_odata, unsigned int n)
{
  extern __shared__ float sdata[];

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int blockSize = blockDim.x;
  unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
  unsigned int gridSize = blockSize*2*gridDim.x;

  float mySum = 0.f;

  // we reduce multiple elements per thread.  The number is determined by the
  // number of active thread blocks (via gridDim).  More blocks will result
  // in a larger gridSize and therefore fewer elements per thread
  while (i < n){
    mySum += g_idata[i];

    // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
    if (nIsPow2 || i + blockSize < n)
      mySum += g_idata[i+blockSize];

    i += gridSize;
  }
  // each thread puts its local sum into shared memory
  sdata[tid] = mySum;
  __syncthreads();

  // do reduction in shared mem
  if ((blockSize >= 512) && (tid < 256)){
    sdata[tid] = mySum = mySum + sdata[tid + 256];
  }
  __syncthreads();

  if ((blockSize >= 256) &&(tid < 128)){
    sdata[tid] = mySum = mySum + sdata[tid + 128];
  }
   __syncthreads();

  if ((blockSize >= 128) && (tid <  64)){
     sdata[tid] = mySum = mySum + sdata[tid +  64];
  }
  __syncthreads();

  #if (__CUDA_ARCH__ >= 300 )
    if (tid < 32){
      // Fetch final intermediate sum from 2nd warp
      if (blockSize >=  64) mySum += sdata[tid + 32];
        // Reduce final warp using shuffle
        for (int offset = warpSize/2; offset > 0; offset /= 2){
          mySum += __shfl_down(mySum, offset);
        }
      }
  #else
    // fully unroll reduction within a single warp
    if ((blockSize >=  64) && (tid < 32)){
      sdata[tid] = mySum = mySum + sdata[tid + 32];
    }
    __syncthreads();

    if ((blockSize >=  32) && (tid < 16)){
      sdata[tid] = mySum = mySum + sdata[tid + 16];
    }
    __syncthreads();

    if ((blockSize >=  16) && (tid <  8)){
      sdata[tid] = mySum = mySum + sdata[tid +  8];
    }
    __syncthreads();

    if ((blockSize >=   8) && (tid <  4)){
      sdata[tid] = mySum = mySum + sdata[tid +  4];
    }
    __syncthreads();

    if ((blockSize >=   4) && (tid <  2)){
      sdata[tid] = mySum = mySum + sdata[tid +  2];
    }
    __syncthreads();

    if ((blockSize >=   2) && ( tid <  1)){
      sdata[tid] = mySum = mySum + sdata[tid +  1];
    }
    __syncthreads();
  #endif

  // write result for this block to global mem
  if (tid == 0) g_odata[blockIdx.x] = mySum;
}

__host__ long NearestPowerOf2 (long n)
{
  if(!n) return n; //(0 ==2^0)

  int x=1;
  while (x < n){
    x<<=1;
  }
  return x;
}

__host__ bool IsPow2 (unsigned int x)
{
  return ((x&(x-1)==0));
}

__host__ float DeviceReduce (float *in, int N)
{
  float *device_out;
  gpuErrchk(cudaMalloc(&device_out, sizeof(float)*1024));
  gpuErrchk(cudaMemset(device_out, 0, sizeof(float)*1024));

  int threads = 32;
  int blocks = min((int(NearestPowerOf2(N)) + threads - 1) / threads, 1024);
  int smemSize = (threads <= 32) ? 2 * threads * sizeof(float) : threads * sizeof(float);

  bool isPower2 = IsPow2(N);
  if(isPower2){
    DeviceReduceKernel<true><<<blocks, threads, smemSize>>>(in, device_out, N);
    gpuErrchk(cudaDeviceSynchronize());
  }
  else{
    DeviceReduceKernel<false><<<blocks, threads, smemSize>>>(in, device_out, N);
    gpuErrchk(cudaDeviceSynchronize());
  }

  float *h_odata = (float *) malloc(blocks*sizeof(float));
  float sum = 0.0;

  gpuErrchk(cudaMemcpy(h_odata, device_out, blocks * sizeof(float),cudaMemcpyDeviceToHost));
  for (int i=0; i<blocks; i++){
    sum += h_odata[i];
  }
  cudaFree(device_out);
  free(h_odata);
	return sum;
}


/* LISTA */
__global__ void MultiplyPolarGridbyConstantKernel (float *Dens, int nrad, int nsec, float ScalingFactor)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = threadIdx.y + blockDim.y*blockIdx.y;

  if (i<nrad+1 && j<nsec)
    Dens[i*nsec + j] *= ScalingFactor;
}

__global__ void Substep2Kernel (float *Dens, float *VradInt, float *VthetaInt, float *TemperInt, int nrad,
  int nsec, float CVNR, float *invdiffRmed, float *invdiffRsup, float *DensInt, int Adiabatic, float *Rmed,
  float dt, float *VradNew, float *VthetaNew, float *Energy, float *EnergyInt)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = threadIdx.y + blockDim.y*blockIdx.y;

  float dv;
  if (i<nrad && j<nsec){
    dv = VradInt[(i+1)*nsec + j] - VradInt[i*nsec + j];

    if (dv < 0.0) DensInt[i*nsec + j] = CVNR*CVNR*Dens[i*nsec + j]*dv*dv;
    else DensInt[i*nsec + j] = 0.0;

    dv = VthetaInt[i*nsec + (j+1)%nsec] - VthetaInt[i*nsec + j];
    if (dv < 0.0) TemperInt[i*nsec + j] = CVNR*CVNR*Dens[i*nsec + j]*dv*dv;
    else TemperInt[i*nsec + j] = 0.0;
  }
  i+=1;

  if (i<nrad && j<nsec){
    VradNew[i*nsec + j] = VradInt[i*nsec + j] - dt*2.0/(Dens[i*nsec + j] + Dens[(i-1)*nsec + j])*(DensInt[i*nsec + j] - DensInt[(i-1)*nsec + j]) *\
      invdiffRmed[i];
  }
  i-=1;

  if (i<nrad && j<nsec){
    VthetaNew[i*nsec + j] = VthetaInt[i*nsec + j] - dt*2.0/(Dens[i*nsec + j] + Dens[i*nsec + ((j-1)+nsec)%nsec])*(DensInt[i*nsec + j] - \
      DensInt[(i-1)*nsec + j])* 1.0/(2.0*M_PI*Rmed[i]/nsec);
  }
  if (Adiabatic){
    i+=1;

    if (i<nrad && j<nsec){
      EnergyInt[i*nsec + j] = Energy[i*nsec + j] - dt*DensInt[i*nsec + j]*(VradInt[i*nsec + j+1] - VradInt[i*nsec + j])*invdiffRsup[i] - \
      dt*TemperInt[i*nsec + j]*(VthetaInt[i*nsec + (j+1)%nsec] - VthetaInt[i*nsec + j])* 1.0/(2.0*M_PI*Rmed[i]/nsec);
    }
  }
}

__global__ void OpenBoundaryKernel (float *Vrad, float *Dens, float *Energy, int nsec, float *SigmaMed)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = 1;

  if(j < nsec){
    Dens[(i-1)*nsec + j] = Dens[i*nsec + j]; // copy first ring into ghost ring
    Energy[(i-1)*nsec + j] = Energy[i*nsec + j];
    if (Vrad[(i+1)*nsec + j] > 0.0 || (Dens[i*nsec + j] < SigmaMed[0]))
      Vrad[i*nsec + j] = 0.0; // we just allow outflow [inwards]
    else
      Vrad[i*nsec +j] = Vrad[(i+1)*nsec + j];
  }
}

__global__ void ReduceCsKernel (float *SoundSpeed, float *cs0, float *cs1, float *csnrm1, float *csnrm2, int nsec, int nrad)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i=0;

  if(j<nsec){
    cs0[j] = SoundSpeed[i*nsec +j];
    cs1[j] = SoundSpeed[(i+1)*nsec +j];
  }
  i = nrad-1;
  if(j<nsec){
    csnrm2[j] = SoundSpeed[(i-1)*nsec +j];
    csnrm1[j] = SoundSpeed[i*nsec +j];
  }
}

__global__ void ReduceMeanKernel (float *Dens, float *Energy, int nsec, float *mean_dens, float *mean_energy,
  float *mean_dens2, float *mean_energy2, int nrad)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = 0;

  if(j<nsec){
    mean_dens[j] = Dens[i*nsec+ j];
    mean_energy[j] = Energy[i*nsec +j];
  }
  i = nrad-1;
  if(j<nsec){
    mean_dens2[j] = Dens[i*nsec + j];
    mean_energy2[j] = Energy[i*nsec + j];
  }
}

__global__ void NonReflectingBoundaryKernel (float *Dens, float *Energy, int i_angle, int nsec, float *Vrad, float *SoundSpeed,
  float SigmaMed, int nrad, float SigmaMed2, int i_angle2)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = 1;
  float Vrad_med;

  if (j<nsec){

    if(j+i_angle >= nsec){
      Dens[j+i_angle - nsec] = Dens[i*nsec + j];
      Energy[j+i_angle - nsec] = Energy[i*nsec + j];
    }

    else if(j+i_angle < 0){
      Dens[j+i_angle + nsec] = Dens[i*nsec + j];
      Energy[j+i_angle + nsec] = Energy[i*nsec + j];
    }
    else{
      Dens[j+i_angle] = Dens[i*nsec + j];
      Energy[j+i_angle] = Energy[i*nsec + j];
    }
    __syncthreads();

    Vrad_med = -SoundSpeed[i*nsec + j]*(Dens[i*nsec + j]-SigmaMed)/SigmaMed;
    Vrad[i*nsec + j] = 2.*Vrad_med-Vrad[(i+1)*nsec + j];
  }

  i = nrad-1;
  if (j<nsec){
    if (j-i_angle2 >= nsec){
      Dens[i*nsec + j] = Dens[j-i_angle2 + (i-2)*nsec ];
      Energy[i*nsec + j] = Energy[j-i_angle2 + (i-2)*nsec ];
    }
    else if (j-i_angle2 < 0){
      Dens[i*nsec + j] = Dens[j-i_angle2 + i*nsec];
      Energy[i*nsec + j] = Energy[j-i_angle2 + i*nsec];
    }
    else{
      Dens[i*nsec + j] = Dens[j-i_angle2 + (i-1)*nsec];
      Energy[i*nsec + j] = Energy[j-i_angle2 + (i-1)*nsec];
    }
    __syncthreads();

    Vrad_med = SoundSpeed[i*nsec + j]*(Dens[(i-1)*nsec + j]-SigmaMed2)/SigmaMed2;
    Vrad[i*nsec + j] = 2.*Vrad_med - Vrad[(i-1)*nsec + j];
  }
}

__global__ void MinusMeanKernel (float *Dens, float *Energy, float SigmaMed, float mean_dens_r, float mean_dens_r2,
  float mean_energy_r,float mean_energy_r2, float EnergyMed, int nsec, int nrad, float SigmaMed2, float EnergyMed2)
{
    int j = threadIdx.x + blockDim.x*blockIdx.x;
    int i = 0;
    if (j< nsec){
      Dens[i*nsec + j] += SigmaMed - mean_dens_r;
      Energy[i*nsec + j] += EnergyMed - mean_energy_r;
    }

    i = nrad-1;
    if (j < nsec){
      Dens[i*nsec + j] += SigmaMed2 - mean_dens_r2;
      Energy[i*nsec + j] += EnergyMed2 - mean_energy_r2;
    }
  }

__global__ void Make1DprofileKernel (float *gridfield, float *axifield, int nsec, int nrad)
{
  int i = threadIdx.x + blockDim.x*blockIdx.x;
  int j;

  if (i < nrad){
    float sum = 0.0;

    for (j = 0; j < nsec; j++)
      sum += gridfield[i*nsec + j];

    axifield[i] = sum/(float)nsec;
  }
}


__host__ void Make1Dprofile (int option)
{

  /* GLOBAL AxiSGAccr option */
  if (option == 1){
    gpuErrchk(cudaMemcpy(gridfield_d, SG_Accr_d, size_grid*sizeof(float), cudaMemcpyDeviceToDevice));
    //gpuErrchk(cudaMemcpy(GLOBAL_AxiSGAccr_d, axifield_d, NRAD*sizeof(float), cudaMemcpyDeviceToHost));

  }
  /* GLOBAL_bufarray option */
  if (option == 2){
    //gpuErrchk(cudaMemcpy(gridfield_d, SG_Accr_d, size_grid*sizeof(float), cudaMemcpyDeviceToDevice));
    //gpuErrchk(cudaMemcpy(GLOBAL_AxiSGAccr_d, axifield_d, NRAD*sizeof(float), cudaMemcpyDeviceToHost));
  }

  Make1DprofileKernel<<<dimGrid4, dimBlock>>>(gridfield_d, axifield_d, NSEC, NRAD);
  gpuErrchk(cudaDeviceSynchronize());

}

/* LISTO */
__global__ void InitGasVelocitiesKernel (float *viscosity_array, int nsec, int nrad, int SelfGravity, float *Rmed,
  float ASPECTRATIO, float FLARINGINDEX, float SIGMASLOPE, int CentrifugalBalance, float *Vrad, float *Vtheta,
  float ViscosityAlpha, float IMPOSEDDISKDRIFT, float SIGMA0, float *SigmaInf, float OmegaFrame, float *Rinf, float *vt_cent)
{
    int j = threadIdx.x + blockDim.x*blockIdx.x;
    int i = threadIdx.y + blockDim.y*blockIdx.y;

    float omega, r, ri;

    if (i < nrad+1 && j < nsec){
      if (i == nrad){
        r = Rmed[nrad - 1];
        ri = Rinf[nrad - 1];
      }
      else{
        r = Rmed[i];
        ri = Rinf[i];
      }

      if (!SelfGravity){
        omega = sqrtf(G*1.0/r/r/r);
        Vtheta[i*nsec + j] = omega*r*sqrtf(1.0-powf(ASPECTRATIO,2.0)*powf(r,2.0*FLARINGINDEX)* \
        (1.+SIGMASLOPE-2.0*FLARINGINDEX));
      }
      Vtheta[i*nsec + j] -= OmegaFrame*r;

      if (CentrifugalBalance) Vtheta[i*nsec + j] = vt_cent[i];

      if (i == nrad) Vrad[i*nsec + j] = 0.0;
      else Vrad[i*nsec + j] = IMPOSEDDISKDRIFT*SIGMA0/SigmaInf[i]/ri;

      if (ViscosityAlpha) Vrad[i*nsec + j] -= 3.0*viscosity_array[i]/r*(-SIGMASLOPE+2.0*FLARINGINDEX+1.0);
      else Vrad[i*nsec + j] -= 3.0*viscosity_array[i]/r*(-SIGMASLOPE+.5);

      __syncthreads();

      if (j < nsec) Vrad[j] = Vrad[nrad*nsec + j] = 0.0;
    }
  }



__global__ void ComputeForceKernel (float *CellAbscissa, float *CellOrdinate, float *Surf, float *Dens, float x,
  float y, float rsmoothing, float *forcesxi, float *forcesyi, float *forcesxo, float *forcesyo, int nsec,
  int nrad, float a, float *Rmed, int dimfxy, float rh)
{

    int j = threadIdx.x + blockDim.x*blockIdx.x;
    int i = threadIdx.y + blockDim.y*blockIdx.y;
    int k;
    float cellmass, dx, dy, d2, InvDist3, dist2, distance, resultxi, resultyi;
    float resultxo, resultyo, hillcutfactor, hill_cut;

    if (i<nrad && j<nsec){
      cellmass = Surf[i] * Dens[i*nsec + j];
      dx = CellAbscissa[i*nsec + j] - x;
      dy = CellOrdinate[i*nsec + j] - y;
      d2 = dx*dx + dy*dy;
      dist2 = d2 + rsmoothing*rsmoothing;
      distance = sqrtf(dist2);
      InvDist3 = 1.0/dist2/distance;

      for (k = 0; k < dimfxy; k++){
        hillcutfactor =  (float) k / (float)(dimfxy-1);
        if (k != 0){
          rh *= hillcutfactor;
          hill_cut = 1.-expf(-d2/(rh*rh));
        }
        else hill_cut = 1.;

        if (Rmed[i] < a){
          resultxi = G * cellmass * dx * InvDist3 * hill_cut;
          resultyi = G * cellmass * dy * InvDist3 * hill_cut;
          atomicAdd(&(forcesxi[k]), resultxi);
          atomicAdd(&(forcesyi[k]), resultyi);
        }
        else{
          resultxo = G * cellmass * dx * InvDist3 * hill_cut;
          resultyo = G * cellmass * dy * InvDist3 * hill_cut;
          atomicAdd(&(forcesxo[k]), resultxo);
          atomicAdd(&(forcesyo[k]), resultyo);
        }
      }
    }
  }

__global__ void ViscousTermsKernel (float *Vradial, float *Vazimutal , float *DRR, float *DPP, float *DivergenceVelocity,
  float *DRP, float *invdiffRsup, float *invRmed, float *Rsup, float *Rinf, float *invdiffRmed, int nrad,
  int nsec, float *TAURR, float *TAUPP, float *dens, float *viscosity_array, float *TAURP, float *invRinf)
{
   int j = threadIdx.x + blockDim.x*blockIdx.x;
   int i = threadIdx.y + blockDim.y*blockIdx.y;

   float dphi, invdphi, onethird;
   if (i<nrad && j<nsec){ /* Drr, Dpp and divV computation */
     dphi = 2.0*M_PI/(float)nsec;
     invdphi = 1.0/dphi;
     onethird = 1.0/3.0;

     DRR[i*nsec + j] = (Vradial[(i+1)*nsec + j] - Vradial[i*nsec + j])*invdiffRsup[i];
     DPP[i*nsec + j] = (Vazimutal[i*nsec + (j+1)%nsec] - Vazimutal[i*nsec + j])*invdphi*invRmed[i]+0.5* \
       (Vradial[(i+1)*nsec + j]+Vradial[i*nsec + j])*invRmed[i];
     DivergenceVelocity[i*nsec + j] = (Vradial[i*nsec + (j+1)]*Rsup[i]-Vradial[i*nsec + j]*Rinf[i])*invdiffRsup[i] * \
       invRmed[i];
     DivergenceVelocity[i*nsec + j] += (Vazimutal[i*nsec + (j+1)%nsec]-Vazimutal[i*nsec + j])*invdphi*invRmed[i];

     if (i > 0) DRP[i*nsec + j] = 0.5*(Rinf[i]*(Vazimutal[i*nsec + j]*invRmed[i]-Vazimutal[(i-1)*nsec + j])*invRmed[i-1])* \
          invdiffRmed[i] + (Vradial[i*nsec + j]-Vradial[i*nsec + (j-1)%nsec])*invdphi*invRinf[i];
   }

   if (i<nrad && j<nsec){ /* TAUrr and TAUpp computation */
     TAURR[i*nsec + j] = 2.0*dens[i*nsec + j]*viscosity_array[i]*(DRR[i*nsec + j]-onethird*DivergenceVelocity[i*nsec + j]);
     TAUPP[i*nsec + j] = 2.0*dens[i*nsec + j]*viscosity_array[i]*(DPP[i*nsec + j]-onethird*DivergenceVelocity[i*nsec + j]);

     if (i > 0) TAURP[i*nsec + j] = 2.0*0.25*(dens[i*nsec + j] + dens[(i-1)*nsec + j] + dens[(i-1)*nsec + ((j-1)+nsec)%nsec])* \
         viscosity_array[i]*DRP[i*nsec + j];
   }
 }

__global__ void LRMomentaKernel (float *RadMomP, float *RadMomM, float *ThetaMomP, float *ThetaMomM, float *Dens,
  float *Vrad, float *Vtheta, int nrad, int nsec, float *Rmed, float OmegaFrame)
{
   int j = threadIdx.x + blockDim.x*blockIdx.x;
   int i = threadIdx.y + blockDim.y*blockIdx.y;

   if (i<nrad && j<nsec){
     RadMomP[i*nsec + j] = Dens[i*nsec + j] * Vrad[(i)*nsec + j]; // (i+1)*nsec
     RadMomM[i*nsec + j] = Dens[i*nsec + j] * Vrad[i*nsec + j];
     /* it is the angular momentum -> ThetaMomP */
     ThetaMomP[i*nsec + j] = Dens[i*nsec + j] * (Vtheta[i*nsec + (j+1)%nsec]+Rmed[i]*OmegaFrame)*Rmed[i];
     ThetaMomM[i*nsec + j] = Dens[i*nsec + j] * (Vtheta[i*nsec + j]+Rmed[i]*OmegaFrame)*Rmed[i];
   }
 }

__global__ void ExtQtyKernel (float *ExtLabel, float *Dens, float *Label, int nsec, int nrad)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = threadIdx.y + blockDim.y*blockIdx.y;

  if (i<nrad && j<nsec)
    ExtLabel[i*nsec + j] = Dens[i*nsec + j]*Label[i*nsec + j];
}

__global__ void StarRadKernel (float *Qbase, float *Vrad, float *QStar, float dt, int nrad, int nsec,
  float *invdiffRmed, float *Rmed, float *dq)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = threadIdx.y + blockDim.y*blockIdx.y;

  float dqm, dqp;

  if (i<nrad && j<nsec){
    if (i == 0 || i == nrad-1) dq[j*nrad + i] = 0.0;
    else {
      dqm = (Qbase[i*nsec + j] - Qbase[(i-1)*nsec + j])*invdiffRmed[i];
      dqp = (Qbase[(i+1)*nsec + j] - Qbase[i*nsec + j])*invdiffRmed[i+1];

      if (dqp * dqm > 0.0) dq[j*nrad + i] = 2.0*dqp*dqm/(dqp+dqm);
      else dq[j*nrad + i] = 0.0;
    }
    __syncthreads();

    if (Vrad[i*nsec + j] > 0.0) // aca asumen vrad = 0 para i=0?
      QStar[i*nsec + j] = Qbase[(i-1)*nsec + j] + (Rmed[i]-Rmed[i-1]-Vrad[i*nsec + j]*dt)*0.5*dq[j*nrad + i-1];
    else
      QStar[i*nsec + j] = Qbase[i*nsec + j]-(Rmed[i+1]-Rmed[i]+Vrad[i*nsec + j]*dt)*0.5*dq[j*nrad + i];
  }
  if (j<nsec)
    QStar[j] = QStar[j+nsec*nrad] = 0.0;
}

__global__ void ComputeFFTKernel (float *Radii, cufftComplex *SGP_Kr, cufftComplex *SGP_Kt, float SGP_eps, int nrad, int nsec,
cufftComplex *SGP_Sr, cufftComplex *SGP_St, float *Dens, float *Rmed, float *Kr_aux, float *Kt_aux)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = threadIdx.y + blockDim.y*blockIdx.y;
  float u, cosj, sinj, coshu, expu, den_SGP_K, theta, base;
  float a, var, radii;

  if (i<2*nrad && j<nsec){
    SGP_Kr[i*nsec + j].x = Kr_aux[i*nsec + j];
    SGP_Kr[i*nsec + j].y = 0.;

    SGP_Kt[i*nsec + j].x = Kt_aux[i*nsec + j];
    SGP_Kt[i*nsec + j].y = 0.;

    SGP_Sr[i*nsec + j].y = 0.;
    SGP_St[i*nsec + j].y = 0.;

    if (i<nrad){
      var = Dens[i*nsec + j] * sqrtf(Rmed[i]/Rmed[0]);
      SGP_Sr[i*nsec + j].x = var;
      SGP_St[i*nsec + j].x = var*(Rmed[i]/Rmed[0]);
    }
    else{
      SGP_Sr[i*nsec + j].x = 0.;
      SGP_St[i*nsec + j].x = 0.;
    }
  }
}

__global__ void ComputeConvolutionKernel (cufftComplex *Gr, cufftComplex *Gphi, cufftComplex *SGP_Kr, cufftComplex *SGP_Kt,
  cufftComplex *SGP_Sr, cufftComplex *SGP_St, int nsec, int nrad)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = threadIdx.y + blockDim.y*blockIdx.y;

  if (i<2*nrad && j<nsec){
    Gphi[i*nsec + j].x = -G*(SGP_Kt[i*nsec + j].x * SGP_St[i*nsec + j].x - \
      SGP_Kt[i*nsec + j].y * SGP_St[i*nsec + j].y);

    Gphi[i*nsec + j].y = -G*(SGP_Kt[i*nsec + j].x * SGP_St[i*nsec + j].y + \
      SGP_Kt[i*nsec + j].y * SGP_St[i*nsec + j].x);

    Gr[i*nsec + j].x = -G*(SGP_Kr[i*nsec + j].x * SGP_Sr[i*nsec + j].x - \
      SGP_Kr[i*nsec + j].y * SGP_Sr[i*nsec + j].y);

    Gr[i*nsec + j].y = -G*(SGP_Kr[i*nsec + j].x * SGP_Sr[i*nsec + j].y + \
      SGP_Kr[i*nsec + j].y *SGP_Sr[i*nsec + j].x);
  }
}

__global__ void ComputeSgAccKernel (float *SG_Accr, float *SG_Acct, float *Dens , float SGP_rstep, float SGP_tstep,
  float SGP_eps, int nrad, int nsec, float *Rmed, cufftComplex *Gr, cufftComplex *Gphi)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = threadIdx.y + blockDim.y*blockIdx.y;

  float normaccr, normacct, divRmed;

  if (i<nrad && j<nsec){
    divRmed = Rmed[i]/Rmed[0];
    normaccr = SGP_rstep * SGP_tstep / ((float)(2*nrad) * (float)nsec);
    normacct = normaccr;
    normaccr /= sqrtf(divRmed);
    normacct /= (divRmed * sqrtf(divRmed));
    SG_Acct[i*nsec + j] = Gphi[i*nsec + j].x * normaccr;

    SG_Accr[i*nsec + j] = Gr[i*nsec + j].x * normaccr;
    SG_Accr[i*nsec + j] += G*Dens[i*nsec + j]*SGP_rstep*SGP_tstep / SGP_eps;
  }
}

__global__ void Update_sgvelocityKernel (float *Vradial, float *Vazimutal, float *SG_Accr, float *SG_Acct, float *Rinf,
  float *Rmed, float *invdiffRmed, float dt, int nrad, int nsec)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = threadIdx.y + blockDim.y*blockIdx.y;

  int jm1, lm1;

  /* Here we update velocity fields to take into acount self-gravity */
  if (i<nrad && j<nsec){
    /* We compute VRAD - half-centered in azimuth - from centered-in-cell radial sg acceleration. */
    if (i > 0) Vradial[i*nsec + j] +=  dt*((Rinf[i] - Rmed[i-1]) * SG_Accr[i*nsec + j] + \
    (Rmed[i] - Rinf[i]) * SG_Accr[(i-1)*nsec + j]) *invdiffRmed[i]; // caso !SGZeroMode

    /* We compute VTHETA - half-centered in radius - from centered-in-cell azimutal sg acceleration. */
    Vazimutal[i*nsec + j] = 0.5 * dt * (SG_Acct[i*nsec + j] + SG_Acct[i*nsec + (j-1)%nsec]);
  }
}

__global__ void Azimutalvelocity_withSGKernel (float *Vtheta, float *Rmed, float FLARINGINDEX, float SIGMASLOPE,
  float ASPECTRATIO, float *GLOBAL_bufarray, int nrad, int nsec)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = threadIdx.y + blockDim.y*blockIdx.y;

  float omegakep, omega, invr;
  if (i<nrad && j<nsec){
    invr = 1./Rmed[i];
    omegakep = sqrtf(G*1.0*invr*invr*invr);
    omega = sqrtf(omegakep*omegakep* (1.0 - (1.+SIGMASLOPE-2.0*FLARINGINDEX)*powf(ASPECTRATIO,2.0)* \
      powf(Rmed[i],2.0*FLARINGINDEX)) - invr*GLOBAL_bufarray[i]);

    Vtheta[i*nsec + j] = Rmed[i]*omega;
  }
}

__global__ void CrashKernel (float *array, int nrad, int nsec, int Crash)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = threadIdx.y + blockDim.y*blockIdx.y;

  if (i<nrad && j<nsec){
    if (array[i*nsec + j] < 0.0)
      array[i*nsec + j] = 1.0;
    else
      array[i*nsec + j] = 0.0;
  }
}

__global__ void EvanescentBoundaryKernel(float *Rmed, float *Vrad, float *Vtheta, float *Energy, float *Dens,
  float *viscosity_array, float DRMIN, float DRMAX, int nrad, int nsec, float Tin,
  float Tout, float step, float SIGMASLOPE, float FLARINGINDEX, float *GLOBAL_bufarray, float OmegaFrame,
  float *SigmaMed, float *EnergyMed, int Adiabatic, int SelfGravity, float ASPECTRATIO, float TRANSITIONWIDTH,
  float TRANSITIONRADIUS, float TRANSITIONRATIO, float PhysicalTime, float PhysicalTimeInitial, float LAMBDADOUBLING)
{
    int j = threadIdx.x + blockDim.x*blockIdx.x;
    int i = threadIdx.y + blockDim.y*blockIdx.y;

    float damping, lambda, vtheta0, vrad0, energy0, dens0, AspectRatio;
    if (i<nrad && j<nsec){
      if ((Rmed[i] < DRMIN) || (Rmed[i] > DRMAX)){
        /* Damping operates only inside the wave killing zones */
        if(Rmed[i] < DRMIN){
          damping = (Rmed[i]-DRMIN)/(Rmed[0]-DRMIN);
          lambda = damping*damping*10.0*step/Tin;
        }
        if (Rmed[i] > DRMAX){
          damping = (Rmed[i]-DRMAX)/(Rmed[nrad-1]-DRMAX);
          lambda = damping*damping*10.0*step/Tout;
        }
        if(!SelfGravity){
          AspectRatio = AspectRatioDevice(Rmed[i], ASPECTRATIO, TRANSITIONWIDTH, TRANSITIONRADIUS, TRANSITIONRATIO,
            PhysicalTime, PhysicalTimeInitial, LAMBDADOUBLING);
          vtheta0 = sqrtf(G*1.0/Rmed[i] * (1.0 - (1.0+SIGMASLOPE-2.0*FLARINGINDEX)*powf(AspectRatio,2.0) * \
          powf(Rmed[i],2.0*FLARINGINDEX)));
        }
        if (SelfGravity){
          AspectRatio = AspectRatioDevice(Rmed[i], ASPECTRATIO, TRANSITIONWIDTH, TRANSITIONRADIUS, TRANSITIONRATIO,
            PhysicalTime, PhysicalTimeInitial, LAMBDADOUBLING);
          vtheta0 = sqrtf(G*1.0/Rmed[i] * (1.0 - (1.0+SIGMASLOPE-2.0*FLARINGINDEX)*powf(AspectRatio,2.0) * \
          powf(Rmed[i],2.0*FLARINGINDEX)) - Rmed[i]*GLOBAL_bufarray[i]);
        }
        /* this could be refined if CentrifugalBalance is used... */
        vtheta0 -= Rmed[i]*OmegaFrame;
        vrad0 = -3.0*viscosity_array[i]/Rmed[i]*(-SIGMASLOPE+.5);
        dens0 = SigmaMed[i];
        energy0 = EnergyMed[i];

        Vrad[i*nsec + j] = (Vrad[i*nsec + j] + lambda*vrad0)/(1.0+lambda);
        Vtheta[i*nsec + j] = (Vtheta[i*nsec + j] + lambda*vtheta0)/(1.0+lambda);
        Dens[i*nsec + j] = (Dens[i*nsec + j] + lambda*dens0)/(1.0+lambda);
        if (Adiabatic)
          Energy[i*nsec + j] = (Energy[i*nsec + j] + lambda*energy0)/(1.0+lambda);
      }
    }
}

__global__ void DivisePolarGridKernel (float *res, float *num, float *denom, int nrad, int nsec)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = threadIdx.y + blockDim.y*blockIdx.y;

  if (i<nrad && j<nsec) // i <= nrad
    res[i*nsec + j] = num[i*nsec + j]/(denom[i*nsec + j] + 1e-20);
}

__global__ void VanLeerRadialKernel (float *Rinf, float *Rsup, float *QRStar, float *DensStar, float *Vrad,
  float *LostByDisk, int nsec, int nrad, float dt, int OpenInner)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = threadIdx.y + blockDim.y*blockIdx.y;

  float varq, dtheta;

  if (i<nrad && j<nsec){
    dtheta = 2.0*M_PI/float(nsec);
    varq = dt*dtheta*Rinf[i]*QRStar[i*nsec + j]* DensStar[i*nsec + j]*Vrad[i*nsec + j];
    varq -= dt*dtheta*Rsup[i]*QRStar[(i+1)*nsec + j]* DensStar[(i+1)*nsec + j]*Vrad[(i+1)*nsec + j];
    if (i==0 && OpenInner)
      LostByDisk[j] = varq;
  }
}

__global__ void VanLeerThetaKernel (float *Rsup, float *Rinf, float *Surf, float dt, int nrad, int nsec,
  int UniformTransport, int *NoSplitAdvection, float *QRStar, float *DensStar, float *Vtheta, float *Qbase)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = threadIdx.y + blockDim.y*blockIdx.y;

  float dxrad, invsurf, varq;

  if (i<nrad && j<nsec){
    if ((UniformTransport == NO) || (NoSplitAdvection[i] == NO)){
      dxrad = (Rsup[i]-Rinf[i])*dt;
      invsurf = 1.0/Surf[i];
      varq = dxrad*QRStar[i*nsec + j]*DensStar[i*nsec + j]*Vtheta[i*nsec + j];
      varq -= dxrad*QRStar[i*nsec + (j+1)%nsec]*DensStar[i*nsec + (j+1)%nsec]*Vtheta[i*nsec + (j+1)%nsec];
      Qbase[i*nsec + j] += varq*invsurf;
    }
  }
}

__global__ void ComputeAverageThetaVelocitiesKernel(float *Vtheta, float *VMed, int nsec, int nrad)
{
  int i = threadIdx.x + blockDim.x*blockIdx.x;

  float moy = 0.0;
  if (i<nrad){
    for (int j = 0; j < nsec; j++)
      moy += Vtheta[i*nsec + j];

    VMed[i] = moy/(float)nsec;
  }
}

__global__ void ComputeResidualsKernel (float *VthetaRes, float *VMed, int nsec, int nrad)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = threadIdx.y + blockDim.y*blockIdx.y;

  if (i<nrad && j<nsec)
    VthetaRes[i*nsec + j] = -VMed[i];
}

__global__ void ComputeConstantResidualKernel (float *VMed, float *invRmed, int *Nshift, int *NoSplitAdvection,
  int nsec, int nrad, float dt, float *Vtheta, float *VthetaRes, float *Rmed, int FastTransport)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = threadIdx.y + blockDim.y*blockIdx.y;

  float maxfrac, Ntilde, Nround, invdt;
  int nitemp;

  if (i<nrad && j<nsec){
    if (FastTransport) maxfrac = 1.0;
    else maxfrac = 0.0;

    invdt = 1.0/dt;
    Ntilde = VMed[i]*invRmed[i]*dt*(float)nsec/2.0/M_PI;
    Nround = floor(Ntilde+0.5);
    nitemp = (int)Nround;
    Nshift[i] = (int)nitemp;

    Vtheta[i*nsec + j] = (Ntilde-Nround)*Rmed[i]*invdt*2.0*M_PI/(float)nsec;

    if (maxfrac < 0.5){
      NoSplitAdvection[i] = YES;
      VthetaRes[i*nsec + j] += Vtheta[i*nsec + j];
      Vtheta[i*nsec + j] = 0.0;
    }
    else{
      NoSplitAdvection[i] = NO;
    }
  }
}

__global__ void StarThetaKernel (float *Qbase, float *Rmed, float *Vtheta, float *QStar, int nrad, int nsec,
  float *dq, float dt)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = threadIdx.y + blockDim.y*blockIdx.y;

  float dxtheta, ksi, invdxtheta, dqp, dqm;
  if (i<nrad && j<nsec){
    if (i<nrad){
      dxtheta = 2.0*M_PI/(float)nsec*Rmed[i];
      invdxtheta = 1.0/dxtheta;
    }
    dqm = (Qbase[i*nsec + j] - Qbase[i*nsec + ((j-1)+nsec)%nsec]);
    dqp = (Qbase[i*nsec + (j+1)%nsec] - Qbase[i*nsec + j]);

    if (dqp * dqm > 0.0)
      dq[i*nsec + j] = dqp*dqm/(dqp+dqm)*invdxtheta;
    else
      dq[i*nsec + j] = 0.0;
    __syncthreads();

    ksi = Vtheta[i*nsec + j]*dt;

    if (ksi > 0.0)
      QStar[i*nsec + j] = Qbase[i*nsec + ((j-1)+nsec)%nsec]+(dxtheta-ksi)*dq[i*nsec + ((j-1)+nsec)%nsec];
    else
      QStar[i*nsec + j] = Qbase[i*nsec + j]-(dxtheta+ksi)*dq[i*nsec + j];
   }
}

__global__ void AdvectSHIFTKernel (float *array, float *TempShift, int nsec, int nrad, int *Nshift)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = threadIdx.y + blockDim.y*blockIdx.y;

  int ji, modji;

  if (i<nrad && j<nsec){
    ji = j-Nshift[i];
    modji = ji%nsec;
    if (ji < 0) ji += nsec;

    TempShift[i*nsec + j] = array[i*nsec + modji];
    __syncthreads();

    array[i*nsec + j] = TempShift[i*nsec + j];
  }
}

__global__ void ComputeVelocitiesKernel (float *Vrad, float *Vtheta, float *Dens, float *Rmed, float *ThetaMomP,
  float *ThetaMomM, float *RadMomP, float *RadMomM, int nrad, int nsec, float OmegaFrame)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = threadIdx.y + blockDim.y*blockIdx.y;

  if (i<nrad && j<nsec){
    if (i == 0) Vrad[i*nsec + j] = 0.0;
    else {
      Vrad[i*nsec + j] = (RadMomP[(i-1)*nsec + j] + RadMomM[i*nsec + j])/(Dens[i*nsec + j] +
        Dens[(i-1)*nsec + j] + 1e-20);
      Vtheta[i*nsec + j] = (ThetaMomP[i*nsec + ((j-1)+nsec)%nsec] + ThetaMomM[i*nsec + j])/(Dens[i*nsec + j] +
        Dens[i*nsec + ((j-1)+nsec)%nsec] + 1e-15)/Rmed[i] - Rmed[i]*OmegaFrame;
      /* It was the angular momentum */
    }
  }
}

__global__ void ComputeSpeQtyKernel (float *Label, float *Dens, float *ExtLabel, int nrad, int nsec)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = threadIdx.y + blockDim.y*blockIdx.y;

  if (i<nrad && j<nsec){
    Label[i*nsec + j] = ExtLabel[i*nsec + j]/Dens[i*nsec + j];
    /* Compressive flow if line commentarized
    Label[i*nsec + j] = ExtLabel[i*nsec + j] */
  }
}

__global__ void FillForcesArraysKernel (float *Rmed, int nsec, int nrad, float xplanet, float yplanet, float smooth,
  float mplanet, int Indirect_Term, float InvPlanetDistance3, float *Potential, Pair IndirectTerm)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = threadIdx.y + blockDim.y*blockIdx.y;

  float InvDistance, angle, x, y, distance, distancesmooth, pot;

  if (i<nrad && j<nsec){
    InvDistance = 1.0/Rmed[i];
    angle = (float)j/(float)nsec*2.0*M_PI;
    x = Rmed[i]*cosf(angle);
    y = Rmed[i]*sinf(angle);
    distance = (x-xplanet)*(x-xplanet)+(y-yplanet)*(y-yplanet);
    distancesmooth = sqrtf(distance+smooth);
    pot = -G*mplanet/distancesmooth; /* Direct term from planet */
    if (Indirect_Term)
      pot += G*mplanet*InvPlanetDistance3*(x*xplanet+y*yplanet); /* Indirect term from planet */
    Potential[i*nsec + j] += pot;

    /* -- Gravitational potential from star on gas -- */
    pot = -G*1.0*InvDistance; /* Direct term from star */
    pot -= IndirectTerm.x*x + IndirectTerm.y*y; /* Indirect term from star */
    Potential[i*nsec + j] += pot;
  }
}

__global__ void CorrectVthetaKernel (float *Vtheta, float domega, float *Rmed, int nrad, int nsec)
{
    int j = threadIdx.x + blockDim.x*blockIdx.x;
    int i = threadIdx.y + blockDim.y*blockIdx.y;

    if (i<nrad && j<nsec)
      Vtheta[i*nsec + j] -= domega*Rmed[i];
}

__global__ void ConditionCFLKernel1D (float *Rsup, float *Rinf, float *Rmed, int nrad, int nsec,
  float *Vtheta, float *Vmoy)
{
  int i = threadIdx.x + blockDim.x*blockIdx.x;
  int j;
  float dxrad, dxtheta;

  if (i<nrad){
    dxrad = Rsup[i]-Rinf[i];
    dxtheta = Rmed[i]*2.0*M_PI/(float)nsec;
    Vmoy[i] = 0.0;

    for (j = 0; j < nsec; j++)
      Vmoy[i] += Vtheta[i*nsec + j];

    Vmoy[i] /= (float)nsec;
  }
}

__global__ void ConditionCFLKernel2D1 (float *Rsup, float *Rinf, float *Rmed, int nsec, int nrad,
  float *Vresidual, float *Vtheta, float *Vmoy, int FastTransport, float *SoundSpeed, float *Vrad,
  float CVNR, float *DT2D, float CFLSECURITY)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = threadIdx.y + blockDim.y*blockIdx.y;

  float dxrad, dxtheta, invdt1, invdt2, invdt3, invdt4, dvr, dvt, dt;

  if (i > 0 && i<nrad && j<nsec){
    dxrad = Rsup[i]-Rinf[i];
    dxtheta = Rmed[i]*2.0*M_PI/(float)nsec;
    if (FastTransport) Vresidual[i*nsec + j] = Vtheta[i*nsec + j]-Vmoy[i]; /* Fargo algorithm */
    else Vresidual[i*nsec + j] = Vtheta[i*nsec + j];                       /* Standard algorithm */
    //Vresidual[i*nsec + nsec] = Vresidual[i*nsec];
    invdt1 = SoundSpeed[i*nsec + j]/(min2(dxrad,dxtheta));
    invdt2 = fabs(Vrad[i*nsec + j])/dxrad;
    invdt3 = fabs(Vresidual[i*nsec + j])/dxtheta;
    dvr = Vrad[(i+1)*nsec + j]-Vrad[i*nsec + j];
    dvt = Vtheta[i*nsec + (j+1)%nsec]-Vtheta[i*nsec + j];
    if (dvr >= 0.0) dvr = 1e-10;
    else dvr = -dvr;
    if (dvr >= 0.0) dvt = 1e-10;
    else dvt = -dvt;
    invdt4 = max2(dvr/dxrad, dvt/dxtheta);
    invdt4*= 4.0*CVNR*CVNR;
    dt = CFLSECURITY/sqrtf(invdt1*invdt1+invdt2*invdt2+invdt3*invdt3+invdt4*invdt4);
    DT2D[i*nsec + j] = dt; // array nrad*nsec size dt
  }
}



__global__ void ConditionCFLKernel2D2 (float *newDT, float *DT2D, float *DT1D, float *Vmoy, float *invRmed,
  int *CFL, int nsec, int nrad, float CFLSECURITY, float DeltaT)
{
  int i = threadIdx.x + blockDim.x*blockIdx.x;
  int k;
  float dt;
  float newdt = 1e30;

  if (i>0 && i<nrad){
    newDT[i] = newdt;
    for (k = 0; k < nsec; k++)
      if (DT2D[i*nsec + k] < newDT[i]) newDT[i] = DT2D[i*nsec + k]; // for each dt in nrad

  }
  if (i<nrad-1){
    dt = 2.0*M_PI*CFLSECURITY/(float)nsec/fabs(Vmoy[i]*invRmed[i]-Vmoy[i+1]*invRmed[i+1]);
    DT1D[i] = dt; // array nrad size dt
    __syncthreads();

    if (i == 1){ // one thread
      newdt = newDT[1];
      for (k = 2; k < nrad; k++)
        if (newDT[k] < newdt) newdt = newDT[k]; // min dt

      for (k = 0; k < nrad-1; k++) {
        if (DT1D[k] < newdt)
          newdt = DT1D[k];
      }

      if (DeltaT < newdt)
        newdt = DeltaT;

      CFL[0] = (int)(ceil(DeltaT/newdt));
    }
  }
}

__device__ float max2(float a, float b)
{
  if (a > b) return a;
  return b;
}

__device__ float min2(float a, float b)
{
  if (a < b) return a;
  return b;
}

__device__ float AspectRatioDevice(float r, float ASPECTRATIO, float TRANSITIONWIDTH, float TRANSITIONRADIUS,
  float TRANSITIONRATIO, float PhysicalTime, float PhysicalTimeInitial, float LAMBDADOUBLING)
{
  float aspectratio, rmin, rmax, scale;
  aspectratio = ASPECTRATIO;
  rmin = TRANSITIONRADIUS-TRANSITIONWIDTH*ASPECTRATIO;
  rmax = TRANSITIONRADIUS+TRANSITIONWIDTH*ASPECTRATIO;
  scale = 1.0+(PhysicalTime-PhysicalTimeInitial)*LAMBDADOUBLING;
  rmin *= scale;
  rmax *= scale;
  if (r < rmin) aspectratio *= TRANSITIONRATIO;
  if ((r >= rmin) && (r <= rmax)){
    aspectratio *= expf((rmax-r)/(rmax-rmin)*logf(TRANSITIONRATIO));
  }
  return aspectratio;
}
