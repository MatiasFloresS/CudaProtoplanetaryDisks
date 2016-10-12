#include "Main.cuh"

extern int blocksize2, size_grid, nrad2pot, nsec2pot, NRAD, NSEC;
extern float *GLOBAL_bufarray, *gridfield_d, *GLOBAL_bufarray_d;
extern dim3 dimGrid, dimBlock;

__global__ void Substep1Kernel (float *Pressure, float *dens, float *VradInt, float *invdiffRmed, float *pot,
   float *Rinf, float *invRinf, float *vrad, float *vthetaint, float *vtheta, float *Rmed, float dt,
   int nrad, int nsec, float OmegaFrame, bool ZMPlus, float IMPOSEDDISKDRIFT, float SIGMASLOPE,
   float *powRmed)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = threadIdx.y + blockDim.y*blockIdx.y;
  float gradp, gradphi, vt2;
  i+= 1;
  // i=1->nrad , j=0->nsec

  if (i<nrad && j<nsec)
  {
    gradp = 2.0*(Pressure[i*nsec + j] - Pressure[(i-1)*nsec + j])/(dens[i*nsec + j] + dens[(i-1)*nsec + j])*invdiffRmed[i];
    gradphi = (pot[i*nsec + j]-pot[(i-1)*nsec + j])*invdiffRmed[i];
    vt2 = Pressure[i*nsec + j] + Pressure[(i-1)*nsec + j] + Pressure[i*nsec + (j+1)%nsec] + Pressure[(i-1)*nsec + (j+1)%nsec];
    vt2 = vt2/4.0+Rinf[i]*OmegaFrame;
    VradInt[i*nsec + j] = vrad[i*nsec + j] + dt*(-gradp - gradphi + vt2*vt2*invRinf[i]);
  }

  i-=1;

  // i=0->nrad ,   j=0->nsec
  if (i<nrad && j<nsec)
  {
    gradp = 2.0*(Pressure[i*nsec + j] - Pressure[i*nsec + ((j-1)+nsec)%nsec])/(dens[i*nsec +j] +dens[i*nsec + ((j-1)+nsec)%nsec]) \
    *1.0/(2.0*CUDART_PI_F/nsec*Rmed[i]);

    if (ZMPlus) gradp *= 1; //gradp *= SG_aniso_coeff;  Definir mas adelante SG_aniso_coeff

    gradphi = (pot[i*nsec+ j] - pot[i*nsec + ((j-1)+nsec)%nsec])*1.0/(2.0*CUDART_PI_F/nsec*Rmed[i]);
    vthetaint[i*nsec + j] = vtheta[i*nsec + j]- dt*(gradp+gradphi);
    vthetaint[i*nsec + j] += dt*IMPOSEDDISKDRIFT*0.5*powRmed[i];
  }
}

__global__ void Substep3Kernel (float *dens, float *qplus, float *viscosity_array, float *Trr, float *Trp,float *Tpp,
  float *divergence, int nrad, int nsec, float *Rmed, int Cooling, float *energynew, float dt, float *EnergyMed,
  float *SigmaMed, float *CoolingTimeMed, float *energy, float ADIABATICINDEX, float *QplusMed)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = threadIdx.y + blockDim.y*blockIdx.y;
  float den, num;

  i+= 1;

  if (i<nrad && j<nsec)
  {
    if (viscosity_array[i] != 0.0)
    {
      qplus[i*nsec + j] = 0.5/viscosity_array[i]/dens[i*nsec + j]*(Trr[i*nsec + j]*Trr[i*nsec + j] + \
        Trp[i*nsec + j]* Trp[i*nsec + j] + Tpp[i*nsec + j]*Tpp[i*nsec + j]);

      qplus[i*nsec + j] += (2.0/9.0)*viscosity_array[i]*dens[i*nsec + j]*divergence[i*nsec + j]* \
        divergence[i*nsec + j];
    }
    else qplus[i*nsec + j] = 0.0;

  }
  __syncthreads();
  i-= 1;
  if (i==0 && j<nsec)
  {
    /* We calculate the heating source term Qplus for i=0 */

    if (viscosity_array[i] != 0) {
      /* power-law extrapolation */
      qplus[i*nsec + j] = qplus[(i+1)*nsec + j]*expf(logf(qplus[(i+1)*nsec + j]/qplus[(i+2)*nsec + j]) * \
        logf(Rmed[i]/Rmed[i+1]) / logf(Rmed[i+1]/Rmed[i+2]));
    }
    else qplus[i*nsec + j] = 0.0;
  }

  __syncthreads();

  if (i<nrad && j<nsec)
  {
    if (!Cooling)
    {
      num = dt*qplus[i*nsec + j] + energy[i*nsec + j];
      den = 1.0+(ADIABATICINDEX-1.0)*dt*divergence[i*nsec + j];
      energynew[i*nsec + j] = num/den;
    }
    else
    {
      num = EnergyMed[i]*dt*dens[i*nsec + j]/SigmaMed[i] + CoolingTimeMed[i]*energy[i*nsec + j] + \
        dt*CoolingTimeMed[i]*(qplus[i*nsec + j]-QplusMed[i]*dens[i*nsec + j]/SigmaMed[i]);

      den = dt + CoolingTimeMed[i] + (ADIABATICINDEX-1.0)*dt*CoolingTimeMed[i]*divergence[i*nsec + j];
      energynew[i*nsec + j] = num/den;
    }
  }
}

__global__ void UpdateVelocitiesKernel (float *vthetaint, float *VradInt, float *invRmed, float *Rmed, float *Rsup,
  float *Rinf, float *invdiffRmed, float *invdiffRsup, float *dens, float *invRinf, float *TAURR, float *TAURP,
  float *TAUPP, float DeltaT, int nrad, int nsec)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = threadIdx.y + blockDim.y*blockIdx.y;

  i+=1;
  /* Now we can update velocities
     with the viscous source term
     of Navier-Stokes equation */

  /* vtheta first */
  if (i<nrad-1 && j<nsec)
  {
    vthetaint[i*nsec +j] += DeltaT*invRmed[i]*((Rsup[i]*TAURP[(i+1)*nsec+ j]-Rinf[i]*TAURP[i*nsec +j])*invdiffRsup[i] + \
    (TAUPP[i*nsec +j]-TAUPP[i*nsec + ((j-1)+nsec)%nsec])*1.0/(2.0*CUDART_PI_F/nsec) + 0.5*(TAURP[i*nsec + j] + TAURP[(i+1)*nsec +j]))/ \
    (0.5*(dens[i*nsec +j]+dens[i*nsec + ((j-1)+nsec)%nsec]));
  }

  /* now vrad */
  if (i<nrad && j<nsec)
  {
    VradInt[i*nsec +j] += DeltaT*invRinf[i]*((Rmed[i]*TAURR[i*nsec +j]- Rmed[i-1]*TAURR[(i-1)*nsec + j])*invdiffRmed[i] + \
    (TAURP[i*nsec + (j+1)%nsec] -TAURP[i*nsec + j])*1.0/(2.0*CUDART_PI_F/nsec) - 0.5*(TAURP[i*nsec +j] + TAURP[(i-1)*nsec + j]))/ \
    (0.5*(dens[i*nsec +j] + dens[(i-1)*nsec + j]));

  }
}

__global__ void InitComputeAccelKernel (float *CellAbscissa, float *CellOrdinate, float *Rmed, int nsec, int nrad)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = threadIdx.y + blockDim.y*blockIdx.y;

  if (i<nrad && j<nsec)
  {
    CellAbscissa[i*nsec+j] = Rmed[i] * cosf((2.0*M_PI*j)/(float)nsec);
    CellOrdinate[i*nsec+j] = Rmed[i] * sinf((2.0*M_PI*j)/(float)nsec);
  }
}

__global__ void ComputeSoundSpeedKernel (float *SoundSpeed, float *dens, float *Rmed, float *energy, int nsec, int nrad,
  int Adiabaticc, float ADIABATICINDEX, float FLARINGINDEX, float *AspectRatioRmed)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = threadIdx.y + blockDim.y*blockIdx.y;

  if (i<nrad && j<nsec)
  {
    if (!Adiabaticc) SoundSpeed[j+i*nsec] = AspectRatioRmed[i]*sqrtf(1.0*1.0/Rmed[i])*powf(Rmed[i], FLARINGINDEX);
    else SoundSpeed[j+i*nsec] = sqrtf(ADIABATICINDEX*(ADIABATICINDEX-1.0)*energy[j+i*nsec]/dens[j+i*nsec]);
  }
}

__global__ void ComputePressureFieldKernel (float *SoundSpeed, float *dens, float *Pressure, int Adiabaticc, int nrad,
  int nsec, int ADIABATICINDEX, float *energy)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = threadIdx.y + blockDim.y*blockIdx.y;

  if (i<nrad && j<nsec)
  {
    if (~Adiabaticc) Pressure[j+i*nsec] = dens[j+i*nsec]*SoundSpeed[j+i*nsec]*SoundSpeed[j+i*nsec];
    else Pressure[j+i*nsec] = (ADIABATICINDEX-1.0)*energy[j+i*nsec];
  }
}

__global__ void ComputeTemperatureFieldKernel (float *dens, float *Temperature, float *Pressure, float *energy, float MU,
  float R, float ADIABATICINDEX, int Adiabaticc, int nsec, int nrad)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = threadIdx.y + blockDim.y*blockIdx.y;

  if (i<nrad && j<nsec)
  {
    if (~Adiabaticc) Temperature[j+i*nsec] = MU/R*Pressure[j+i*nsec]/dens[j+i*nsec];
    else Temperature[j+i*nsec] = MU/R*(ADIABATICINDEX-1.0)*energy[j+i*nsec]/dens[j+i*nsec];
  }
}

__global__ void InitLabelKernel (float *label, float xp, float yp, float rhill, float *Rmed, int nrad, int nsec)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = threadIdx.y + blockDim.y*blockIdx.y;

  if (i<nrad && j<nsec)
  {
    float distance, angle, x, y;
    angle = 2.0*CUDART_PI_F*j/nsec;
    x = Rmed[i] * cosf(angle);
    y = Rmed[i] * sinf(angle);
    distance = sqrtf( (x - xp) * (x - xp) + (y - yp)*(y -yp) );
    if (distance < rhill) label[i] = 1.0;
    else label[i] = 0.0;
  }
}

__global__ void CircumPlanetaryMassKernel (float *dens, float *Surf, float *CellAbscissa, float *CellOrdinate,
  float xpl, float ypl, int nrad, int nsec, float HillRadius, float *mdcp0)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = threadIdx.y + blockDim.y*blockIdx.y;

  float dist;

  if (i<nrad && j<nsec)
  {
    dist = sqrtf((CellAbscissa[j+i*nsec]-xpl)*(CellAbscissa[j+i*nsec]-xpl) + (CellOrdinate[j+i*nsec]-ypl)*(CellOrdinate[j+i*nsec]-ypl));
    if (dist < HillRadius) mdcp0[j+i*nsec] =  Surf[i]* dens[j+i*nsec];
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
    while (i < n)
    {
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
    if ((blockSize >= 512) && (tid < 256))
    {
        sdata[tid] = mySum = mySum + sdata[tid + 256];
    }

    __syncthreads();

    if ((blockSize >= 256) &&(tid < 128))
    {
            sdata[tid] = mySum = mySum + sdata[tid + 128];
    }

     __syncthreads();

    if ((blockSize >= 128) && (tid <  64))
    {
       sdata[tid] = mySum = mySum + sdata[tid +  64];
    }

    __syncthreads();

    #if (__CUDA_ARCH__ >= 300 )
        if ( tid < 32 )
        {
            // Fetch final intermediate sum from 2nd warp
            if (blockSize >=  64) mySum += sdata[tid + 32];
            // Reduce final warp using shuffle
            for (int offset = warpSize/2; offset > 0; offset /= 2)
            {
                mySum += __shfl_down(mySum, offset);
            }
        }
    #else
        // fully unroll reduction within a single warp
        if ((blockSize >=  64) && (tid < 32))
        {
            sdata[tid] = mySum = mySum + sdata[tid + 32];
        }

        __syncthreads();

        if ((blockSize >=  32) && (tid < 16))
        {
            sdata[tid] = mySum = mySum + sdata[tid + 16];
        }

        __syncthreads();

        if ((blockSize >=  16) && (tid <  8))
        {
            sdata[tid] = mySum = mySum + sdata[tid +  8];
        }

        __syncthreads();

        if ((blockSize >=   8) && (tid <  4))
        {
            sdata[tid] = mySum = mySum + sdata[tid +  4];
        }

        __syncthreads();

        if ((blockSize >=   4) && (tid <  2))
        {
            sdata[tid] = mySum = mySum + sdata[tid +  2];
        }

        __syncthreads();

        if ((blockSize >=   2) && ( tid <  1))
        {
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
  while (x < n)
  {
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
  }else{
    DeviceReduceKernel<false><<<blocks, threads, smemSize>>>(in, device_out, N);
    gpuErrchk(cudaDeviceSynchronize());
  }

  float *h_odata = (float *) malloc(blocks*sizeof(float));
  float sum = 0.0;

  gpuErrchk(cudaMemcpy(h_odata, device_out, blocks * sizeof(float),cudaMemcpyDeviceToHost));
  for (int i=0; i<blocks; i++)
  {
    sum += h_odata[i];
  }
  cudaFree(device_out);
  free(h_odata);
	return sum;
}

__global__ void MultiplyPolarGridbyConstantKernel (float *dens, int nrad, int nsec, float ScalingFactor)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = threadIdx.y + blockDim.y*blockIdx.y;

  if (i<nrad && j<nsec) dens[j+i*nsec] *= ScalingFactor;
}

__global__ void Substep2Kernel (float *dens, float *VradInt, float *vthetaint, float *TemperInt, int nrad,
  int nsec, float CVNR, float *invdiffRmed, float *invdiffRsup, float *densint, int Adiabaticc, float *Rmed,
  float dt, float *vradnew, float *vthetanew, float *energy, float *energyint)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = threadIdx.y + blockDim.y*blockIdx.y;

  float dv;
  if (i<nrad && j<nsec)
  {
    dv = VradInt[(i+1)*nsec + j] - VradInt[i*nsec + j];

    if (dv < 0.0) densint[i*nsec + j] = CVNR*CVNR*dens[i*nsec + j]*dv*dv;
    else densint[i*nsec + j] = 0.0;

    dv = vthetaint[i*nsec + (j+1)%nsec] - vthetaint[i*nsec + j];
    if (dv < 0.0) TemperInt[i*nsec + j] = CVNR*CVNR*dens[i*nsec + j]*dv*dv;
    else TemperInt[i*nsec + j] = 0.0;

  }
  i+=1;

  if (i<nrad && j<nsec)
  {
    vradnew[i*nsec + j] = VradInt[i*nsec + j] - dt*2.0/(dens[i*nsec + j] + dens[(i-1)*nsec + j])*(densint[i*nsec + j] - densint[(i-1)*nsec + j]) *\
      invdiffRmed[i];
  }
  i-=1;

  if (i<nrad && j<nsec)
  {
    vthetanew[i*nsec + j] = vthetaint[i*nsec + j] - dt*2.0/(dens[i*nsec + j] + dens[i*nsec + ((j-1)+nsec)%nsec])*(densint[i*nsec + j] - \
      densint[(i-1)*nsec + j])* 1.0/(2.0*CUDART_PI_F*Rmed[i]/nsec);
  }
  if (Adiabaticc)
  {
    i+=1;

    if (i<nrad && j<nsec)
    {
      energyint[i*nsec + j] = energy[i*nsec + j] - dt*densint[i*nsec + j]*(VradInt[i*nsec + j+1] - VradInt[i*nsec + j])*invdiffRsup[i] - \
      dt*TemperInt[i*nsec + j]*(vthetaint[i*nsec + (j+1)%nsec] - vthetaint[i*nsec + j])* 1.0/(2.0*CUDART_PI_F*Rmed[i]/nsec);
    }
  }
}

__global__ void OpenBoundaryKernel (float *vrad, float *dens, float *energy, int nsec, float *SigmaMed)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i=1;

  if(j<nsec)
  {
    dens[(i-1)*nsec + j] = dens[i*nsec + j]; // copy first ring into ghost ring
    energy[(i-1)*nsec + j] = energy[i*nsec + j];
    if(vrad[(i+1)*nsec + j] > 0.0 || (dens[i*nsec + j] < SigmaMed[0])) vrad[i*nsec + j] = 0.0; // we just allow outflow [inwards]
    else vrad[i*nsec +j] = vrad[(i+1)*nsec + j];
  }
}

__global__ void ReduceCsKernel (float *SoundSpeed, float *cs0, float *cs1, float *csnrm1, float *csnrm2, int nsec, int nrad)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i=0;

  if(j<nsec)
  {
    cs0[j] = SoundSpeed[i*nsec +j];
    cs1[j] = SoundSpeed[(i+1)*nsec +j];
  }
  i = nrad-1;
  if(j<nsec)
  {
    csnrm2[j] = SoundSpeed[(i-1)*nsec +j];
    csnrm1[j] = SoundSpeed[i*nsec +j];
  }
}

__global__ void ReduceMeanKernel (float *dens, float *energy, int nsec, float *mean_dens, float *mean_energy,
  float *mean_dens2, float *mean_energy2, int nrad)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = 0;

  if(j<nsec)
  {
    mean_dens[j] = dens[i*nsec+ j];
    mean_energy[j] = energy[i*nsec +j];
  }
  i = nrad-1;
  if(j<nsec)
  {
    mean_dens2[j] = dens[i*nsec + j];
    mean_energy2[j] = energy[i*nsec + j];
  }
}

__global__ void NonReflectingBoundaryKernel (float *dens, float *energy, int i_angle, int nsec, float *vrad, float *SoundSpeed,
  float SigmaMed, int nrad, float SigmaMed2, int i_angle2)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = 1;
  float vrad_med;

  if (j<nsec)
  {

    if(j+i_angle >= nsec)
    {
      dens[j+i_angle - nsec] = dens[i*nsec + j];
      energy[j+i_angle - nsec] = energy[i*nsec + j];
    }

    else if(j+i_angle < 0)
    {
      dens[j+i_angle + nsec] = dens[i*nsec + j];
      energy[j+i_angle + nsec] = energy[i*nsec + j];
    }
    else
    {
      dens[j+i_angle] = dens[i*nsec + j];
      energy[j+i_angle] = energy[i*nsec + j];
    }

    __syncthreads();

    vrad_med = -SoundSpeed[i*nsec + j]*(dens[i*nsec + j]-SigmaMed)/SigmaMed;
    vrad[i*nsec + j] = 2.*vrad_med-vrad[(i+1)*nsec + j];
  }

  i = nrad-1;
  if (j<nsec)
  {
    if (j-i_angle2 >= nsec)
    {
      dens[i*nsec + j] = dens[j-i_angle2 + (i-2)*nsec ];
      energy[i*nsec + j] = energy[j-i_angle2 + (i-2)*nsec ];
    }
    else if (j-i_angle2 < 0)
    {
      dens[i*nsec + j] = dens[j-i_angle2 + i*nsec];
      energy[i*nsec + j] = energy[j-i_angle2 + i*nsec];
    }
    else
    {
      dens[i*nsec + j] = dens[j-i_angle2 + (i-1)*nsec];
      energy[i*nsec + j] = energy[j-i_angle2 + (i-1)*nsec];
    }

    __syncthreads();

    vrad_med = SoundSpeed[i*nsec + j]*(dens[(i-1)*nsec + j]-SigmaMed2)/SigmaMed2;
    vrad[i*nsec + j] = 2.*vrad_med - vrad[(i-1)*nsec + j];
  }
}

__global__ void MinusMeanKernel (float *dens, float *energy, float SigmaMed, float mean_dens_r, float mean_dens_r2,
  float mean_energy_r,float mean_energy_r2, float EnergyMed, int nsec, int nrad, float SigmaMed2, float EnergyMed2)
{
    int j = threadIdx.x + blockDim.x*blockIdx.x;
    int i = 0;

    if (j< nsec)
    {
      dens[i*nsec + j] += SigmaMed - mean_dens_r;
      energy[i*nsec + j] += EnergyMed - mean_energy_r;
    }

    i = nrad-1;

    if (j < nsec)
    {
      dens[i*nsec + j] += SigmaMed2 - mean_dens_r2;
      energy[i*nsec + j] += EnergyMed2 - mean_energy_r2;
    }
  }

__global__ void Make1DprofileKernel (float *gridfield, float *GLOBAL_bufarray, int nsec, int nrad)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;

  if (j < nrad)
  {
    float sum = 0.0;

    for (int t = 0; t < nsec; t++)
      sum += gridfield[j*nsec + t];

    GLOBAL_bufarray[j] = sum/nsec; // every thread updates one EvanescentBoundary
  }
}

__host__ void Make1Dprofile (float *gridfield)
{

  gpuErrchk(cudaMemcpy(gridfield_d, gridfield, size_grid*sizeof(float), cudaMemcpyHostToDevice));
  Make1DprofileKernel<<<dimGrid, dimBlock>>>(gridfield_d, GLOBAL_bufarray_d, NSEC, NRAD);
  gpuErrchk(cudaDeviceSynchronize());
  gpuErrchk(cudaMemcpy(GLOBAL_bufarray, GLOBAL_bufarray_d, NRAD*sizeof(float), cudaMemcpyDeviceToHost));

}

__global__ void InitGasVelocitiesKernel (float *viscosity_array, int nsec, int nrad, int SelfGravity, float *Rmed, float G,
  float ASPECTRATIO, float FLARINGINDEX, float SIGMASLOPE, int CentrifugalBalance, float *vrad, float *vtheta,
  float ViscosityAlpha, float IMPOSEDDISKDRIFT, float SIGMA0, float *SigmaInf, float OmegaFrame, float *Rinf, float *vt_cent)
{
    int j = threadIdx.x + blockDim.x*blockIdx.x;
    int i = threadIdx.y + blockDim.y*blockIdx.y;

    float omega, r, ri;

    if (i < nrad && j < nsec)
    {
      if (i == nrad)
      {
        r = Rmed[nrad - 1];
        ri = Rinf[nrad - 1];
      }
      else
      {
        r = Rmed[i];
        ri = Rinf[i];
      }

      if (!SelfGravity)
      {
        omega = sqrtf(G*1.0/r/r/r);
        vtheta[i*nsec + j] = omega*r*sqrtf(1.0-powf(ASPECTRATIO,2.0)* powf(r,2.0*FLARINGINDEX)* \
        (1.+SIGMASLOPE-2.0*FLARINGINDEX));
      }
      vtheta[i*nsec + j] -= OmegaFrame*r;
      if (CentrifugalBalance) vtheta[i*nsec + j] = vt_cent[i];

      if (i == nrad) vrad[i*nsec + j] = 0.0;
      else vrad[i*nsec + j] = IMPOSEDDISKDRIFT*SIGMA0/SigmaInf[i]/ri;

      if (ViscosityAlpha) vrad[i*nsec + j] -= 3.0*viscosity_array[i]/r*(-SIGMASLOPE+2.0*FLARINGINDEX+1.0);
      else vrad[i*nsec + j] -= 3.0*viscosity_array[i]/r*(-SIGMASLOPE+.5);

    }
  }

__global__ void ComputeForceKernel (float *CellAbscissa, float *CellOrdinate, float *Surf, float *dens, float x,
  float y, float rsmoothing, float *forcesxi, float *forcesyi, float *forcesxo, float *forcesyo, int nsec,
  int nrad, float G, float a, float *Rmed, int dimfxy, float rh)
{

    int j = threadIdx.x + blockDim.x*blockIdx.x;
    int i = threadIdx.y + blockDim.y*blockIdx.y;
    float cellmass, dx, dy, d2, InvDist3, dist2, distance, resultxi, resultyi;
    float resultxo, resultyo, hillcutfactor, hill_cut;

    if (i<nrad && j<nsec)
    {
      cellmass = Surf[i] * dens[i*nsec + j];
      dx = CellAbscissa[i*nsec + j] - x;
      dy = CellOrdinate[i*nsec + j] - y;
      d2 = dx*dx + dy*dy;
      dist2 = d2 + rsmoothing*rsmoothing;
      distance = sqrtf(dist2);
      InvDist3 = 1.0/dist2/distance;

      for (int k = 0; k < dimfxy; k++)
      {
        hillcutfactor =  k / (float)(dimfxy-1);
        if (k != 0)
        {
          rh *= hillcutfactor;
          hill_cut = 1.-expf(-d2/(rh*rh));
        }
        else hill_cut = 1.;


        if (Rmed[i] < a)
        {
          resultxi = G * cellmass * dx * InvDist3 * hill_cut;
          resultyi = G * cellmass * dy * InvDist3 * hill_cut;
          atomicAdd(&(forcesxi[k]), resultxi);
          atomicAdd(&(forcesyi[k]), resultyi);
        }
        else
        {
          resultxo = G * cellmass * dx * InvDist3 * hill_cut;
          resultyo = G * cellmass * dy * InvDist3 * hill_cut;
          atomicAdd(&(forcesxo[k]), resultxo);
          atomicAdd(&(forcesyo[k]), resultyo);
        }
      }
    }
  }

__global__ void ViscousTermsKernel (float *vradial, float *vazimutal , float *DRR, float *DPP, float *DivergenceVelocity,
  float *DRP, float *invdiffRsup, int invdphi, float *invRmed, float *Rsup, float *Rinf, float *invdiffRmed, int nrad,
  int nsec, float *TAURR, float *TAUPP, float *dens, float *viscosity_array, float onethird, float *TAURP, float *invRinf)
{
   int j = threadIdx.x + blockDim.x*blockIdx.x;
   int i = threadIdx.y + blockDim.y*blockIdx.y;

   if (i<nrad && j<nsec) /* Drr, Dpp and divV computation */
   {
     DRR[i*nsec + j] = (vradial[(i+1)*nsec + j] - vradial[i*nsec + j])*invdiffRsup[i];
     DPP[i*nsec + j] = (vazimutal[i*nsec + (j+1)%nsec] - vazimutal[i*nsec + j])*invdphi*invRmed[i]+0.5* \
       (vradial[(i+1)*nsec + j]+vradial[i*nsec + j])*invRmed[i];
     DivergenceVelocity[i*nsec + j] = (vradial[i*nsec + (j+1)]*Rsup[i]-vradial[i*nsec + j]*Rinf[i])*invdiffRsup[i] * \
       invRmed[i];
     DivergenceVelocity[i*nsec + j] += (vazimutal[i*nsec + (j+1)%nsec]-vazimutal[i*nsec + j])*invdphi*invRmed[i];

     if (i > 0) DRP[i*nsec + j] = 0.5*(Rinf[i]*(vazimutal[i*nsec + j]*invRmed[i]-vazimutal[(i-1)*nsec + j])*invRmed[i-1])* \
          invdiffRmed[i] + (vradial[i*nsec + j]-vradial[i*nsec + (j-1)%nsec])*invdphi*invRinf[i];
   }

   if (i<nrad && j<nsec) /* TAUrr and TAUpp computation */
   {
     TAURR[i*nsec + j] = 2.0*dens[i*nsec + j]*viscosity_array[i]*(DRR[i*nsec + j]-onethird*DivergenceVelocity[i*nsec + j]);
     TAUPP[i*nsec + j] = 2.0*dens[i*nsec + j]*viscosity_array[i]*(DPP[i*nsec + j]-onethird*DivergenceVelocity[i*nsec + j]);

     if (i > 0) TAURP[i*nsec + j] = 2.0*0.25*(dens[i*nsec + j] + dens[(i-1)*nsec + j] + dens[(i-1)*nsec + ((j-1)+nsec)%nsec])* \
         viscosity_array[i]*DRP[i*nsec + j];
   }
 }

__global__ void LRMomentaKernel (float *RadMomP, float *RadMomM, float *ThetaMomP, float *ThetaMomM, float *dens,
  float *vrad, float *vtheta, int nrad, int nsec, float *Rmed, float OmegaFrame)
{
   int j = threadIdx.x + blockDim.x*blockIdx.x;
   int i = threadIdx.y + blockDim.y*blockIdx.y;

   if (i<nrad && j<nsec)
   {
     RadMomP[i*nsec + j] = dens[i*nsec + j] * vrad[(i)*nsec + j]; // (i+1)*nsec
     RadMomM[i*nsec + j] = dens[i*nsec + j] * vrad[i*nsec + j];
     /* it is the angular momentum -> ThetaMomP */
     ThetaMomP[i*nsec + j] = dens[i*nsec + j] * (vtheta[i*nsec + (j+1)%nsec]+Rmed[i]*OmegaFrame)*Rmed[i];
     ThetaMomM[i*nsec + j] = dens[i*nsec + j] * (vtheta[i*nsec + j]+Rmed[i]*OmegaFrame)*Rmed[i];
   }
 }

__global__ void ExtQtyKernel (float *Extlabel, float *dens, float *label, int nsec, int nrad)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = threadIdx.y + blockDim.y*blockIdx.y;

  if (i<nrad && j<nsec)
  {
    Extlabel[i*nsec + j] = dens[i*nsec + j]*label[i*nsec + j];
  }

}

__global__ void StarRadKernel (float *Qbase, float *vrad, float *QStar, float dt, int nrad, int nsec,
  float *invdiffRmed, float *Rmed, float *dq)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = threadIdx.y + blockDim.y*blockIdx.y;

  float dqm, dqp;

  if (i<nrad && j<nsec)
  {
    if (i == 0 || i == nrad-1) dq[j*nrad + i] = 0.0;
    else
    {
      dqm = (Qbase[i*nsec + j] - Qbase[(i-1)*nsec + j])*invdiffRmed[i];
      dqp = (Qbase[(i+1)*nsec + j] - Qbase[i*nsec + j])*invdiffRmed[i+1];

      if (dqp * dqm > 0.0) dq[j*nrad + i] = 2.0*dqp*dqm/(dqp+dqm);
      else dq[j*nrad + i] = 0.0;
    }
    __syncthreads();

    if (vrad[i*nsec + j] > 0.0) // aca asumen vrad = 0 para i=0?
      QStar[i*nsec + j] = Qbase[(i-1)*nsec + j] + (Rmed[i]-Rmed[i-1]-vrad[i*nsec + j]*dt)*0.5*dq[j*nrad + i-1];
    else
      QStar[i*nsec + j] = Qbase[i*nsec + j]-(Rmed[i+1]-Rmed[i]+vrad[i*nsec + j]*dt)*0.5*dq[j*nrad + i];
  }
  if (j<nsec)  QStar[j] = QStar[j+nsec*nrad] = 0.0;
}

__global__ void FftKernel (float *Radii, cufftComplex *SGP_Kr, cufftComplex *SGP_Kt, float SGP_eps, int nrad, int nsec,
cufftComplex *SGP_Sr, cufftComplex *SGP_St, float *dens, float *Rmed, float *Kr_aux, float *Kt_aux)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = threadIdx.y + blockDim.y*blockIdx.y;
  float u, cosj, sinj, coshu, expu, den_SGP_K, theta, base;
  float a, var, radii;

  if (i<2*nrad && j<nsec)
  {
    SGP_Kr[i*nsec + j].x = Kr_aux[i*nsec + j];
    SGP_Kr[i*nsec + j].y = 0.;

    SGP_Kt[i*nsec + j].x = Kt_aux[i*nsec + j];
    SGP_Kt[i*nsec + j].y = 0.;

    SGP_Sr[i*nsec + j].y = 0.;
    SGP_St[i*nsec + j].y = 0.;

    if (i<nrad)
    {
      var = dens[i*nsec + j] * sqrtf(Rmed[i]/Rmed[0]);
      SGP_Sr[i*nsec + j].x = var;
      SGP_St[i*nsec + j].x = var*(Rmed[i]/Rmed[0]);
    }
    else
    {
      SGP_Sr[i*nsec + j].x = 0.;
      SGP_St[i*nsec + j].x = 0.;
    }
  }
}

__global__ void FftmulKernel (cufftComplex *Gr, cufftComplex *Gphi, cufftComplex *SGP_Kr, cufftComplex *SGP_Kt,
  cufftComplex *SGP_Sr, cufftComplex *SGP_St, int nsec, float G, int nrad)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = threadIdx.y + blockDim.y*blockIdx.y;

  if (i<2*nrad && j<nsec)
  {
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

__global__ void Sg_AccKernel (float *SG_Accr, float *SG_Acct, float *dens , float SGP_rstep, float SGP_tstep,
  float SGP_eps, int nrad, int nsec, float *Rmed, cufftComplex *Gr, cufftComplex *Gphi, float G)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = threadIdx.y + blockDim.y*blockIdx.y;

  float normaccr, normacct, divRmed;

  if (i<nrad && j<nsec)
  {
    divRmed = Rmed[i]/Rmed[0];
    normaccr = SGP_rstep*SGP_tstep / ((float)(2*nrad) * (float) nsec);
    normacct = normaccr;
    normaccr /= sqrtf(divRmed);
    normacct /= (divRmed * sqrtf(divRmed));
    SG_Acct[i*nsec + j] = Gphi[i*nsec + j].x * normaccr;

    SG_Accr[i*nsec + j] = Gr[i*nsec + j].x * normaccr;
    SG_Accr[i*nsec + j] += G*dens[i*nsec + j]*SGP_rstep*SGP_tstep / SGP_eps;
  }
}

__global__ void Update_sgvelocityKernel (float *vrad, float *vtheta, float *SG_Accr, float *SG_Acct, float *Rinf, float *Rmed,
  float *invdiffRmed, float dt, int nrad, int nsec)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = threadIdx.y + blockDim.y*blockIdx.y;

  int jm1, lm1;

  /* Here we update velocity fields to take into acount self-gravity */
  if (i<nrad && j<nsec)
  {
    /* We compute VRAD - half-centered in azimuth - from centered-in-cell radial sg acceleration. */
    if (i > 0) vrad[i*nsec + j] =  dt*((Rinf[i] - Rmed[i-1]) * SG_Accr[i*nsec + j] + \
    (Rmed[i] - Rinf[i]) * SG_Accr[(i-1)*nsec + j]) *invdiffRmed[i]; // caso !SGZeroMode

    /* We compute VTHETA - half-centered in radius - from centered-in-cell azimutal sg acceleration. */
    if (j==0) jm1 = nsec-1;
    else jm1 = j-1;
    lm1 = i*nsec + jm1;
    vtheta[i*nsec + j] = 0.5 * dt * (SG_Acct[i*nsec + j] + SG_Acct[lm1]);
  }
}

__global__ void Azimutalvelocity_withSGKernel (float *vtheta, float *Rmed, float FLARINGINDEX, float SIGMASLOPE,
  float ASPECTRATIO, float G, float *GLOBAL_bufarray, int nrad, int nsec)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = threadIdx.y + blockDim.y*blockIdx.y;

  float omegakep, omega, invr;
  if (i<nrad && j<nsec)
  {
    invr = 1./Rmed[i];
    omegakep = sqrtf(G*1.0*invr*invr*invr);
    omega = sqrtf(omegakep*omegakep* (1.0 - (1.+SIGMASLOPE-2.0*FLARINGINDEX)*powf(ASPECTRATIO,2.0)* \
      powf(Rmed[i],2.0*FLARINGINDEX)) - invr*GLOBAL_bufarray[i]);

    vtheta[i*nsec + j] = Rmed[i]*omega;
  }
}

__global__ void CrashKernel (float *array, int nrad, int nsec, bool Crash)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = threadIdx.y + blockDim.y*blockIdx.y;

  if (i<nrad && j<nsec)
  {
    if (array[i*nsec + j] < 0.0)
    {
      array[i*nsec + j] = 1.0;
    }
    else
      array[i*nsec + j] = 0.0;
  }
}

__global__ void EvanescentBoundaryKernel(float *Rmed, float *vrad, float *vtheta, float *energy, float *dens, float *AspectRatioRmed, float *viscosity_array,
  float DRMIN, float DRMAX, int nrad, int nsec, float Tin, float Tout, float step, float G, float SIGMASLOPE, float FLARINGINDEX, float *GLOBAL_bufarray,
  float OmegaFrame1, float *SigmaMed, float *EnergyMed, int Adiabaticc, int SelfGravity)
{
    int j = threadIdx.x + blockDim.x*blockIdx.x;
    int i = threadIdx.y + blockDim.y*blockIdx.y;

    float damping, lambda, vtheta0, vrad0, energy0, dens0;
    if (i<nrad && j<nsec)
    {
      if ((Rmed[i] < DRMIN) || (Rmed[i] > DRMAX))
      {  /* Damping operates only inside the wave killing zones */
        if(Rmed[i] < DRMIN)
        {
          damping = (Rmed[i]-DRMIN)/(Rmed[0]-DRMIN);
          lambda = damping*damping*10.0*step/Tin;
        }
        if (Rmed[i] > DRMAX)
        {
          damping = (Rmed[i]-DRMAX)/(Rmed[nrad-1]-DRMAX);
          lambda = damping*damping*10.0*step/Tout;
        }
        if(!SelfGravity)
        {
          vtheta0 = sqrtf(G*1.0/Rmed[i] * (1.0 - (1.0+SIGMASLOPE-2.0*FLARINGINDEX)*powf(AspectRatioRmed[i],2.0) * \
          powf(Rmed[i],2.0*FLARINGINDEX)));
        }
        if (SelfGravity)
        {
          vtheta0 = sqrtf(G*1.0/Rmed[i] * (1.0 - (1.0+SIGMASLOPE-2.0*FLARINGINDEX)*powf(AspectRatioRmed[i],2.0) * \
          powf(Rmed[i],2.0*FLARINGINDEX)) - Rmed[i]*GLOBAL_bufarray[i]);
        }
        /* this could be refined if CentrifugalBalance is used... */
        vtheta0 -= Rmed[i]*OmegaFrame1;
        vrad0 = -3.0*viscosity_array[i]/Rmed[i]*(-SIGMASLOPE+.5);
        dens0 = SigmaMed[i];
        energy0 = EnergyMed[i];

        vrad[i*nsec + j] = (vrad[i*nsec + j] + lambda*vrad0)/(1.0+lambda);
        vtheta[i*nsec + j] = (vtheta[i*nsec + j] + lambda*vtheta0)/(1.0+lambda);
        dens[i*nsec + j] = (dens[i*nsec + j] + lambda*dens0)/(1.0+lambda);
        if (Adiabaticc)
          energy[i*nsec + j] = (energy[i*nsec + j] + lambda*energy0)/(1.0+lambda);
      }
    }
}

__global__ void DivisePolarGridKernel (float *res, float *num, float *denom, int nrad, int nsec)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = threadIdx.y + blockDim.y*blockIdx.y;

  if (i<nrad && j<nsec) // i <= nrad
  {
    res[i*nsec + j] = num[i*nsec + j]/(denom[i*nsec + j] + 1e-20);
  }
}
