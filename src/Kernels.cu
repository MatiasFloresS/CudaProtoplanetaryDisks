#include "Kernels.cuh"
#include <stdio.h>
using namespace std;

extern int blocksize2, size_grid, nrad2pot, nsec2pot, NRAD, NSEC;
extern float *GLOBAL_bufarray;

__global__ void substep1(float *press, float *dens, float *vradint, float *invdiffRmed, float *pot,
   float *Rinf, float *invRinf, float *vrad, float *vthetaint, float *vtheta, float *Rmed, float dt,
 int nrad, int nsec, float OmegaFrame, bool ZMPlus, float IMPOSEDDISKDRIFT, float SIGMASLOPE, float *powRmed)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = threadIdx.y + blockDim.y*blockIdx.y;
  float gradp, gradphi, vt2;
  i+= 1;
  // i=1->nrad , j=0->nsec

  if (i<nrad && j<nsec)
  {
    gradp = 2.0*(press[i*nsec + j] - press[(i-1)*nsec + j])/(dens[i*nsec + j] + dens[(i-1)*nsec + j])*invdiffRmed[i];
    gradphi = (pot[i*nsec + j]-pot[(i-1)*nsec + j])*invdiffRmed[i];
    vt2 = press[i*nsec + j] + press[(i-1)*nsec + j] + press[i*nsec + (j+1)%nsec] + press[(i-1)*nsec + (j+1)%nsec];
    vt2 = vt2/4.0+Rinf[i]*OmegaFrame;
    vradint[i*nsec + j] = vrad[i*nsec + j] + dt*(-gradp - gradphi + vt2*vt2*invRinf[i]);
  }

  i-=1;

  // i=0->nrad ,   j=0->nsec
  if (i<nrad && j<nsec)
  {
    gradp = 2.0*(press[i*nsec + j] - press[i*nsec + ((j-1)+nsec)%nsec])/(dens[i*nsec +j] +dens[i*nsec + ((j-1)+nsec)%nsec]) \
    *1.0/(2.0*CUDART_PI_F/nsec*Rmed[i]);

    if (ZMPlus) gradp *= 1; //gradp *= SG_aniso_coeff;  Definir mas adelante SG_aniso_coeff

    gradphi = (pot[i*nsec+ j] - pot[i*nsec + ((j-1)+nsec)%nsec])*1.0/(2.0*CUDART_PI_F/nsec*Rmed[i]);
    vthetaint[i*nsec + j] = vtheta[i*nsec + j]- dt*(gradp+gradphi);
    vthetaint[i*nsec + j] += dt*IMPOSEDDISKDRIFT*0.5*powRmed[i];
  }
}

__global__ void substep3(float *dens, float *qplus, float *viscosity_array, float *Trr, float *Trp,float *Tpp,
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

  if (i<nrad && j<nsec)
  {
    if (!Cooling)
    {
      num = dt*qplus[i*nsec + j] + energy[i*nsec + j];
      den = 1.0+(ADIABATICINDEX-1.0)*dt*divergence[i*nsec + j];
      energynew[i*nsec + j] = num/den;
    }
    if (Cooling)
    {
      num = EnergyMed[i]*dt*dens[i*nsec + j]/SigmaMed[i] + CoolingTimeMed[i]*energy[i*nsec + j] + \
        dt*CoolingTimeMed[i]*(qplus[i*nsec + j]-QplusMed[i]*dens[i*nsec + j]/SigmaMed[i]);

      den = dt + CoolingTimeMed[i] + (ADIABATICINDEX-1.0)*dt*CoolingTimeMed[i]*divergence[i*nsec + j];
      energynew[i*nsec + j] = num/den;
    }
  }

}


__global__ void UpdateVelocities(float *vtheta, float *vrad, float *invRmed, float *Rmed, float *Rsup,
  float *Rinf, float *invdiffRmed, float *invdiffRsup, float *dens, float *invRinf, float *Trr,
  float *Trp, float *Tpp, float DeltaT, int nrad, int nsec)
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
    vtheta[i*nsec +j] += DeltaT*invRmed[i]*((Rsup[i]*Trp[(i+1)*nsec+ j]-Rinf[i]*Trp[i*nsec +j])*invdiffRsup[i] + \
    (Tpp[i*nsec +j]-Tpp[i*nsec + ((j-1)+nsec)%nsec])*1.0/(2.0*CUDART_PI_F/nsec) + 0.5*(Trp[i*nsec + j] + Trp[(i+1)*nsec +j]))/ \
    (0.5*(dens[i*nsec +j]+dens[i*nsec + ((j-1)+nsec)%nsec]));
  }

  /* now vrad */
  if (i<nrad && j<nsec)
  {
    vrad[i*nsec +j] += DeltaT*invRinf[i]*((Rmed[i]*Trr[i*nsec +j]- Rmed[i-1]*Trr[(i-1)*nsec + j])*invdiffRmed[i] + \
    (Trp[i*nsec + (j+1)%nsec] -Trp[i*nsec + j])*1.0/(2.0*CUDART_PI_F/nsec) - 0.5*(Trp[i*nsec +j] + Trp[(i-1)*nsec + j]))/ \
    (0.5*(dens[i*nsec +j] + dens[(i-1)*nsec + j]));

  }
}


__global__ void InitComputeAccel(float *CellAbscissa, float *CellOrdinate, float *Rmed, float *cosns, float *sinns, int nsec, int nrad)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = threadIdx.y + blockDim.y*blockIdx.y;

  if (i<nrad && j<nsec)
  {
    CellAbscissa[i*nsec+j]= Rmed[i] * cosns[j];
    CellOrdinate[i*nsec+j]= Rmed[i] * sinns[j];
  }
}


__global__ void ComputeSoundSpeed(float *SoundSpeed, float *dens, float *Rmed, float *energy, int nsec, int nrad,
int Adiabaticc, float ADIABATICINDEX, float FLARINGINDEX, float *AspectRatioRmed)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = threadIdx.y + blockDim.y*blockIdx.y;

  if (i<nrad && j<nsec)
  {
    if (!Adiabaticc){
      SoundSpeed[j+i*nsec] = AspectRatioRmed[i]*sqrtf(1.0*1.0/Rmed[i])*powf(Rmed[i], FLARINGINDEX);
    }
    else SoundSpeed[j+i*nsec] = sqrtf(ADIABATICINDEX*(ADIABATICINDEX-1.0))*energy[j+i*nsec]/dens[j+i*nsec];
  }
}


__global__ void ComputePressureField(float *SoundSpeed, float *dens, float *press, int Adiabaticc, int nrad, int nsec,
  int ADIABATICINDEX, float *energy)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = threadIdx.y + blockDim.y*blockIdx.y;

  if (i<nrad && j<nsec)
  {
    if (~Adiabaticc){
      press[j+i*nsec] = dens[j+i*nsec]*SoundSpeed[j+i*nsec]*SoundSpeed[j+i*nsec];
    }
    else press[j+i*nsec] = (ADIABATICINDEX-1.0)*energy[j+i*nsec];
  }
}

__global__ void ComputeTemperatureField(float *dens, float *temperature, float *press, float *energy, float MU, float R,
float ADIABATICINDEX, int Adiabaticc, int nsec, int nrad)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = threadIdx.y + blockDim.y*blockIdx.y;

  if (i<nrad && j<nsec)
  {
    if (~Adiabaticc){
      temperature[j+i*nsec] = MU/R*press[j+i*nsec]/dens[j+i*nsec];
    }
    else temperature[j+i*nsec] = MU/R*(ADIABATICINDEX-1.0)*energy[j+i*nsec]/dens[j+i*nsec];
  }
}


__global__ void InitLabel (float *label, float xp, float yp, float rhill, float *Rmed, int nrad, int nsec)
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

__global__ void CircumPlanetaryMass (float *dens, float *Surf, float *CellAbscissa, float *CellOrdinate, float xpl, float ypl, int nrad,
  int nsec, float HillRadius, float *mdcp0)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = threadIdx.y + blockDim.y*blockIdx.y;

  float dist;

  if (i<nrad && j<nsec)
  {
    dist = sqrtf((CellAbscissa[j+i*nsec]-xpl)*(CellAbscissa[j+i*nsec]-xpl) + (CellOrdinate[j+i*nsec]-ypl)*(CellOrdinate[j+i*nsec]-ypl));

    if (dist < HillRadius) mdcp0[j+i*nsec] =  Surf[i]* dens[j+i*nsec];
  }
}

template <bool nIsPow2>
__global__ void deviceReduceKernel(float *g_idata, float *g_odata, unsigned int n)
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


__host__ long NearestPowerOf2(long n)
{
  if(!n) return n; //(0 ==2^0)

  int x=1;
  while (x < n)
  {
    x<<=1;
  }
  return x;
}

__host__ bool isPow2(unsigned int x)
{
  return ((x&(x-1)==0));
}

__host__ float deviceReduce(float *in, int N)
{
  float *device_out;
  gpuErrchk(cudaMalloc(&device_out, sizeof(float)*1024));
  gpuErrchk(cudaMemset(device_out, 0, sizeof(float)*1024));

  int threads = 32;
  int blocks = min((int(NearestPowerOf2(N)) + threads - 1) / threads, 1024);
  int smemSize = (threads <= 32) ? 2 * threads * sizeof(float) : threads * sizeof(float);

  bool isPower2 = isPow2(N);
  if(isPower2){
    deviceReduceKernel<true><<<blocks, threads, smemSize>>>(in, device_out, N);
    gpuErrchk(cudaDeviceSynchronize());
  }else{
    deviceReduceKernel<false><<<blocks, threads, smemSize>>>(in, device_out, N);
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

__global__ void MultiplyPolarGridbyConstant(float *dens, int nrad, int nsec, float ScalingFactor)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = threadIdx.y + blockDim.y*blockIdx.y;

  if (i<nrad+1 && j<nsec) dens[j+i*nsec] *= ScalingFactor;
}

__global__ void substep2(float *dens, float *vradint, float *vthetaint, float *temperatureint, int nrad, int nsec, float CVNR,
  float *invdiffRmed, float *invdiffRsup, float *densint, int Adiabaticc, float *Rmed, float dt, float *vradnew,
  float *vthetanew, float *energy, float *energyint)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = threadIdx.y + blockDim.y*blockIdx.y;

  float dv;
  if (i<nrad && j<nsec)
  {
    dv = vradint[(i+1)*nsec + j] - vradint[i*nsec + j];

    if (dv < 0.0) densint[i*nsec + j] = CVNR*CVNR*dens[i*nsec + j]*dv*dv;
    else densint[i*nsec + j] = 0.0;

    dv = vthetaint[i*nsec + (j+1)%nsec] - vthetaint[i*nsec + j];
    if (dv < 0.0) temperatureint[i*nsec + j] = CVNR*CVNR*dens[i*nsec + j]*dv*dv;
    else temperatureint[i*nsec + j] = 0.0;

  }
  __syncthreads();
  i+=1;

  if (i<nrad && j<nsec)
  {
    vradnew[i*nsec + j] = vradint[i*nsec + j] - dt*2.0/(dens[i*nsec + j] + dens[(i-1)*nsec + j])*(densint[i*nsec + j] - densint[(i-1)*nsec + j]) *\
    invdiffRmed[i];

  }
  __syncthreads();
  i-=1;

  if (i<nrad && j<nsec)
  {
    vthetanew[i*nsec + j] = vthetaint[i*nsec + j] - dt*2.0/(dens[i*nsec + j] + dens[i*nsec + ((j-1)+nsec)%nsec])*(densint[i*nsec + j] - \
      densint[(i-1)*nsec + j])* 1.0/(2.0*CUDART_PI_F*Rmed[i]/nsec);

  }
  __syncthreads();
  if (Adiabaticc)
  {
    i+=1;

    if (i<nrad && j<nsec)
    {
      energyint[i*nsec + j] = energy[i*nsec + j] - dt*densint[i*nsec + j]*(vradint[i*nsec + j+1] - vradint[i*nsec + j])*invdiffRsup[i] - \
      dt*temperatureint[i*nsec + j]*(vthetaint[i*nsec + (j+1)%nsec] - vthetaint[i*nsec + j])* 1.0/(2.0*CUDART_PI_F*Rmed[i]/nsec);

    }
  }
}

__global__ void OpenBoundary(float *vrad, float *dens, float *energy, int nsec, float *SigmaMed)
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

__global__ void ReduceCs(float *SoundSpeed, float *cs0, float *cs1, float *csnrm1, float *csnrm2, int nsec, int nrad)
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

__global__ void ReduceMean(float *dens, float *energy, int nsec, float *mean_dens, float *mean_energy, float *mean_dens2,
  float *mean_energy2, int nrad)
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

__global__ void NonReflectingBoundary(float *dens, float *energy, int i_angle, int nsec, float *vrad, float *SoundSpeed,
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

    if(j+i_angle < 0)
    {
      dens[j+i_angle + nsec] = dens[i*nsec + j];
      energy[j+i_angle + nsec] = energy[i*nsec + j];
    }
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
    if (j-i_angle2 < 0)
    {
      vrad_med = SoundSpeed[i*nsec + j]*(dens[(i-1)*nsec + j]-SigmaMed2)/SigmaMed2;
      vrad[i*nsec + j] = 2.*vrad_med - vrad[(i-1)*nsec + j];
    }
  }
}

__global__ void MinusMean(float *dens, float *energy, float SigmaMed, float mean_dens_r, float mean_dens_r2, float mean_energy_r,
  float mean_energy_r2, float EnergyMed, int nsec, int nrad, float SigmaMed2, float EnergyMed2)
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

__global__ void make1Dprofile(float *gridfield, float *GLOBAL_bufarray, int nsec, int nrad)
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


__host__ void make1Dprofilehost(float *gridfield)
{

  dim3 dimGrid( nrad2pot/blocksize2, 1);
  dim3 dimBlock( blocksize2, 1);

  float *gridfield_d, *GLOBAL_bufarray_d;

  gpuErrchk(cudaMalloc((void**)&gridfield_d, size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&GLOBAL_bufarray_d, NRAD*sizeof(float)));

  gpuErrchk(cudaMemcpy(gridfield_d, gridfield, size_grid*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(GLOBAL_bufarray_d, GLOBAL_bufarray, NRAD*sizeof(float), cudaMemcpyHostToDevice));

  make1Dprofile<<<dimGrid, dimBlock>>>(gridfield_d, GLOBAL_bufarray_d, NSEC, NRAD);
  gpuErrchk(cudaDeviceSynchronize());
  gpuErrchk(cudaMemcpy(GLOBAL_bufarray, GLOBAL_bufarray_d, NRAD*sizeof(float), cudaMemcpyDeviceToHost));

  cudaFree(GLOBAL_bufarray_d);
  cudaFree(gridfield_d);
}

__global__ void InitGasVelocities(float *viscosity_array, int nsec, int nrad, int SelfGravity, float *Rmed, float G,
  float ASPECTRATIO, float FLARINGINDEX, float SIGMASLOPE, int CentrifugalBalance, float *vrad, float *vtheta,
  float ViscosityAlpha, float IMPOSEDDISKDRIFT, float SIGMA0, float *SigmaInf, float OmegaFrame, float *Rinf)
  {
    int j = threadIdx.x + blockDim.x*blockIdx.x;
    int i = threadIdx.y + blockDim.y*blockIdx.y;

    float omega, r, ri;

    if (i <= nrad && j < nsec)
    {
      if (i == nrad)
      {
        r = Rmed[nrad - 1];
        ri = Rinf[nrad - 1];
      }
      else {
        r = Rmed[i];
        ri = Rinf[i];
      }

      if (!SelfGravity)
      {
        omega = sqrtf(G*1.0/r/r/r);
        vtheta[i*nsec + j] = omega*r*sqrt(1.0-powf(ASPECTRATIO,2.0)* powf(r,2.0*FLARINGINDEX)* \
        (1.+SIGMASLOPE-2.0*FLARINGINDEX));
      }
      vtheta[i*nsec + j] -= OmegaFrame*r;
      //if (CentrifugalBalance) vtheta[i*nsec + j] = vt_cent[i];

      if (i == nrad) vrad[i*nsec + j] = 0.0;
      else vrad[i*nsec + j] = IMPOSEDDISKDRIFT*SIGMA0/SigmaInf[i]/ri;

      if (ViscosityAlpha) vrad[i*nsec + j] -= 3.0*viscosity_array[i]/r*(-SIGMASLOPE+2.0*FLARINGINDEX+1.0);
      else vrad[i*nsec + j] -= 3.0*viscosity_array[i]/r*(-SIGMASLOPE+.5);

    }
  }


__global__ void ComputeForceKernel(float *CellAbscissa, float *CellOrdinate, float *Surf, float *dens, float x, float y, float rsmoothing,
  float *forcesxi, float *forcesyi, float *forcesxo, float *forcesyo, int nsec, int nrad, float G, float a, float *Rmed,
  int dimfxy, float rh)
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

  __global__ void ViscousTerms(float *vradial, float *vazimutal , float *Drr, float *Dpp, float *divergence, float *Drp,
    float *invdiffRsup, int invdphi, float *invRmed, float *Rsup, float *Rinf, float *invdiffRmed, int nrad, int nsec,
    float *Trr, float *Tpp, float *dens, float *viscosity_array, float onethird, float *Trp, float *invRinf)
 {
   int j = threadIdx.x + blockDim.x*blockIdx.x;
   int i = threadIdx.y + blockDim.y*blockIdx.y;

   if (i<nrad && j<nsec) /* Drr, Dpp and divV computation */
   {
     Drr[i*nsec + j] = (vradial[(i+1)*nsec + j] - vradial[i*nsec + j])*invdiffRsup[i];
     Dpp[i*nsec + j] = (vazimutal[i*nsec + (j+1)%nsec] - vazimutal[i*nsec + j])*invdphi*invRmed[i]+0.5* \
       (vradial[(i+1)*nsec + j]+vradial[i*nsec + j])*invRmed[i];
     divergence[i*nsec + j] = (vradial[i*nsec + (j+1)]*Rsup[i]-vradial[i*nsec + j]*Rinf[i])*invdiffRsup[i] * \
       invRmed[i];
     divergence[i*nsec + j] += (vazimutal[i*nsec + (j+1)%nsec]-vazimutal[i*nsec + j])*invdphi*invRmed[i];

     if (i > 0) Drp[i*nsec + j] = 0.5*(Rinf[i]*(vazimutal[i*nsec + j]*invRmed[i]-vazimutal[(i-1)*nsec + j])*invRmed[i-1])* \
          invdiffRmed[i] + (vradial[i*nsec + j]-vradial[i*nsec + (j-1)%nsec])*invdphi*invRinf[i];
   }

   if (i<nrad && j<nsec) /* TAUrr and TAUpp computation */
   {
     Trr[i*nsec + j] = 2.0*dens[i*nsec + j]*viscosity_array[i]*(Drr[i*nsec + j]-onethird*divergence[i*nsec + j]);
     Tpp[i*nsec + j] = 2.0*dens[i*nsec + j]*viscosity_array[i]*(Dpp[i*nsec + j]-onethird*divergence[i*nsec + j]);

     if (i > 0) Trp[i*nsec + j] = 2.0*0.25*(dens[i*nsec + j] + dens[(i-1)*nsec + j] + dens[(i-1)*nsec + ((j-1)+nsec)%nsec])* \
         viscosity_array[i]*Drp[i*nsec + j];
   }

 }
