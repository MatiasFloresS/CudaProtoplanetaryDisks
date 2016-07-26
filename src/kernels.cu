#include "kernels.cuh"
#include <stdio.h>
using namespace std;


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


    if (ZMPlus)
    {
      gradp *= 1; //gradp *= SG_aniso_coeff;  Definir mas adelante SG_aniso_coeff
    }

    gradphi = (pot[i*nsec+ j] - pot[i*nsec + ((j-1)+nsec)%nsec])*1.0/(2.0*CUDART_PI_F/nsec*Rmed[i]);
    vthetaint[i*nsec + j] = vtheta[i*nsec + j]- dt*(gradp+gradphi);
    vthetaint[i*nsec + j] += dt*IMPOSEDDISKDRIFT*0.5*powRmed[i];

  }

}


__global__ void UpdateVelocities(float *vt, float *vr, float *invRmed, float *Rmed, float *Rsup,
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
    vt[i*nsec +j] += DeltaT*invRmed[i]*((Rsup[i]*Trp[(i+1)*nsec+ j]-Rinf[i]*Trp[i*nsec +j])*invdiffRsup[i] + \
    (Tpp[i*nsec +j]-Tpp[i*nsec + ((j-1)+nsec)%nsec])*1.0/(2.0*CUDART_PI_F/nsec) + 0.5*(Trp[i*nsec + j] + Trp[(i+1)*nsec +j]))/ \
    (0.5*(dens[i*nsec +j]+dens[i*nsec + ((j-1)+nsec)%nsec]));
  }

  /* now vrad */
  if (i<nrad && j<nsec)
  {
    vr[i*nsec +j] += DeltaT*invRinf[i]*((Rmed[i]*Trr[i*nsec +j]- Rmed[i-1]*Trr[(i-1)*nsec + j])*invdiffRmed[i] + \
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
    if (~Adiabaticc){
      SoundSpeed[j+i*nsec] = AspectRatioRmed[i]*sqrtf(1.0*1.0/Rmed[i])*powf(Rmed[i], FLARINGINDEX);
    }
    else SoundSpeed[j+i*nsec] = sqrtf(ADIABATICINDEX*(ADIABATICINDEX-1.0)*energy[j+i*nsec]/dens[j+i*nsec]);
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

__global__ void MultiplyPolarGridbyConstant(float *dens, float *fieldsrc, int nrad, int nsec, float ScalingFactor)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = threadIdx.y + blockDim.y*blockIdx.y;

  if (i<nrad+1 && j<nsec) fieldsrc[j+i*nsec] *= ScalingFactor;

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

__global__ void ReduceCs(float *SoundSpeed, float *cs0, float *cs1, int nsec)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i=0;

  if(j<nsec)
  {
    cs0[i*nsec +j] = SoundSpeed[i*nsec +j];
    cs1[i*nsec +j] = SoundSpeed[(i+1)*nsec +j];
  }
}

__global__ void ReduceMean(float *dens, float *energy, int nsec, float *mean_dens, float *mean_energy)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = 0;

  if(j<nsec)
  {
    mean_dens[i*nsec +j] = dens[i*nsec+ j];
    mean_energy[i*nsec +j] = energy[i*nsec +j];
  }
}

__global__ void NonReflectingBoundary(float *dens, float *energy, int i_angle, int nsec, float *vrad, float *SoundSpeed,
  float SigmaMed)
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
}

/*__global__ void ComputeForceKernel(float *CellAbscissa, float *CellOrdinate, float *Surf, float *dens, float x, float rsmoothing,
  int dimfxy, float mass, float a, float *fxi, float *fxo, float *fyi, float *fyo, float *Rmed)
  {

    int x = threadIdx.x + blockDim.x*blockIdx.x;
    int i = threadIdx.y + blockDim.y*blockIdx.y;

    if (i<nrad && j<nsec)
    {
      for (int k = 0; k < dimfxy; k++) {
        fxi[k] += i;
      }
    }
  }*/
