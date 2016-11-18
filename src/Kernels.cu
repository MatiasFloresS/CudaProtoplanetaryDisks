#include "Main.cuh"

extern int blocksize2, size_grid, NRAD, NSEC;

extern double *GLOBAL_bufarray;
extern double *gridfield_d, *GLOBAL_bufarray_d, *axifield_d, *SG_Accr_d, *GLOBAL_AxiSGAccr_d;

extern double ASPECTRATIO, TRANSITIONWIDTH, TRANSITIONRATIO, TRANSITIONRADIUS;
extern double PhysicalTime, PhysicalTimeInitial, LAMBDADOUBLING;

extern dim3 dimGrid, dimBlock, dimGrid4;

__global__ void Substep1Kernel (double *Pressure, double *Dens, double *VradInt, double *invdiffRmed, double *Potential,
   double *Rinf, double *invRinf, double *Vrad, double *VthetaInt, double *Vtheta, double *Rmed, double dt,
   int nrad, int nsec, double OmegaFrame, int ZMPlus, double IMPOSEDDISKDRIFT, double SIGMASLOPE, double *powRmed)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = threadIdx.y + blockDim.y*blockIdx.y;
  double gradp, gradphi, vt2, supp_torque, dxtheta, invdxtheta;
  double valor;
  double valor2;
  double valor3;

  // i=1->nrad , j=0->nsec
  if (i > 0 && i<nrad && j<nsec){
    gradp = (Pressure[i*nsec + j] - Pressure[(i-1)*nsec + j])*2.0/(Dens[i*nsec + j] + Dens[(i-1)*nsec + j])*invdiffRmed[i];
    gradphi = (Potential[i*nsec + j]-Potential[(i-1)*nsec + j])*invdiffRmed[i];
    vt2 = Vtheta[i*nsec + j] + Vtheta[i*nsec + (j+1)%nsec] + Vtheta[(i-1)*nsec + j] + Vtheta[(i-1)*nsec + (j+1)%nsec];
    vt2 = vt2/4.0+Rinf[i]*OmegaFrame;
    vt2 = vt2*vt2;
    valor = -gradp-gradphi;
    valor2 = vt2*invRinf[i];
    valor3 = valor + valor2;
    //valor = Vrad[i*nsec + j] + valor;
    VradInt[i*nsec + j] = (double)-gradp-(double)gradphi+ (double)vt2*(double)invRinf[i];
    if (i == 1 && j == 0)
    {
      printf("gradp %.30g\n", -gradp);
      printf("gradphi %.30g\n", -gradphi);
      printf("gradp+gradphi %.30g\n", -gradp-gradphi);
      printf("vt2 %.30g\n", vt2*invRinf[i]);
      printf("invdiffRmed %.30g\n", invdiffRmed[i]);
      printf("VradInt %.30g\n", VradInt[i*nsec+j]);
      printf("Vrad %.30g\n", Vrad[i*nsec+j]);
      printf("suma %.30g\n", (-gradp - gradphi + vt2*invRinf[i]));
      printf("dt %.30g\n", dt);
      printf("valor %.30g\n", valor3);
    }
    __syncthreads();
    if (i == 2&& j == 0)
    {
      printf("gradp %.30g\n", -gradp);
      printf("gradphi %.30g\n", -gradphi);
      printf("gradp+gradphi %.30g\n", -gradp-gradphi);
      printf("vt2 %.30g\n", vt2*invRinf[i]);
      printf("invdiffRmed %.30g\n", invdiffRmed[i]);
      printf("VradInt %.30g\n", VradInt[i*nsec+j]);
      printf("Vrad %.30g\n", Vrad[i*nsec+j]);
      printf("suma %.30g\n", (-gradp - gradphi + vt2*invRinf[i]));
      printf("dt %.30g\n", dt);
      printf("valor %.30g\n", valor3);
    }
  }


  // i=0->nrad ,   j=0->nsec
  if (i<nrad && j<nsec){
    supp_torque = IMPOSEDDISKDRIFT*0.5*pow(Rmed[i], -2.5+SIGMASLOPE);
    dxtheta = 2.0*M_PI/(double)nsec*Rmed[i];
    invdxtheta = 1.0/dxtheta;

    gradp = (Pressure[i*nsec + j] - Pressure[i*nsec + ((j-1)+nsec)%nsec])*2.0/(Dens[i*nsec +j] +Dens[i*nsec + ((j-1)+nsec)%nsec]) \
    *invdxtheta;

    if (ZMPlus) gradp *= 1; //gradp *= SG_aniso_coeff;  Definir mas adelante SG_aniso_coeff

    gradphi = (Potential[i*nsec+ j] - Potential[i*nsec + ((j-1)+nsec)%nsec])*invdxtheta;
    VthetaInt[i*nsec + j] = Vtheta[i*nsec + j]- dt*(gradp+gradphi);
    VthetaInt[i*nsec + j] += dt*supp_torque;
  }
}

__global__ void Substep3Kernel (double *Dens, double *Qplus, double *viscosity_array, double *TAURR, double *TAURP,double *TAUPP,
  double *DivergenceVelocity, int nrad, int nsec, double *Rmed, int Cooling, double *EnergyNew, double dt, double *EnergyMed,
  double *SigmaMed, double *CoolingTimeMed, double *Energy, double ADIABATICINDEX, double *QplusMed)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = threadIdx.y + blockDim.y*blockIdx.y;
  double den, num;

  if (i > 0 && i<nrad && j<nsec){
    if (viscosity_array[i] != 0.0){
      Qplus[i*nsec + j] = Dens[i*nsec + j]*(TAURR[i*nsec + j]*TAURR[i*nsec + j] + \
        TAURP[i*nsec + j]* TAURP[i*nsec + j] + TAUPP[i*nsec + j]*TAUPP[i*nsec + j]);

      Qplus[i*nsec + j] += Dens[i*nsec + j]*DivergenceVelocity[i*nsec + j]* \
        DivergenceVelocity[i*nsec + j];
    }
    else
      Qplus[i*nsec + j] = 0.0;

  }
  __syncthreads();
  if (i == 1 && j == 0)
  {
    //printf("Dens =  %g\n",Dens[i*nsec + j]);
    //printf("Qplus = %g\n",Qplus[i*nsec+j]);
    //printf("TAURR = %g\n",TAURR[i*nsec + j]);
    //printf("TAURP = %g\n",TAURP[i*nsec + j]);
    //printf("TAUPP = %g\n",TAUPP[i*nsec + j]);
    //printf("DivergenceVelocity = %g\n",DivergenceVelocity[i*nsec + j]);
  }

  if (i==0 && j<nsec){
    /* We calculate the heating source term Qplus for i=0 */

    if (viscosity_array[i] != 0) {
      /* power-law extrapolation */
      Qplus[i*nsec + j] = Qplus[(i+1)*nsec + j]*exp(log(Qplus[(i+1)*nsec + j]/Qplus[(i+2)*nsec + j]) * \
        log(Rmed[i]/Rmed[i+1]) / log(Rmed[i+1]/Rmed[i+2]));
    }
    else
      Qplus[i*nsec + j] = 0.0;
  }

  __syncthreads();

  /* Now we can update energy with source terms from i=0 */
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



__global__ void UpdateVelocitiesKernel (double *VthetaInt, double *VradInt, double *invRmed, double *Rmed, double *Rsup,
  double *Rinf, double *invdiffRmed, double *invdiffRsup, double *Dens, double *invRinf, double *TAURR, double *TAURP,
  double *TAUPP, double DeltaT, int nrad, int nsec)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = threadIdx.y + blockDim.y*blockIdx.y;

  double dphi, invdphi;

  /* Now we can update velocities
     with the viscous source term
     of Navier-Stokes equation */

  /* vtheta first */
  if (i > 0 && i<nrad-1 && j<nsec){
    dphi = 2.0*M_PI/(double)nsec;
    invdphi = 1.0/dphi;

    VthetaInt[i*nsec +j] += DeltaT*invRmed[i]*((Rsup[i]*TAURP[(i+1)*nsec+ j]-Rinf[i]*TAURP[i*nsec +j])*invdiffRsup[i] + \
    (TAUPP[i*nsec +j]-TAUPP[i*nsec + ((j-1)+nsec)%nsec])*invdphi + 0.5*(TAURP[i*nsec + j] + TAURP[(i+1)*nsec +j]))/ \
    (0.5*(Dens[i*nsec +j]+Dens[i*nsec + ((j-1)+nsec)%nsec]));
  }

  /* now vrad */
  if (i > 0 && i<nrad && j<nsec){
    dphi = 2.0*M_PI/(double)nsec;
    invdphi = 1.0/dphi;

    VradInt[i*nsec +j] += DeltaT*invRinf[i]*((Rmed[i]*TAURR[i*nsec +j]- Rmed[i-1]*TAURR[(i-1)*nsec + j])*invdiffRmed[i] + \
    (TAURP[i*nsec + (j+1)%nsec] -TAURP[i*nsec + j])*invdphi - 0.5*(TAUPP[i*nsec +j] + TAUPP[(i-1)*nsec + j]))/ \
    (0.5*(Dens[i*nsec +j] + Dens[(i-1)*nsec + j]));

    /*if (i == 1 && j == 0)
    {
      printf("%.10f\n", DeltaT);
      printf("%.10f\n", invRinf[i]);
      printf("%.10f\n", TAURR[i*nsec + j]);
      printf("%.10f\n", TAURR[(i-1)*nsec + j]);
      printf("%.10f\n", TAURP[i*nsec + (j+1)%nsec]);
      printf("%.10f\n", TAURP[i*nsec + j]);
      printf("%.10f\n", TAUPP[i*nsec +j]);
      printf("%.10f\n", TAUPP[(i-1)*nsec + j]);
      printf("%.10f\n", Dens[i*nsec +j] + Dens[(i-1)*nsec + j]);
      printf("%.10f\n",  DeltaT*invRinf[i]*((Rmed[i]*TAURR[i*nsec +j]- Rmed[i-1]*TAURR[(i-1)*nsec + j])*invdiffRmed[i] + \
      (TAURP[i*nsec + (j+1)%nsec] -TAURP[i*nsec + j])*1.0/(2.0*M_PI/nsec) - 0.5*(TAUPP[i*nsec +j] + TAUPP[(i-1)*nsec + j]))/ \
      (0.5*(Dens[i*nsec +j] + Dens[(i-1)*nsec + j])));
      printf("vr final %.10f\n", VradInt[i*nsec + j]);
    }
    if ( i == 2 && j == 0) printf("vr final2 %.10f\n", VradInt[i*nsec + j]);*/
  }
}



__global__ void InitComputeAccelKernel (double *CellAbscissa, double *CellOrdinate, double *Rmed, int nsec, int nrad)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = threadIdx.y + blockDim.y*blockIdx.y;


  if (i<nrad && j<nsec){
    CellAbscissa[i*nsec+j] = Rmed[i] * cos((2.0*M_PI*(double)j)/(double)nsec);
    CellOrdinate[i*nsec+j] = Rmed[i] * sin((2.0*M_PI*(double)j)/(double)nsec);
  }
}


__global__ void ComputeSoundSpeedKernel (double *SoundSpeed, double *Dens, double *Rmed, double *Energy, int nsec, int nrad,
  int Adiabatic, double ADIABATICINDEX, double FLARINGINDEX, double ASPECTRATIO, double TRANSITIONWIDTH,
  double TRANSITIONRADIUS, double TRANSITIONRATIO, double PhysicalTime, double PhysicalTimeInitial, double LAMBDADOUBLING)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = threadIdx.y + blockDim.y*blockIdx.y;

  double AspectRatio;
  if (i<nrad && j<nsec){
    if (!Adiabatic){
      AspectRatio = AspectRatioDevice(Rmed[i], ASPECTRATIO, TRANSITIONWIDTH, TRANSITIONRADIUS, TRANSITIONRATIO,
        PhysicalTime, PhysicalTimeInitial, LAMBDADOUBLING);
      SoundSpeed[i*nsec + j] = AspectRatio*sqrt(G*1.0/Rmed[i])*pow(Rmed[i], FLARINGINDEX);
    }
    else SoundSpeed[i*nsec + j] = sqrt(ADIABATICINDEX*(ADIABATICINDEX-1.0)*Energy[i*nsec + j]/Dens[i*nsec + j]);
  }
}


__global__ void ComputePressureFieldKernel (double *SoundSpeed, double *Dens, double *Pressure, int Adiabatic, int nrad,
  int nsec, double ADIABATICINDEX, double *Energy) /* LISTO */
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = threadIdx.y + blockDim.y*blockIdx.y;

  if (i<nrad && j<nsec){
    if (!Adiabatic)
      Pressure[i*nsec + j] = Dens[i*nsec + j]*SoundSpeed[i*nsec + j]*SoundSpeed[i*nsec + j];

    /* Since SoundSpeed is not update from initialization, cs remains axisymmetric*/
    else Pressure[i*nsec + j] = (ADIABATICINDEX-1.0)*Energy[i*nsec + j];
  }
}


__global__ void ComputeTemperatureFieldKernel (double *Dens, double *Temperature, double *Pressure, double *Energy,
  double ADIABATICINDEX, int Adiabatic, int nsec, int nrad) /* LISTO */
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = threadIdx.y + blockDim.y*blockIdx.y;

  if (i<nrad && j<nsec){
    if (!Adiabatic) Temperature[i*nsec + j] = MU/R*Pressure[i*nsec + j]/Dens[i*nsec + j];
    else Temperature[i*nsec + j] = MU/R*(ADIABATICINDEX-1.0)*Energy[i*nsec + j]/Dens[i*nsec + j];
  }
}


/* LISTO */
__global__ void InitLabelKernel (double *Label, double xp, double yp, double rhill, double *Rmed, int nrad, int nsec)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = threadIdx.y + blockDim.y*blockIdx.y;

  if (i<nrad && j<nsec){
    double distance, angle, x, y;
    angle = (double)j / (double)nsec*2.0*M_PI;
    x = Rmed[i] * cos(angle);
    y = Rmed[i] * sin(angle);
    distance = sqrt((x - xp) * (x - xp) + (y - yp)*(y -yp));

    if (distance < rhill) Label[i*nsec + j] = 1.0;
    else Label[i*nsec + j] = 0.0;

  }
}


__global__ void CircumPlanetaryMassKernel (double *Dens, double *Surf, double *CellAbscissa, double *CellOrdinate,
  double xpl, double ypl, int nrad, int nsec, double HillRadius, double *mdcp0) /* LISTA */
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = threadIdx.y + blockDim.y*blockIdx.y;

  double dist;

  if (i<nrad && j<nsec){
    dist = sqrt((CellAbscissa[i*nsec + j]-xpl)*(CellAbscissa[i*nsec + j]-xpl) + (CellOrdinate[i*nsec + j]-ypl) * \
    (CellOrdinate[i*nsec + j]-ypl));
    if (dist < HillRadius) mdcp0[i*nsec + j] = Surf[i]* Dens[i*nsec + j];
    else mdcp0[i*nsec + j] = 0.0;
  }
}


template <bool nIsPow2>
__global__ void DeviceReduceKernel (double *g_idata, double *g_odata, unsigned int n)
{
  extern __shared__ double sdata[];

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int blockSize = blockDim.x;
  unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
  unsigned int gridSize = blockSize*2*gridDim.x;

  double mySum = 0.0;

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


__host__ double DeviceReduce (double *in, int N)
{
  double *device_out;
  gpuErrchk(cudaMalloc(&device_out, sizeof(double)*1024));
  gpuErrchk(cudaMemset(device_out, 0, sizeof(double)*1024));

  int threads = 32;
  int blocks = min((int(NearestPowerOf2(N)) + threads - 1) / threads, 1024);
  int smemSize = (threads <= 32) ? 2 * threads * sizeof(double) : threads * sizeof(double);

  bool isPower2 = IsPow2(N);
  if(isPower2){
    DeviceReduceKernel<true><<<blocks, threads, smemSize>>>(in, device_out, N);
    gpuErrchk(cudaDeviceSynchronize());
  }
  else{
    DeviceReduceKernel<false><<<blocks, threads, smemSize>>>(in, device_out, N);
    gpuErrchk(cudaDeviceSynchronize());
  }

  double *h_odata = (double *) malloc(blocks*sizeof(double));
  double sum = 0.0;

  gpuErrchk(cudaMemcpy(h_odata, device_out, blocks * sizeof(double),cudaMemcpyDeviceToHost));
  for (int i=0; i<blocks; i++){
    sum += h_odata[i];
  }
  cudaFree(device_out);
  free(h_odata);
	return sum;
}


/* LISTA */
__global__ void MultiplyPolarGridbyConstantKernel (double *Dens, int nrad, int nsec, double ScalingFactor)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = threadIdx.y + blockDim.y*blockIdx.y;

  if (i<=nrad && j<nsec)
    Dens[i*nsec + j] *= ScalingFactor;
}



__global__ void Substep2Kernel (double *Dens, double *VradInt, double *VthetaInt, double *TemperInt, int nrad,
  int nsec, double CVNR, double *invdiffRmed, double *invdiffRsup, double *DensInt, int Adiabatic, double *Rmed,
  double dt, double *VradNew, double *VthetaNew, double *Energy, double *EnergyInt)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = threadIdx.y + blockDim.y*blockIdx.y;

  double dv, dxtheta, invdxtheta;

  if (i<nrad && j<nsec){
    dv = VradInt[(i+1)*nsec + j] - VradInt[i*nsec + j];

    if (dv < 0.0)
      DensInt[i*nsec + j] = CVNR*CVNR*Dens[i*nsec + j]*dv*dv;
    else
      DensInt[i*nsec + j] = 0.0;

    dv = VthetaInt[i*nsec + (j+1)%nsec] - VthetaInt[i*nsec + j];

    if (dv < 0.0)
      TemperInt[i*nsec + j] = CVNR*CVNR*Dens[i*nsec + j]*dv*dv;
    else
      TemperInt[i*nsec + j] = 0.0;
  }
  __syncthreads();

  if (i>0 && i<nrad && j<nsec){
    VradNew[i*nsec + j] = VradInt[i*nsec + j] - dt*2.0/(Dens[i*nsec + j] + Dens[(i-1)*nsec + j])* \
      (DensInt[i*nsec + j] - DensInt[(i-1)*nsec + j]) * invdiffRmed[i];
  }

  if (i<nrad && j<nsec){
    dxtheta = 2.0*M_PI/(double)nsec*Rmed[i];
    invdxtheta = 1.0/dxtheta;

    VthetaNew[i*nsec + j] = VthetaInt[i*nsec + j] - dt*2.0/(Dens[i*nsec + j] + Dens[i*nsec + ((j-1)+nsec)%nsec])* \
      (TemperInt[i*nsec + j] - TemperInt[i*nsec + ((j-1)+nsec)%nsec])* invdxtheta;
  }


  /* If gas disk is adiabatic, we add artificial viscosity as a source */
  /* term for advection of thermal energy polargrid */
  if (Adiabatic){
    if (i>0 && i<nrad && j<nsec){
      dxtheta = 2.0*M_PI/(double)nsec*Rmed[i];
      invdxtheta = 1.0/dxtheta;

      EnergyInt[i*nsec + j] = Energy[i*nsec + j] - dt*DensInt[i*nsec + j]* \
        (VradInt[(i+1)*nsec + j] - VradInt[i*nsec + j])*invdiffRsup[i] - \
        dt*TemperInt[i*nsec + j]*(VthetaInt[i*nsec + (j+1)%nsec] - VthetaInt[i*nsec + j])* invdxtheta;
    }
  }
}


__global__ void OpenBoundaryKernel (double *Vrad, double *Dens, double *Energy, int nsec, double *SigmaMed)
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


__global__ void ReduceCsKernel (double *SoundSpeed, double *cs0, double *cs1, double *csnrm1, double *csnrm2, int nsec, int nrad)
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


__global__ void ReduceMeanKernel (double *Dens, double *Energy, int nsec, double *mean_dens, double *mean_energy,
  double *mean_dens2, double *mean_energy2, int nrad)
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


__global__ void NonReflectingBoundaryKernel (double *Dens, double *Energy, int i_angle, int nsec, double *Vrad, double *SoundSpeed,
  double SigmaMed, int nrad, double SigmaMed2, int i_angle2)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = 1;
  double Vrad_med;

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
    Vrad[i*nsec + j] = 2.0*Vrad_med-Vrad[(i+1)*nsec + j];
    if (j == 0) {
      //printf("vrad = %g\n",Vrad[i*nsec+j] );
      //printf("vrad = %g\n",Vrad[(i+1)*nsec+j] );
    }
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


__global__ void MinusMeanKernel (double *Dens, double *Energy, double SigmaMed, double mean_dens_r, double mean_dens_r2,
  double mean_energy_r,double mean_energy_r2, double EnergyMed, int nsec, int nrad, double SigmaMed2, double EnergyMed2)
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


__global__ void Make1DprofileKernel (double *gridfield, double *axifield, int nsec, int nrad)
{
  int i = threadIdx.x + blockDim.x*blockIdx.x;
  int j;

  if (i < nrad){
    double sum = 0.0;

    for (j = 0; j < nsec; j++)
      sum += gridfield[i*nsec + j];

    axifield[i] = sum/(double)nsec;
  }
}


__host__ void Make1Dprofile (int option)
{

  /* GLOBAL AxiSGAccr option */
  if (option == 1){
    gpuErrchk(cudaMemcpy(gridfield_d, SG_Accr_d, size_grid*sizeof(double), cudaMemcpyDeviceToDevice));
    //gpuErrchk(cudaMemcpy(GLOBAL_AxiSGAccr_d, axifield_d, NRAD*sizeof(double), cudaMemcpyDeviceToHost));

  }
  /* GLOBAL_bufarray option */
  if (option == 2){
    //gpuErrchk(cudaMemcpy(gridfield_d, SG_Accr_d, size_grid*sizeof(double), cudaMemcpyDeviceToDevice));
    //gpuErrchk(cudaMemcpy(GLOBAL_AxiSGAccr_d, axifield_d, NRAD*sizeof(double), cudaMemcpyDeviceToHost));
  }

  Make1DprofileKernel<<<dimGrid4, dimBlock>>>(gridfield_d, axifield_d, NSEC, NRAD);
  gpuErrchk(cudaDeviceSynchronize());

}


/* LISTO */
__global__ void InitGasVelocitiesKernel (double *viscosity_array, int nsec, int nrad, int SelfGravity, double *Rmed,
  double ASPECTRATIO, double FLARINGINDEX, double SIGMASLOPE, int CentrifugalBalance, double *Vrad, double *Vtheta,
  double ViscosityAlpha, double IMPOSEDDISKDRIFT, double SIGMA0, double *SigmaInf, double OmegaFrame, double *Rinf, double *vt_cent)
{
    int j = threadIdx.x + blockDim.x*blockIdx.x;
    int i = threadIdx.y + blockDim.y*blockIdx.y;

    double omega, r, ri, algo;

    if (i <= nrad && j < nsec){
      if (i == nrad){
        r = Rmed[nrad - 1];
        ri = Rinf[nrad - 1];
      }
      else{
        r = Rmed[i];
        ri = Rinf[i];
      }

      if (!SelfGravity){
        omega = sqrt(G*1.0/r/r/r);
        Vtheta[i*nsec + j] = omega*r*sqrt(1.0-pow((double)ASPECTRATIO,2.0)*pow(r,2.0*FLARINGINDEX)* \
        (1.+SIGMASLOPE-2.0*FLARINGINDEX));
      }
      Vtheta[i*nsec + j] -= OmegaFrame*r;
      //if (i == 1 && j == 1) printf("%.10g\n", OmegaFrame);

      if (CentrifugalBalance) Vtheta[i*nsec + j] = vt_cent[i];

      if (i == nrad) Vrad[i*nsec + j] = 0.0;
      else {
        Vrad[i*nsec + j] = IMPOSEDDISKDRIFT*SIGMA0/SigmaInf[i]/ri;

        if (ViscosityAlpha) Vrad[i*nsec + j] -= 3.0*viscosity_array[i]/r*(-SIGMASLOPE+2.0*FLARINGINDEX+1.0);
        else Vrad[i*nsec + j] -= 3.0*viscosity_array[i]/r*(-SIGMASLOPE+.5);
      }
      __syncthreads();

      if (i == 0 && j < nsec) Vrad[j] = Vrad[nrad*nsec + j] = 0.0;

    }
  }



__global__ void ComputeForceKernel (double *CellAbscissa, double *CellOrdinate, double *Surf, double *Dens, double x,
  double y, double rsmoothing, int nsec, int nrad, double a, double *Rmed, int dimfxy, double rh, double *fxi,
  double *fxo, double *fyi, double *fyo, int k)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = threadIdx.y + blockDim.y*blockIdx.y;

  double cellmass, dx, dy, d2, InvDist3, dist2, distance, resultxi, resultyi;
  double resultxo, resultyo, hillcutfactor, hill_cut;

  if (i<nrad && j<nsec){
    cellmass = Surf[i]* Dens[i*nsec + j];
    dx = CellAbscissa[i*nsec + j] - x;
    dy = CellOrdinate[i*nsec + j] - y;
    d2 = dx*dx + dy*dy;
    dist2 = d2 + rsmoothing*rsmoothing;
    distance = sqrt(dist2);
    InvDist3 = 1.0/dist2/distance;

    hillcutfactor =  (double) k / (double)(dimfxy-1);
    if (k != 0){
      rh *= hillcutfactor;
      hill_cut = 1.-exp(-d2/(rh*rh));
    }
    else
      hill_cut = 1.;

    if (Rmed[i] < a){
        fxi[i*nsec + j] = G * cellmass * dx * InvDist3 * hill_cut;
        fyi[i*nsec + j] = G * cellmass * dy * InvDist3 * hill_cut;
    }
    else{
      fxo[i*nsec + j] = G * cellmass * dx * InvDist3 * hill_cut;
      fyo[i*nsec + j] = G * cellmass * dy * InvDist3 * hill_cut;
    }
  }
}



__global__ void ViscousTermsKernel (double *Vradial, double *Vazimutal , double *DRR, double *DPP, double *DivergenceVelocity,
  double *DRP, double *invdiffRsup, double *invRmed, double *Rsup, double *Rinf, double *invdiffRmed, int nrad,
  int nsec, double *TAURR, double *TAUPP, double *dens, double *viscosity_array, double *TAURP, double *invRinf)
{
   int j = threadIdx.x + blockDim.x*blockIdx.x;
   int i = threadIdx.y + blockDim.y*blockIdx.y;

   double dphi, invdphi, onethird;
   if (i<nrad && j<nsec){ /* Drr, Dpp and divV computation */
     dphi = 2.0*M_PI/(double)nsec;
     invdphi = 1.0/dphi;
     onethird = 1.0/3.0;

     DRR[i*nsec + j] = (Vradial[(i+1)*nsec + j] - Vradial[i*nsec + j])*invdiffRsup[i];

    // if ( i == 1 && j == 0) printf("drr %.10f\n", -0.0000575267/invdiffRsup[i]);

     //if ( i == 1 && j == 0) printf("vradial %.10f\n", Vradial[(i+1)*nsec+j] - Vradial[i*nsec+j]);

    //  if (i == 1 && j == 0) printf("vradial %g\n", Vradial[(i+1)*nsec + j]);
    //  if (i == 1 && j == 0) printf("vradial %g\n", Vradial[(i)*nsec + j]);
    //  if (i == 1 && j == 0) printf("vradial %g\n", Vradial[(i+1)*nsec + j] - Vradial[(i)*nsec + j]);
     DPP[i*nsec + j] = (Vazimutal[i*nsec + (j+1)%nsec] - Vazimutal[i*nsec + j])*invdphi*invRmed[i]+0.5* \
       (Vradial[(i+1)*nsec + j]+Vradial[i*nsec + j])*invRmed[i];

     DivergenceVelocity[i*nsec + j] = (Vradial[(i+1)*nsec + j]*Rsup[i]-Vradial[i*nsec + j]*Rinf[i])*invdiffRsup[i] * \
       invRmed[i];
     DivergenceVelocity[i*nsec + j] += (Vazimutal[i*nsec + (j+1)%nsec]-Vazimutal[i*nsec + j])*invdphi*invRmed[i];

     if (i > 0) DRP[i*nsec + j] = 0.5*(Rinf[i]*(Vazimutal[i*nsec + j]*invRmed[i]-Vazimutal[(i-1)*nsec + j]*invRmed[i-1])* \
          invdiffRmed[i] + (Vradial[i*nsec + j]-Vradial[i*nsec + ((j-1)+nsec)%nsec])*invdphi*invRinf[i]);
   }

   __syncthreads();

   if (i<nrad && j<nsec){ /* TAUrr and TAUpp computation */
     TAURR[i*nsec + j] = 2.0*dens[i*nsec + j]*viscosity_array[i]*(DRR[i*nsec + j]-onethird*DivergenceVelocity[i*nsec + j]);
     TAUPP[i*nsec + j] = 2.0*dens[i*nsec + j]*viscosity_array[i]*(DPP[i*nsec + j]-onethird*DivergenceVelocity[i*nsec + j]);

     if (i > 0){
      TAURP[i*nsec + j] = 2.0*0.25*(dens[i*nsec + j] + dens[(i-1)*nsec + j] + \
      dens[(i-1)*nsec + ((j-1)+nsec)%nsec] + dens[i*nsec + ((j-1)+nsec)%nsec])* \
      viscosity_array[i]*DRP[i*nsec + j];
    }
    if (i == 1 && j == 0)
    {
      //printf("TAURR = %g\n",TAURR[i*nsec + j]);
      //printf("TAURP = %g\n",TAURP[i*nsec + j]);
      //printf("TAUPP = %g\n",TAUPP[i*nsec + j]);
      //printf("DivergenceVelocity = %g\n",DivergenceVelocity[i*nsec + j]);
      //printf("Drr = %g\n",DRR[i*nsec + j]);
      //printf("DRP = %g\n",DRP[i*nsec + j]);
      //printf("DPP = %g\n",DPP[i*nsec + j]);
    }

   }
 }


__global__ void LRMomentaKernel (double *RadMomP, double *RadMomM, double *ThetaMomP, double *ThetaMomM, double *Dens,
  double *Vrad, double *Vtheta, int nrad, int nsec, double *Rmed, double OmegaFrame)
{
   int j = threadIdx.x + blockDim.x*blockIdx.x;
   int i = threadIdx.y + blockDim.y*blockIdx.y;

   if (i<nrad && j<nsec){
     RadMomP[i*nsec + j] = Dens[i*nsec + j] * Vrad[(i+1)*nsec + j]; // (i+1)*nsec
     RadMomM[i*nsec + j] = Dens[i*nsec + j] * Vrad[i*nsec + j];
     /* it is the angular momentum -> ThetaMomP */
     ThetaMomP[i*nsec + j] = Dens[i*nsec + j] * (Vtheta[i*nsec + (j+1)%nsec]+Rmed[i]*OmegaFrame)*Rmed[i];
     ThetaMomM[i*nsec + j] = Dens[i*nsec + j] * (Vtheta[i*nsec + j]+Rmed[i]*OmegaFrame)*Rmed[i];


     if ( i == 50 && j == 300)
     {
       printf("RadMomP%.20f\n", RadMomP[i*nsec + j]);
       printf("RadMomM%.20f\n", RadMomM[i*nsec + j]);
       printf("ThetaMomP%.20f\n", ThetaMomP[i*nsec + j]);
       printf("ThetaMomM%.20f\n", ThetaMomM[i*nsec + j]);
     }
   }
 }


__global__ void ExtQtyKernel (double *ExtLabel, double *Dens, double *Label, int nsec, int nrad)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = threadIdx.y + blockDim.y*blockIdx.y;

  if (i<nrad && j<nsec)
    ExtLabel[i*nsec + j] = Dens[i*nsec + j]*Label[i*nsec + j];
}


__global__ void StarRadKernel (double *Qbase2, double *Vrad, double *QStar, double dt, int nrad, int nsec,
  double *invdiffRmed, double *Rmed, double *dq)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = threadIdx.y + blockDim.y*blockIdx.y;

  double dqm, dqp;

  if (i<nrad && j<nsec){
    if ((i == 0 || i == nrad-1)) dq[j*nrad + i] = 0.0;
    else {
      dqm = (Qbase2[i*nsec + j] - Qbase2[(i-1)*nsec + j])*invdiffRmed[i];
      dqp = (Qbase2[(i+1)*nsec + j] - Qbase2[i*nsec + j])*invdiffRmed[i+1];

      if (dqp * dqm > 0.0)
        dq[j*nrad + i] = 2.0*dqp*dqm/(dqp+dqm);
      else
        dq[j*nrad + i] = 0.0;
    }
    __syncthreads();

    if (Vrad[i*nsec + j] > 0.0)
      QStar[i*nsec + j] = Qbase2[(i-1)*nsec + j] + (Rmed[i]-Rmed[i-1]-Vrad[i*nsec + j]*dt)*0.5*dq[j*nrad + i-1];
    else
      QStar[i*nsec + j] = Qbase2[i*nsec + j]-(Rmed[i+1]-Rmed[i]+Vrad[i*nsec + j]*dt)*0.5*dq[j*nrad + i];

    if ( i == 50 && j == 300)
    {
      printf("QSSS %.20f\n", QStar[i*nsec + j]);
      printf("Vrad %.20f\n", Vrad[i*nsec + j]);
      printf("coso %.20f\n", (Rmed[i]-Rmed[i-1]-Vrad[i*nsec + j]*dt)*0.5*dq[j*nrad + i-1]);
      printf("coso2 %.20f\n", Qbase2[(i-1)*nsec + j]);

    }
  }
  __syncthreads();
  if (i == 0 && j<nsec)
    QStar[j] = QStar[j+nsec*nrad] = 0.0;
}


__global__ void ComputeFFTKernel (double *Radii, cufftComplex *SGP_Kr, cufftComplex *SGP_Kt, double SGP_eps, int nrad, int nsec,
cufftComplex *SGP_Sr, cufftComplex *SGP_St, double *Dens, double *Rmed, double *Kr_aux, double *Kt_aux)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = threadIdx.y + blockDim.y*blockIdx.y;
  double u, cosj, sinj, coshu, expu, den_SGP_K, theta, base;
  double a, var, radii;

  if (i<2*nrad && j<nsec){
    SGP_Kr[i*nsec + j].x = Kr_aux[i*nsec + j];
    SGP_Kr[i*nsec + j].y = 0.;

    SGP_Kt[i*nsec + j].x = Kt_aux[i*nsec + j];
    SGP_Kt[i*nsec + j].y = 0.;

    SGP_Sr[i*nsec + j].y = 0.;
    SGP_St[i*nsec + j].y = 0.;

    if (i<nrad){
      var = Dens[i*nsec + j] * sqrt(Rmed[i]/Rmed[0]);
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


__global__ void ComputeSgAccKernel (double *SG_Accr, double *SG_Acct, double *Dens , double SGP_rstep, double SGP_tstep,
  double SGP_eps, int nrad, int nsec, double *Rmed, cufftComplex *Gr, cufftComplex *Gphi)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = threadIdx.y + blockDim.y*blockIdx.y;

  double normaccr, normacct, divRmed;

  if (i<nrad && j<nsec){
    divRmed = Rmed[i]/Rmed[0];
    normaccr = SGP_rstep * SGP_tstep / ((double)(2*nrad) * (double)nsec);
    normacct = normaccr;
    normaccr /= sqrt(divRmed);
    normacct /= (divRmed * sqrt(divRmed));
    SG_Acct[i*nsec + j] = Gphi[i*nsec + j].x * normaccr;

    SG_Accr[i*nsec + j] = Gr[i*nsec + j].x * normaccr;
    SG_Accr[i*nsec + j] += G*Dens[i*nsec + j]*SGP_rstep*SGP_tstep / SGP_eps;
  }
}


__global__ void Update_sgvelocityKernel (double *Vradial, double *Vazimutal, double *SG_Accr, double *SG_Acct, double *Rinf,
  double *Rmed, double *invdiffRmed, double dt, int nrad, int nsec)
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


__global__ void Azimutalvelocity_withSGKernel (double *Vtheta, double *Rmed, double FLARINGINDEX, double SIGMASLOPE,
  double ASPECTRATIO, double *GLOBAL_bufarray, int nrad, int nsec)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = threadIdx.y + blockDim.y*blockIdx.y;

  double omegakep, omega, invr;
  if (i<nrad && j<nsec){
    invr = 1./Rmed[i];
    omegakep = sqrt(G*1.0*invr*invr*invr);
    omega = sqrt(omegakep*omegakep* (1.0 - (1.+SIGMASLOPE-2.0*FLARINGINDEX)*pow(ASPECTRATIO,2.0)* \
      pow(Rmed[i],2.0*FLARINGINDEX)) - invr*GLOBAL_bufarray[i]);

    Vtheta[i*nsec + j] = Rmed[i]*omega;
  }
}


__global__ void CrashKernel (double *array, int nrad, int nsec, int Crash)
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


__global__ void EvanescentBoundaryKernel(double *Rmed, double *Vrad, double *Vtheta, double *Energy, double *Dens,
  double *viscosity_array, double DRMIN, double DRMAX, int nrad, int nsec, double Tin,
  double Tout, double step, double SIGMASLOPE, double FLARINGINDEX, double *GLOBAL_bufarray, double OmegaFrame,
  double *SigmaMed, double *EnergyMed, int Adiabatic, int SelfGravity, double ASPECTRATIO, double TRANSITIONWIDTH,
  double TRANSITIONRADIUS, double TRANSITIONRATIO, double PhysicalTime, double PhysicalTimeInitial, double LAMBDADOUBLING)
{
    int j = threadIdx.x + blockDim.x*blockIdx.x;
    int i = threadIdx.y + blockDim.y*blockIdx.y;

    double damping, lambda, vtheta0, vrad0, energy0, dens0, AspectRatio;
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
          vtheta0 = sqrt(G*1.0/Rmed[i] * (1.0 - (1.0+SIGMASLOPE-2.0*FLARINGINDEX)*pow(AspectRatio,2.0) * \
          pow(Rmed[i],2.0*FLARINGINDEX)));
        }
        if (SelfGravity){
          AspectRatio = AspectRatioDevice(Rmed[i], ASPECTRATIO, TRANSITIONWIDTH, TRANSITIONRADIUS, TRANSITIONRATIO,
            PhysicalTime, PhysicalTimeInitial, LAMBDADOUBLING);
          vtheta0 = sqrt(G*1.0/Rmed[i] * (1.0 - (1.0+SIGMASLOPE-2.0*FLARINGINDEX)*pow(AspectRatio,2.0) * \
          pow(Rmed[i],2.0*FLARINGINDEX)) - Rmed[i]*GLOBAL_bufarray[i]);
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


__global__ void DivisePolarGridKernel (double *Qbase, double *DensInt, double *Work, int nrad, int nsec)
{
  int i = threadIdx.x + blockDim.x*blockIdx.x;
  int j = threadIdx.y + blockDim.y*blockIdx.y;

  if (i<=nsec && j<nrad)
    Work[i*nrad + j] = Qbase[i*nrad + j]/(DensInt[i*nrad + j] + 1e-20);
}


__global__ void VanLeerRadialKernel (double *Rinf, double *Rsup, double *QRStar, double *DensStar, double *Vrad,
  double *LostByDisk, int nsec, int nrad, double dt, int OpenInner, double *Qbase, double *invSurf)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = threadIdx.y + blockDim.y*blockIdx.y;

  double varq, dtheta;

  if (i<nrad && j<nsec){
    dtheta = 2.0*M_PI/double(nsec);
    varq = dt*dtheta*Rinf[i]*QRStar[i*nsec + j]* DensStar[i*nsec + j]*Vrad[i*nsec + j];
    varq -= dt*dtheta*Rsup[i]*QRStar[(i+1)*nsec + j]* DensStar[(i+1)*nsec + j]*Vrad[(i+1)*nsec + j];
    Qbase[i*nsec + j] += varq*invSurf[i];
    if ( i == 1 && j == 1)
    {
      //printf("Qbase %.15f\n", Qbase[i*nsec + j]);
      //printf("invSurf%.15f\n", invSurf[i]);
    }
    if (i==0 && OpenInner)
    {
      LostByDisk[j] = varq;
      //printf("OpenInner%d\n", OpenInner);
    }
  }
}


__global__ void VanLeerThetaKernel (double *Rsup, double *Rinf, double *Surf, double dt, int nrad, int nsec,
  int UniformTransport, int *NoSplitAdvection, double *QRStar, double *DensStar, double *Vtheta, double *Qbase)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = threadIdx.y + blockDim.y*blockIdx.y;

  double dxrad, invsurf, varq;

  if (i<nrad && j<nsec){
    if ((UniformTransport == NO) || (NoSplitAdvection[i] == NO)){
      dxrad = (Rsup[i]-Rinf[i])*dt;
      invsurf = 1.0/Surf[i];
      varq = dxrad*QRStar[i*nsec + j]*DensStar[i*nsec + j]*Vtheta[i*nsec + j];
      varq -= dxrad*QRStar[i*nsec + (j+1)%nsec]*DensStar[i*nsec + (j+1)%nsec]*Vtheta[i*nsec + (j+1)%nsec];
      Qbase[i*nsec + j] += varq*invsurf;
      /*if ( i == 1 && j == 1)
        printf("qbase %.15f\n",Qbase[i*nsec+j] );*/
    }
  }
}



__global__ void ComputeAverageThetaVelocitiesKernel(double *Vtheta, double *VMed, int nsec, int nrad)
{
  int i = threadIdx.x + blockDim.x*blockIdx.x;

  double moy = 0.0;
  if (i<nrad){
    for (int j = 0; j < nsec; j++)
      moy += Vtheta[i*nsec + j];

    VMed[i] = moy/(double)nsec;
    //if (i == 1) printf("VMed [i]%.15f\n", VMed[i]);
  }
}


__global__ void ComputeResidualsKernel (double *VthetaRes, double *VMed, int nsec, int nrad, double *Vtheta)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = threadIdx.y + blockDim.y*blockIdx.y;

  if (i<nrad && j<nsec)
    VthetaRes[i*nsec + j] = Vtheta[i*nsec + j]-VMed[i];
    //if ( i == 1 && j == 1) printf("VthetaRes%.15f\n", VthetaRes[i*nsec+j]);
}


__global__ void ComputeConstantResidualKernel (double *VMed, double *invRmed, long *Nshift, int *NoSplitAdvection,
  int nsec, int nrad, double dt, double *Vtheta, double *VthetaRes, double *Rmed, int FastTransport)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = threadIdx.y + blockDim.y*blockIdx.y;

  double maxfrac, Ntilde, Nround, invdt;
  long nitemp;

  if (i<nrad && j<nsec){
    if (FastTransport)
      maxfrac = 1.0;
    else
      maxfrac = 0.0;

    invdt = 1.0/dt;
    Ntilde = VMed[i]*invRmed[i]*dt*(double)nsec/2.0/M_PI;
    Nround = floor(Ntilde+0.5);
    nitemp = (long)Nround;
    Nshift[i] = (long)nitemp;

    Vtheta[i*nsec + j] = (Ntilde-Nround)*Rmed[i]*invdt*2.0*M_PI/(double)nsec;
  /*  if ( i == 1 && j == 1)
    {
      printf("Ntilde %.15f\n", Ntilde);
      printf("Nround %.15f\n", Nround);
      printf("Vtheta %.15f\n", Vtheta[i*nsec+j]);
    }*/
    if (maxfrac < 0.5){
      NoSplitAdvection[i] = YES;
      VthetaRes[i*nsec + j] += Vtheta[i*nsec + j];
      Vtheta[i*nsec + j] = 0.0;
    }
    else{
      NoSplitAdvection[i] = NO;
    }



    //if(i == 1 && j == 1) printf("Vtheta %.15f\n",Vtheta[i*nsec+ j] );
  }
}


__global__ void StarThetaKernel (double *Qbase, double *Rmed, double *Vazimutal, double *QStar, int nrad, int nsec,
  double *dq, double dt)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = threadIdx.y + blockDim.y*blockIdx.y;

  double dxtheta, ksi, invdxtheta, dqp, dqm;
  if (i<nrad && j<nsec){
    if (i<nrad){
      dxtheta = 2.0*M_PI/(double)nsec*Rmed[i];
      invdxtheta = 1.0/dxtheta;
    }
    dqm = (Qbase[i*nsec + j] - Qbase[i*nsec + ((j-1)+nsec)%nsec]);
    dqp = (Qbase[i*nsec + (j+1)%nsec] - Qbase[i*nsec + j]);

    if (dqp * dqm > 0.0)
      dq[i*nsec + j] = dqp*dqm/(dqp+dqm)*invdxtheta;
    else
      dq[i*nsec + j] = 0.0;
    __syncthreads();

    ksi = Vazimutal[i*nsec + j]*dt;

    if (ksi > 0.0)
      QStar[i*nsec + j] = Qbase[i*nsec + ((j-1)+nsec)%nsec]+(dxtheta-ksi)*dq[i*nsec + ((j-1)+nsec)%nsec];
    else
      QStar[i*nsec + j] = Qbase[i*nsec + j]-(dxtheta+ksi)*dq[i*nsec + j];
   }
}


__global__ void AdvectSHIFTKernel (double *array, double *TempShift, int nsec, int nrad, long *Nshift)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = threadIdx.y + blockDim.y*blockIdx.y;

  int ji, modji;

  if (i<nrad && j<nsec){
    ji = j-Nshift[i];
    while (ji < 0 ) ji += nsec;
    while (ji >= nsec) ji -= nsec;//modji = ji%nsec;

    TempShift[i*nsec + j] = array[i*nsec + ji];
    __syncthreads();

    array[i*nsec + j] = TempShift[i*nsec + j];

    if (i == 50 && j == 300)
    {
      printf("shift %.20f\n", array[i*nsec+j]);
    }
  }
}


__global__ void ComputeVelocitiesKernel (double *Vrad, double *Vtheta, double *Dens, double *Rmed, double *ThetaMomP,
  double *ThetaMomM, double *RadMomP, double *RadMomM, int nrad, int nsec, double OmegaFrame)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = threadIdx.y + blockDim.y*blockIdx.y;

  if (i<nrad && j<nsec){
    if (i == 0)
      Vrad[i*nsec + j] = 0.0;
    else {
      Vrad[i*nsec + j] = (RadMomP[(i-1)*nsec + j] + RadMomM[i*nsec + j])/(Dens[i*nsec + j] +
        Dens[(i-1)*nsec + j] + 1e-20);
    }
    Vtheta[i*nsec + j] = (ThetaMomP[i*nsec + ((j-1)+nsec)%nsec] + ThetaMomM[i*nsec + j])/(Dens[i*nsec + j] +
      Dens[i*nsec + ((j-1)+nsec)%nsec] + 1e-15)/Rmed[i] - Rmed[i]*OmegaFrame;
      /* It was the angular momentum */


      // if ( i == 50 && j == 300)
      // {
      //   printf("ThetaMomP%.15f\n", ThetaMomP[i*nsec+j]);
      //   printf("ThetaMomM%.15f\n", ThetaMomM[i*nsec+j]);
      //   printf("RadMomP%.15f\n", RadMomP[i*nsec+j]);
      //   printf("RadMomM%.15f\n", RadMomM[i*nsec+j]);
      // }
  }
}


__global__ void ComputeSpeQtyKernel (double *Label, double *Dens, double *ExtLabel, int nrad, int nsec)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = threadIdx.y + blockDim.y*blockIdx.y;

  if (i<nrad && j<nsec){
    Label[i*nsec + j] = ExtLabel[i*nsec + j]/Dens[i*nsec + j];
    /* Compressive flow if line commentarized
    Label[i*nsec + j] = ExtLabel[i*nsec + j] */
  }
}


__global__ void FillForcesArraysKernel (double *Rmed, int nsec, int nrad, double xplanet, double yplanet, double smooth,
  double mplanet, int Indirect_Term, double InvPlanetDistance3, double *Potential, Pair IndirectTerm, int k)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = threadIdx.y + blockDim.y*blockIdx.y;

  double InvDistance, angle, x, y, distance, distancesmooth, pot;

  if (i<nrad && j<nsec){
    InvDistance = 1.0/Rmed[i];
    angle = (double)j/(double)nsec*2.0*M_PI;
    x = Rmed[i]*cos(angle);
    y = Rmed[i]*sin(angle);
    distance = (x-xplanet)*(x-xplanet)+(y-yplanet)*(y-yplanet);
    distancesmooth = sqrt(distance+smooth);
    pot = -G*mplanet/distancesmooth; /* Direct term from planet */
    if (Indirect_Term == YES)
      pot += G*mplanet*InvPlanetDistance3*(x*xplanet+y*yplanet); /* Indirect term from planet */
    Potential[i*nsec + j] += pot;
    if ( i == 2 && j == 0)
    {
      //printf("Potencialllll %f\n", Potential[i*nsec+j]);

    }

    __syncthreads();

    if (k == 0) {
     /* -- Gravitational potential from star on gas -- */
     pot = -G*1.0*InvDistance; /* Direct term from star */
     //pot -=  IndirectTerm.x*x + IndirectTerm.y*y; /* Indirect term from star */
     Potential[i*nsec + j] += pot;

     if ( i == 2 && j == 0)
     {
       printf("Potencialllll2 %.30f\n", Potential[i*nsec+j]);
     }

     if ( i == 1 && j == 0)
     {
       printf("Potencialllll1 %.30f\n", Potential[i*nsec+j]);
     }

     if ( i == 0 && j == 0)
     {
       printf("Potencialllll0 %.30f\n", Potential[i*nsec+j]);
     }

     if ( i == 3 && j == 0)
     {
       //printf("Potencialllll3 %f\n", Potential[i*nsec+j]);
     }

     if ( i == 4 && j == 0)
     {
       //printf("Potencialllll4 %f\n", Potential[i*nsec+j]);
     }

     if ( i == 5 && j == 0)
     {
       //printf("Potencialllll5 %f\n", Potential[i*nsec+j]);
     }
    }
  }
}


__global__ void CorrectVthetaKernel (double *Vtheta, double domega, double *Rmed, int nrad, int nsec)
{
    int j = threadIdx.x + blockDim.x*blockIdx.x;
    int i = threadIdx.y + blockDim.y*blockIdx.y;

    if (i<nrad && j<nsec)
      Vtheta[i*nsec + j] -= domega*Rmed[i];
}


__global__ void ConditionCFLKernel1D (double *Rsup, double *Rinf, double *Rmed, int nrad, int nsec,
  double *Vtheta, double *Vmoy)
{
  int i = threadIdx.x + blockDim.x*blockIdx.x;
  int j;

  if (i<nrad){
    Vmoy[i] = 0.0;

    for (j = 0; j < nsec; j++)
      Vmoy[i] += Vtheta[i*nsec + j];

    Vmoy[i] /= (double)nsec;
  }
}


__global__ void ConditionCFLKernel2D1 (double *Rsup, double *Rinf, double *Rmed, int nsec, int nrad,
  double *Vresidual, double *Vtheta, double *Vmoy, int FastTransport, double *SoundSpeed, double *Vrad,
  double CVNR, double *DT2D, double CFLSECURITY)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = threadIdx.y + blockDim.y*blockIdx.y;

  double dxrad, dxtheta, invdt1, invdt2, invdt3, invdt4, dvr, dvt, dt;

  if (i > 0 && i<nrad && j<nsec){
    dxrad = Rsup[i]-Rinf[i];
    dxtheta = Rmed[i]*2.0*M_PI/(double)nsec;
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
    if (dvt >= 0.0) dvt = 1e-10;
    else dvt = -dvt;
    invdt4 = max2(dvr/dxrad, dvt/dxtheta);
    invdt4*= 4.0*CVNR*CVNR;
    dt = CFLSECURITY/sqrt(invdt1*invdt1+invdt2*invdt2+invdt3*invdt3+invdt4*invdt4);
    DT2D[i*nsec + j] = dt; // array nrad*nsec size dt

    /*if ( i == 1 && j == 0)
    {
      printf("invdt1 %.10f\n", invdt1);
      printf("invdt2 %.10f\n", invdt2);
      printf("invdt3 %.10f\n", invdt3);
      printf("invdt4 %.10f\n", invdt4);
      printf("invdt4 %.10f\n", max2(dvr/dxrad, dvt/dxtheta));
      printf("dvr/dxrad %.10f\n", dvr/dxrad);
      printf("dvt/dxtheta %.10f\n", dvt/dxtheta);
    }*/

    //if ( dt == 0) printf("%d %d\n",i,j );
  }
}



__global__ void ConditionCFLKernel2D2 (double *newDT, double *DT2D, double *DT1D, double *Vmoy, double *invRmed,
  int *CFL, int nsec, int nrad, double CFLSECURITY, double DeltaT)
{
  int i = threadIdx.x + blockDim.x*blockIdx.x;
  int k;
  double dt;
  double newdt = 1e30;

  if (i>0 && i<nrad){
    newDT[i] = newdt;
    for (k = 0; k < nsec; k++)
      if (DT2D[i*nsec + k] < newDT[i]) newDT[i] = DT2D[i*nsec + k]; // for each dt in nrad
  }

  if (i<nrad-1){
    dt = 2.0*M_PI*CFLSECURITY/(double)nsec/fabs(Vmoy[i]*invRmed[i]-Vmoy[i+1]*invRmed[i+1]);
    DT1D[i] = dt; // array nrad size dt
    __syncthreads();

    if (i == 1){ // one thread
      newdt = newDT[1];
      for (k = 2; k < nrad; k++)
        if (newDT[k] < newdt) newdt = newDT[k]; // min dt

      //printf("el dt mejor %g\n", newdt);
      for (k = 0; k < nrad-1; k++) {
        if (DT1D[k] < newdt)
          newdt = DT1D[k];
      }

      if (DeltaT < newdt)
        newdt = DeltaT;
      //printf("DeltaT%.10f\n", DeltaT);
      //printf("newdt%.15f\n", newdt);
      CFL[0] = (int)(ceil(DeltaT/newdt));
    }
  }
}


__device__ double max2(double a, double b)
{
  if (b > a) return b;
  return a;
}


__device__ double min2(double a, double b)
{
  if (b < a) return b;
  return a;
}


__device__ double AspectRatioDevice(double r, double ASPECTRATIO, double TRANSITIONWIDTH, double TRANSITIONRADIUS,
  double TRANSITIONRATIO, double PhysicalTime, double PhysicalTimeInitial, double LAMBDADOUBLING)
{
  double aspectratio, rmin, rmax, scale;
  aspectratio = ASPECTRATIO;
  rmin = TRANSITIONRADIUS-TRANSITIONWIDTH*ASPECTRATIO;
  rmax = TRANSITIONRADIUS+TRANSITIONWIDTH*ASPECTRATIO;
  scale = 1.0+(PhysicalTime-PhysicalTimeInitial)*LAMBDADOUBLING;
  rmin *= scale;
  rmax *= scale;
  if (r < rmin) aspectratio *= TRANSITIONRATIO;
  if ((r >= rmin) && (r <= rmax)){
    aspectratio *= exp((rmax-r)/(rmax-rmin)*log(TRANSITIONRATIO));
  }
  return aspectratio;
}


__global__ void ApplySubKeplerianBoundaryKernel(double *VthetaInt, double *Rmed, double OmegaFrame, int nsec,
  int nrad, double VKepIn, double VKepOut)
{
  int j = threadIdx.x + blockDim.x*blockIdx.x;
  int i = 0;

  if (j<nsec)
    VthetaInt[i*nsec + j] = VKepIn - Rmed[i]*OmegaFrame;

  i = nrad - 1;

  if (j<nsec)
    VthetaInt[i*nsec + j] = VKepOut - Rmed[i]*OmegaFrame;

}
