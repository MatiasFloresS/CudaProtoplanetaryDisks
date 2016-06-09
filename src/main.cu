// nvcc hello-world.cu -L /usr/local/cuda/lib -lcudart -o hello-world

#include "main.cuh"
#include "kernels.cuh"

int blocksize = 32;
int nrad = 128;
int nsec = 384;
int size_grid = nrad*nsec;
float OmegaFrame = 0.12871;
bool ZMPlus = 0;
float IMPOSEDDISKDRIFT = 0.0;
float SIGMASLOPE = 0.0;


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


__host__ int main(int argc, char *argv[])
{
  float *press, *rho, *vradint, *invdiffRmed, *pot, *invRinf, *Rinf, *vrad, *vthetaint, *vtheta, *Rmed;
  float *press_d,*rho_d,*vradint_d,*invdiffRmed_d,*pot_d, *invRinf_d, *Rinf_d, *vrad_d, *vthetaint_d, *vtheta_d, *Rmed_d;
  int nrad2pot, nsec2pot;

  printf("%s\n",argv[0]);
  
  ReadFile();

  float dt = 0.999;
  press = (float *) malloc(sizeof(float)*size_grid);
  rho = (float *) malloc(sizeof(float)*size_grid );
  vradint = (float *) malloc(sizeof(float)*size_grid);
  pot = (float *) malloc(sizeof(float)*size_grid);
  vrad = (float *) malloc(sizeof(float)*size_grid);
  vthetaint = (float *) malloc(sizeof(float)*size_grid);
  vtheta = (float *) malloc(sizeof(float)*size_grid);
  invdiffRmed = (float *) malloc(sizeof(float)*nrad);
  invRinf = (float *) malloc(sizeof(float)*nrad);
  Rinf = (float *) malloc(sizeof(float)*nrad);
  Rmed = (float *) malloc(sizeof(float)*nrad);

  for (int i  = 0; i < size_grid; i++) {
    press[i] = i;
    rho[i] = 2*i;
    pot[i] = 0.001*i;
    vrad[i] = 0.212*i;
    vtheta[i] = 0.1;
    if (i < nrad) {
      invdiffRmed[i]= 0.002;
      Rinf[i] = 0.001;
      invRinf[i] = 1/Rinf[i];
      Rmed[i] = 1/invdiffRmed[i];
    }
  }

  if(!isPow2(nrad)) nrad2pot = NearestPowerOf2(nrad);
  if(!isPow2(nsec)) nsec2pot = NearestPowerOf2(nsec);

	cudaMalloc((void**)&press_d, size_grid*sizeof(float));
	cudaMalloc((void**)&rho_d, size_grid*sizeof(float) );
  cudaMalloc((void**)&vradint_d, size_grid*sizeof(float));
  cudaMalloc((void**)&pot_d, size_grid*sizeof(float));
  cudaMalloc((void**)&vrad_d, size_grid*sizeof(float));
  cudaMalloc((void**)&vthetaint_d, size_grid*sizeof(float));
  cudaMalloc((void**)&vtheta_d, size_grid*sizeof(float));
  cudaMalloc((void**)&invdiffRmed_d, nrad*sizeof(float));
  cudaMalloc((void**)&invRinf_d,nrad*sizeof(float));
  cudaMalloc((void**)&Rinf_d,nrad*sizeof(float));
  cudaMalloc((void**)&Rmed_d,nrad*sizeof(float));


	cudaMemcpy(press_d, press, size_grid*sizeof(float), cudaMemcpyHostToDevice );
	cudaMemcpy(rho_d, rho, size_grid*sizeof(float), cudaMemcpyHostToDevice );
  cudaMemcpy(vradint_d, vradint, size_grid*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(pot_d, pot, size_grid*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(vrad_d, vrad, size_grid*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(vthetaint_d, vthetaint, size_grid*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(vtheta_d, vtheta, size_grid*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(invdiffRmed_d, invdiffRmed, nrad*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(invRinf_d, invRinf, nrad*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(Rinf_d, Rinf, nrad*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(Rmed_d, Rmed, nrad*sizeof(float), cudaMemcpyHostToDevice);


	dim3 dimGrid( nsec2pot/blocksize, nrad2pot/blocksize );
	dim3 dimBlock( blocksize, blocksize );

	substep1<<<dimGrid, dimBlock>>>(press_d, rho_d, vradint_d, invdiffRmed_d,pot_d,Rinf_d,
    invRinf_d, vrad_d, vthetaint_d, vtheta_d, Rmed_d,  dt, nrad, nsec, OmegaFrame, ZMPlus,
    IMPOSEDDISKDRIFT, SIGMASLOPE);

	cudaMemcpy(vradint, vradint_d, size_grid*sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(vthetaint, vthetaint_d, size_grid*sizeof(float), cudaMemcpyDeviceToHost);


	cudaFree(press_d );
	cudaFree(rho_d );
  cudaFree(vradint_d);
  cudaFree(pot_d);
  cudaFree(invdiffRmed_d);
  cudaFree(invRinf_d);
  cudaFree(Rinf_d);
  cudaFree(vrad_d);
  cudaFree(vthetaint_d);
  cudaFree(vtheta_d);
  cudaFree(Rmed_d);

  /*if (SelfGravity){
    selfgravityupdate = YES;
    compute_selfgravity(Rho, VradInt, VthetaInt, dt, selfgravityupdate);
  }
  ComputeViscousTerms (VradInt, VthetaInt, Rho);*/
  printf("%d\n", nsec);
  UpdateVelocitiesWithViscosity(vradint, vthetaint, rho, dt);
/*
  if (!Evanescent) ApplySubKeplerianBoundary(VthetaInt);
*/
  FILE *f;
  f = fopen("datos.txt","w");

  for (int i = 0; i < size_grid; i++)
  {
    fprintf(f, "%f\n",vthetaint[i] );
  }

  fclose(f);
	return EXIT_SUCCESS;
}
