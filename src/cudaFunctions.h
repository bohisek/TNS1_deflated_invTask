#ifndef CUDAFUNCTIONS_H_
#define CUDAFUNCTIONS_H_

// declare CUDA fields
T *dT , *dr , *dp , *dq , *dz;
T *dcc, *dss, *dww;  // matrix A center, south, west stencil
T *kcc, *kss, *kww;  // matrix M-1 center, south, west stencil
T *k3; // TNS1f
T *dpp;              // matrix P = sqrt(D)
T *drh, *dsg;        // partial dot products
T *dV;               // cell volume
T *dqB;              // Neumann BC bottom

// initialize CUDA fields
template <class T>
void cudaInit( T *hT,
		T *hV,
		T *hcc,
		T *hss,
		T *hww,
		T *hqB,
		const int blocks,
		const int Nx,
		const int Ny)
{
	cudaMalloc((void**)&dT ,sizeof(T)*(Nx*Ny+2*Nx));
	cudaMalloc((void**)&dr ,sizeof(T)*(Nx*Ny+2*Nx));
	cudaMalloc((void**)&dV ,sizeof(T)*(Nx*Ny+2*Nx));
	cudaMalloc((void**)&dp ,sizeof(T)*(Nx*Ny+2*Nx));
	cudaMalloc((void**)&dq ,sizeof(T)*(Nx*Ny+2*Nx));
	cudaMalloc((void**)&dz ,sizeof(T)*(Nx*Ny+2*Nx));
	cudaMalloc((void**)&dpp,sizeof(T)*(Nx*Ny+2*Nx));
	cudaMalloc((void**)&kcc,sizeof(T)*(Nx*Ny+2*Nx));
	cudaMalloc((void**)&kss,sizeof(T)*(Nx*Ny+Nx  ));
	cudaMalloc((void**)&k3 ,sizeof(T)*(Nx*Ny+Nx-1)); // TNS1f
	cudaMalloc((void**)&kww,sizeof(T)*(Nx*Ny+1   ));
	cudaMalloc((void**)&dcc,sizeof(T)*(Nx*Ny+2*Nx));
	cudaMalloc((void**)&dss,sizeof(T)*(Nx*Ny+Nx  ));
	cudaMalloc((void**)&dww,sizeof(T)*(Nx*Ny+1   ));
	cudaMalloc((void**)&drh,sizeof(T)*(1         ));
	cudaMalloc((void**)&dsg,sizeof(T)*(1         ));
	cudaMalloc((void**)&dqB,sizeof(T)*(Nx        ));

	cudaMemcpy(dT ,hT ,sizeof(T)*(Nx*Ny+2*Nx),cudaMemcpyHostToDevice);
	cudaMemcpy(dcc,hcc,sizeof(T)*(Nx*Ny+2*Nx),cudaMemcpyHostToDevice);
	cudaMemcpy(dV, hV ,sizeof(T)*(Nx*Ny+2*Nx),cudaMemcpyHostToDevice);
	cudaMemcpy(dss,hss,sizeof(T)*(Nx*Ny+Nx  ),cudaMemcpyHostToDevice);
	cudaMemcpy(dww,hww,sizeof(T)*(Nx*Ny+1   ),cudaMemcpyHostToDevice);
	cudaMemcpy(dqB,hqB,sizeof(T)*(Nx        ),cudaMemcpyHostToDevice);

	cudaMemset(dr ,0,sizeof(T)*(Nx*Ny+2*Nx));
	cudaMemset(dp ,0,sizeof(T)*(Nx*Ny+2*Nx));
	cudaMemset(dq ,0,sizeof(T)*(Nx*Ny+2*Nx));
	cudaMemset(dz ,0,sizeof(T)*(Nx*Ny+2*Nx));
	cudaMemset(dpp,0,sizeof(T)*(Nx*Ny+2*Nx));
	cudaMemset(kcc,0,sizeof(T)*(Nx*Ny+2*Nx));
	cudaMemset(kss,0,sizeof(T)*(Nx*Ny+Nx  ));
	cudaMemset(k3 ,0,sizeof(T)*(Nx*Ny+Nx-1));  // TNS1f
	cudaMemset(kww,0,sizeof(T)*(Nx*Ny+1   ));
	cudaMemset(drh,0,sizeof(T)*(1         ));
	cudaMemset(dsg,0,sizeof(T)*(1         ));

}

// destroy CUDA fields
void cudaFinalize()
{
	cudaFree(dT);
	cudaFree(dr);
	cudaFree(dp);
	cudaFree(dq);
	cudaFree(dz);
	cudaFree(dV);
	cudaFree(dpp);
	cudaFree(kcc);
	cudaFree(kss);
	cudaFree(k3);     // TNS1f
	cudaFree(kww);
	cudaFree(dcc);
	cudaFree(dss);
	cudaFree(dww);
	cudaFree(drh);
	cudaFree(dsg);
	cudaFree(dqB);
	
	cudaDeviceReset();
}


// AXPY (y := alpha*x + beta*y)
template <class T>
__global__ void AXPY(T *y,
		const T *x,
		const T alpha,
		const T beta,
		const int Nx)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x + Nx;
	y[tid] = alpha * x[tid] + beta * y[tid];
}


// SPMV (sparse matrix-vector multiplication)
template <class T>
__global__ void SpMVv1(T *y,
		const T *stC,
		const T *stS,
		const T *stW,
		const T *x,
		const int Nx)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x + Nx;
	
	y[tid]  = stC[tid]      * x[tid]     // center
	    	+ stS[tid]      * x[tid+Nx]  // north               N
	        + stW[tid-Nx+1] * x[tid+1]   // east              W C E
	        + stS[tid-Nx]   * x[tid-Nx]  // south               S
	        + stW[tid-Nx]   * x[tid-1];  // west
}

// SPMV (sparse matrix-vector multiplication)
template <class T>
__global__ void SpMVv2(T *y,
		const T *stC,
		const T *stS,
		const T *st3,
		const T *stW,
		const T *x,
		const int Nx)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x + Nx;

	y[tid]  = stC[tid]      * x[tid]       // center
	    	+ stS[tid]      * x[tid+Nx]    // north               N
	        + stW[tid-Nx+1] * x[tid+1]     // east              W C E
	        + stS[tid-Nx]   * x[tid-Nx]    // south               S
	        + stW[tid-Nx]   * x[tid-1]     // west
	        + st3[tid-Nx]   * x[tid-Nx+1]  // k3 on lhs
	        + st3[tid-1]    * x[tid+Nx-1]; // k3 on rhs
}





// DOT PRODUCT
template <class T, unsigned int blockSize>
__global__ void DOTGPU(T *c,
		const T *a,
		const T *b,
		const int Nx,
		const int Ny)
{
	extern __shared__ T cache[];

	unsigned int tid = threadIdx.x;
	unsigned int i = tid + blockIdx.x * (blockSize * 2);
	unsigned int gridSize = (blockSize*2)*gridDim.x;


	cache[tid] = 0;

	while(i<Nx*Ny) {
		cache[tid] += a[i+Nx] * b[i+Nx] + a[i+Nx+blockSize] * b[i+Nx+blockSize];
		i += gridSize;
	}

	__syncthreads();

	if(blockSize >= 512) {	if(tid < 256) { cache[tid] += cache[tid + 256]; } __syncthreads(); }
	if(blockSize >= 256) {	if(tid < 128) { cache[tid] += cache[tid + 128]; } __syncthreads(); }
	if(blockSize >= 128) {	if(tid < 64 ) { cache[tid] += cache[tid + 64 ]; } __syncthreads(); }

	if(tid < 32) {
		cache[tid] += cache[tid + 32];
		cache[tid] += cache[tid + 16];
		cache[tid] += cache[tid + 8];
		cache[tid] += cache[tid + 4];
		cache[tid] += cache[tid + 2];
		cache[tid] += cache[tid + 1];
	}

	if (tid == 0) atomicAdd(c, cache[0]);
}


//
template <class T>
__global__ void elementWiseMul(T *x,
		const T *p,
		const int Nx)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x + Nx;
	x[tid] *= p[tid];
}


// Truncated Neumann series 1
template <class T>
__global__ void makeTNS1(T *smC,
		T *smS,
		T *smW,
		const T *stC,
		const T *stS,
		const T *stW,
		const int Nx,
		const int Ny)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	T tstC1 = 1. / stC[tid+Nx];
	T tstC2 = 0.;
	T tstC3 = 0.;
	if (tid < Nx*Ny-Nx) tstC2 = 1. / stC[tid+2*Nx];
	if (tid < Nx*Ny-1)  tstC3 = 1. / stC[tid+Nx+1];

	smC[tid+Nx] = tstC1 * ( 1 + stS[tid+Nx] * stS[tid+Nx]  * tstC1 * tstC2  + stW[tid+1] * stW[tid+1]  * tstC1 * tstC3 );

	smS[tid+Nx] = -stS[tid+Nx] * tstC1 * tstC2;

	smW[tid+1]  = -stW[tid+1]  * tstC1 * tstC3;
}

// Truncated Neumann series 1 full
template <class T>
__global__ void makeTNS1f(T *smC,
		T *smS,
		T *smW,
		T *sm3,
		const T *stC,
		const T *stS,
		const T *stW,
		const int Nx,
		const int Ny)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	T tstC1 = 1. / stC[tid+Nx];
	T tstC2 = 0.;
	T tstC3 = 0.;
	T tstC4 = 0.;
	if (tid < Nx*Ny-Nx)   tstC2 = 1. / stC[tid+2*Nx];
	if (tid < Nx*Ny-1)    tstC3 = 1. / stC[tid+Nx+1];
	if (tid < Nx*Ny-Nx+1) tstC4 = 1. / stC[tid+2*Nx-1];

	smC[tid+Nx] = tstC1 * ( 1 + stS[tid+Nx] * stS[tid+Nx]  * tstC1 * tstC2  + stW[tid+1] * stW[tid+1]  * tstC1 * tstC3 );

	smS[tid+Nx] = -stS[tid+Nx] * tstC1 * tstC2;

	smW[tid+1]  = -stW[tid+1]  * tstC1 * tstC3;

	sm3[tid+Nx-1] = (-stS[tid+Nx] * tstC1 * tstC2) * (-stW[tid+Nx] * tstC4);
}


// for thermal boundary condition
template <class T>
__global__ void addNeumannBC(T *x,
		const T *Q,
		const T HeatFlux,
		const int Nx)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	x[tid+Nx] += HeatFlux * Q[tid];
}

#endif /* CUDAFUNCTIONS_H_ */
