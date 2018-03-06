#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <iomanip>
#include <numeric>
#include <stdlib.h>
#include <math.h>

using namespace std;

#define USETNS1


typedef float T;

typedef struct {
	unsigned int blocks;
	unsigned int blockSize;
	unsigned int Nx;
	unsigned int Ny;
	unsigned int size;
	unsigned int steps;
	unsigned int maxiter;
	T dt;
	T maxres;
	unsigned int nRowsZ;
} Parameters;


typedef struct {
	T rho;
	T cp;
	T k;
} MaterialProperties;

Parameters params;

MaterialProperties steel;
MaterialProperties Ag;
MaterialProperties MgO;
MaterialProperties inconel;
MaterialProperties NiCr;

#include "cpuFunctions.h"
#include "cudaFunctions.h"

int main(void)
{
    // Parameters
	params.blocks = 128;
	params.blockSize = 128;  
	params.Nx  = 1024;
	params.Ny  = 1024;
	params.size = params.Nx*params.Ny + 2*params.Nx;
	params.steps  = 1;
	params.maxiter  = 1000000;
	params.dt = 0.003333333333;    // 0.0033333333333333;
	params.maxres = 1e-3;
	params.nRowsZ = 32;    // number of rows for one deflation vector
	
	// steel
	steel.rho = 7610.0; // 7700
    steel.cp  = 545.0; // 560
    steel.k   = 21.0;

    // Ag
    Ag.rho = 8957.0;
    Ag.cp  = 362.0; // 368
    Ag.k   = 120; // 111.5

    // MgO
    MgO.rho = 3150.0;
    MgO.cp  = 1110.0; // 1140
    MgO.k   = 11.5;   // 10
    
    // inconel
    inconel.rho = 8470.0;
    inconel.cp  = 500.0; // 520
    inconel.k   = 20.5; 
    
    // NiCr
    NiCr.rho = 8200.0;
    NiCr.cp  = 528.0;
    NiCr.k   = 24.5;
    
    T t = 0; // time
    T totalIter = 0;
    
    
    dim3 dimGrid(params.blocks);
    dim3 dimBlock(params.blockSize);
	
	cout << "example 1: inverse task (Pohanka)" << endl;
	
	cpuInit(params.blocks, params.Nx, params.Ny);
	//cpuInitDeflation(params.nRowsZ, params.Nx, params.Ny);
	
	
	readGeometry(hm, params.Nx, params.Ny);	// materials
	
	readCoords(xc, dx, "xCoords1024.txt");
	readCoords(yc, dy, "yCoords1024.txt");
	
	readBC(tHF, params.steps);
	
	initX(hT, params.Nx, params.Ny);
	initA(params.dt, params.Nx, params.Ny);
	
	//initAZ(params.dt, params.nRowsZ, params.Nx, params.Ny);
	//initE(params.nRowsZ, params.Nx, params.Ny);
	
	//saveData<int>(hm, "materials1024", params.Nx, params.Ny);
	//saveDataInTime(hT, t, "temperature1024_res1e-3_TNS1", params.Nx, params.Ny);
	
	//check_var(dy, params.Nx, params.Ny);

	// CUDA
	cudaInit(hT, hV, hcc, hss, hww, hqB, params.blocks, params.Nx, params.Ny);
	
#ifdef USETNS1
	//makeTNS1<<<1024,1024>>>(kcc, kss, kww, dcc, dss, dww, params.Nx, params.Ny);
	makeTNS1f<<<1024,1024>>>(kcc, kss, kww, k3, dcc, dss, dww, params.Nx, params.Ny);
#endif	
	
	cudaEvent_t startT, stopT;
	float elapsedTime;
	cudaEventCreate(&startT);
	cudaEventCreate(&stopT);
	cudaEventRecord(startT,0);

	for (int miter=0; miter<params.steps; miter++) {

		cudaMemcpy(dr, dT, sizeof(T)*params.size, cudaMemcpyDeviceToDevice);	// r = rhs
		elementWiseMul<<<1024,1024>>>(dr, dV, params.Nx);	// r = V*r
		
		// add Neumann boundary here ... r = r + NeumannBC (dqB)
		//addNeumannBC<<<1,1024>>>(dr, dqB, (T)-1.0e6, params.Nx);   // constant
		addNeumannBC<<<1,1024>>>(dr, dqB, tHF[miter], params.Nx);   // time dependent


		SpMVv1<<<1024,1024>>>(dq, dcc, dss, dww, dT, params.Nx);   // q = Ax (version 1)

		AXPY<<<1024,1024>>>(dr, dq, (T)-1., (T)1., params.Nx);   // r = r - q

#ifdef USETNS1
		//SpMVv1<<<1024,1024>>>(dz, kcc, kss, kww, dr, params.Nx);  // z = M^(-1)r (version 1)
		SpMVv2<<<1024,1024>>>(dz, kcc, kss, k3, kww, dr, params.Nx);  // z = M^(-1)r (version 2)
		DOTGPU<T,128><<<dimGrid,dimBlock,params.blockSize*sizeof(T)>>>(drh, dr, dz, params.Nx, params.Ny);
#else
		DOTGPU<T,128><<<dimGrid,dimBlock,params.blockSize*sizeof(T)>>>(drh, dr, dr, params.Nx, params.Ny);
#endif
		cudaMemcpy(&rhNew, drh, 1*sizeof(T), cudaMemcpyDeviceToHost);
		cudaMemset(drh,0,sizeof(T)); // reset;
		stop = rhNew * params.maxres * params.maxres;
		iter = 0;

		//cout << "stop:" << stop << ", residual: " << rhNew << endl;
		
		while (rhNew > stop && iter < params.maxiter) {

			iter++;
			totalIter++;
			//cout << "iteration:" << iter << ", residual: " << rhNew << endl;

			if (iter==1) {
#ifdef USETNS1
				cudaMemcpy(dp, dz, sizeof(T)*params.size,cudaMemcpyDeviceToDevice);
#else
				cudaMemcpy(dp, dr, sizeof(T)*params.size,cudaMemcpyDeviceToDevice);
#endif
			}
			else {
				bt = rhNew/rhOld;
#ifdef USETNS1	
				AXPY<<<1024,1024>>>(dp, dz, (T)1., bt, params.Nx);   // p = z + beta*p	
#else
				AXPY<<<1024,1024>>>(dp, dr, (T)1., bt, params.Nx);   // p = r + beta*p	
#endif

			}


			SpMVv1<<<1024,1024>>>(dq, dcc, dss, dww, dp, params.Nx);  // q = Ap (version 1)
			DOTGPU<T,128><<<dimGrid,dimBlock,params.blockSize*sizeof(T)>>>(dsg, dp, dq, params.Nx, params.Ny);   // sigma = <p,q>
			cudaMemcpy(&sg, dsg, 1*sizeof(T), cudaMemcpyDeviceToHost);
			cudaMemset(dsg,0,sizeof(T)); // reset;
			ap = rhNew/sg;	// alpha = rhoNew / sigma
			AXPY<<<1024,1024>>>(dr, dq, -ap, (T)1., params.Nx);   // r = r - alpha*q
			AXPY<<<1024,1024>>>(dT, dp,  ap, (T)1., params.Nx);   // x = x + alpha*p

#ifdef USETNS1
			//SpMVv1<<<1024,1024>>>(dz, kcc, kss, kww, dr, params.Nx);  // z = M^(-1)r (version 1)
			SpMVv2<<<1024,1024>>>(dz, kcc, kss, k3, kww, dr, params.Nx);  // z = M^(-1)r (version 2)
#endif

			rhOld = rhNew;

#ifdef USETNS1
			DOTGPU<T,128><<<dimGrid,dimBlock,params.blockSize*sizeof(T)>>>(drh, dr, dz, params.Nx, params.Ny);   // rhoNew = <r,z>		
#else
			DOTGPU<T,128><<<dimGrid,dimBlock,params.blockSize*sizeof(T)>>>(drh, dr, dr, params.Nx, params.Ny);   // rhoNew = <r,r>
#endif
			cudaMemcpy(&rhNew, drh, 1*sizeof(T), cudaMemcpyDeviceToHost);
			cudaMemset(drh,0,sizeof(T)); // reset;	
			

		}
		t += params.dt;
		cout << "time: " << t << " ,timestep:" << miter << " ,iteration:" << iter << endl;
		
		//if ((miter+1)%4000==0)
		//{
		//cudaMemcpy(hT, dT, sizeof(T)*params.size, cudaMemcpyDeviceToHost);
		//saveDataInTime(hT, t, "temperature1024_res1e-3_TNS1float", params.Nx, params.Ny);
		//}

	}
	
	cudaEventRecord(stopT,0);
	cudaEventSynchronize(stopT);
	cudaEventElapsedTime(&elapsedTime, startT, stopT);
	cout<< "ellapsed time (cuda): " << elapsedTime << " miliseconds"	<< endl;
	
	cout << "Simulation finished." << endl;
	cout << "total number of iterations: " << totalIter << endl;
	
	cudaEventDestroy(startT);
	cudaEventDestroy(stopT);
	cudaFinalize();
	cpuFinalize();
	cpuFinalizeDeflation();
	return 0;
}

