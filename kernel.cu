#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <string.h>
#include <cuda.h>
#include "math.h"
#include "time.h"
#include <iostream>
#include <fstream>
#include <iomanip>

#define BLOCK_SIZE 32
__global__ void kernel_global(float* a, float* b, int n, float* c)
{
	int bx = blockIdx.x; // number of block x
	int by = blockIdx.y; // number of block y
	int tx = threadIdx.x; // number of thread  in block x
	int ty = threadIdx.y; // number of thread  in block y
	float sum = 0.0f;
	int ia = n * (BLOCK_SIZE * by + ty); // row number
	int ib = BLOCK_SIZE * bx + tx; // column number
	int ic = ia + ib; // number of element
	// calculating a matrix element
	for (int k = 0; k < n; k++) sum += a[ia + k] * b[ib + k * n];
	c[ic] = sum;
}
int main()
{
	int N = 1024;
	int m, n, k;
	// creating  events
	float timerValueGPU, timerValueCPU;
	cudaEvent_t start, stop;
	cudaEventCreate(&start); cudaEventCreate(&stop);
	int numBytes = N * N * sizeof(float);

	float* adev, * bdev, * cdev, * a, * b, * bT, * c, * cc;
	// memory allocation on host
	a = (float*)malloc(numBytes);
	b = (float*)malloc(numBytes); //matrix B
	bT = (float*)malloc(numBytes); //transposed matrix B
	c = (float*)malloc(numBytes); //matrix C for GPU
	cc = (float*)malloc(numBytes); //matrix C for CPU


	// setting matrix A, B and transposed matrix B
	for (n = 0; n < N; n++)
	{
		for (m = 0; m < N; m++)
		{
			a[m + n * N] = 2.0f * m + n; b[m + n * N] = m - n; bT[m + n * N] = n - m;
		}
	}
	// setting a mesh of threads and blocks
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocks(N / threads.x, N / threads.y);
	//GPU memory allocation
	cudaMalloc((void**)&adev, numBytes);
	cudaMalloc((void**)&bdev, numBytes);
	cudaMalloc((void**)&cdev, numBytes);
	// ---------------- GPU ------------------------
// copy matrices A and B from host to device
	cudaMemcpy(adev, a, numBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(bdev, b, numBytes, cudaMemcpyHostToDevice);
	// start timer
	cudaEventRecord(start, 0);
	// running kernel function
	kernel_global << <blocks, threads >> > (adev, bdev, N, cdev);
	// GPU computation time estimate
	cudaThreadSynchronize();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&timerValueGPU, start, stop);
	printf("\n GPU calculation time %f msec\n", timerValueGPU);
	// copy the computed matrix C from device to host
	cudaMemcpy(c, cdev, numBytes, cudaMemcpyDeviceToHost);



	// -------------------- CPU --------------------
	// start timer
	cudaEventRecord(start, 0);
	// computation of matrix C
	for (n = 0; n < N; n++)
	{
		for (m = 0; m < N; m++)
		{
			cc[m + n * N] = 0.f;
			for (k = 0; k < N; k++) cc[m + n * N] += a[k + n * N] * bT[k + m * N]; // bT !!!
		}
	}
	// CPU computation time estimate
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&timerValueCPU, start, stop);
	printf("\n CPU calculation time %f msec\n", timerValueCPU);
	//	printf("\n Rate %f x\n", timerValueCPU / timerValueGPU);

			// clean memory on GPU and CPU
	cudaFree(adev);
	cudaFree(bdev);
	cudaFree(cdev);
	cudaFreeHost(a);
	cudaFreeHost(b);
	cudaFreeHost(bT);
	cudaFreeHost(c);
	cudaFreeHost(cc);
	// destroy event
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	return 0;
}