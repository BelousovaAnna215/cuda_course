/* Jacobi-3 program */

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>  
#include <time.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <iostream>
#include <cstring>

#define SAFE_CALL(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
} while (0)

#define Max(a, b) ((a) > (b) ? (a) : (b))

#define A(i,j,k) A[((i)*L+(j))*L+(k)]
#define B(i,j,k) B[((i)*L+(j))*L+(k)]
#define C(i,j,k) C[((i)*L+(j))*L+(k)]

#define L 887
#define ITMAX 100

double eps;
double MAXEPS = 0.5;

__global__ void initialKernel(double *A, double *B) {
    int i = blockIdx.z * blockDim.z + threadIdx.z;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < L && j < L && k < L) {
        A(i, j, k) = 0;
        if (i == 0 || j == 0 || k == 0 || i == L - 1 || j == L - 1 || k == L - 1) {
            B(i, j, k) = 0;
        } else {
            B(i, j, k) = 4 + i + j + k;
        }
    }
}

__global__ void differenceKernel(double *A, const double *B, double* C) {
    int i = blockIdx.z * blockDim.z + threadIdx.z;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
	
    if (i > 0 && i < L - 1)
	if (j > 0 && j < L - 1)
	    if (k > 0 && k < L - 1) {
                C(i,j,k) = fabs(B(i,j,k) - A(i,j,k));
                A(i,j,k) = B(i,j,k);
	    }
}

__global__ void functionKernel(const double *A, double *B) {
    int i = blockIdx.z * blockDim.z + threadIdx.z;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > 0 && i < L - 1)
	if (j > 0 && j < L - 1)
	    if (k > 0 && k < L - 1) {
		B(i,j,k) = (A(i - 1,j,k) + A(i,j - 1,k) + A(i,j,k - 1) + A(i,j,k + 1) + A(i,j + 1,k) + A(i + 1,j,k)) / 6.0f;
	    }
}

double compare_arrays(double *cpuArray, double *gpuArray) {
    double max_diff = 0;
    for (int i = 0; i < L*L*L; ++i) {
	double tmp = fabs(cpuArray[i] - gpuArray[i]);
	max_diff = Max(tmp, max_diff);
    }
    return max_diff;
}

void initial(double *A, double *B) {
    for (int i = 0; i < L; ++i)
	for (int j = 0; j < L; ++j)
	    for (int k = 0; k < L; ++k) {
		A(i,j,k) = 0;
		if (i == 0 || j == 0 || k == 0 || i == L - 1 || j == L - 1 || k == L - 1)
		    B(i,j,k) = 0;
		else
		    B(i,j,k) = 4 + i + j + k;
	    }
}

void print3DArray(double *A) {
    for (int i = 0; i < L; ++i) {
        std::cout << "Layer i = " << i << ":\n";
        for (int j = 0; j < L; ++j) {
            for (int k = 0; k < L; ++k) {
                std::cout << A(i,j,k) << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
}

int main(int an, char **as) {
    bool cpu = false;
    bool gpu = false;
    bool cpu_gpu = false;

    if (an > 1) {
	if (strcmp(as[1], "c") == 0) {
	    cpu = true;
	} else if (strcmp(as[1], "g") == 0) {
	    gpu = true;
	} else if (strcmp(as[1], "cg") == 0) {
	    cpu_gpu = true;
	}
    }
    
    float time_taken_cpu = 0;
    
    size_t size = L * L * L * sizeof(double);
    
    double *A = (double*)malloc(size);
    double *B = (double*)malloc(size);

    if (cpu || cpu_gpu) {
	initial(A, B);
	clock_t start_time = clock();

	for (int it = 1; it <= ITMAX; it++) {
	    eps = 0;
				
	    for (int i = 1; i < L - 1; i++)
	        for (int j = 1; j < L - 1; j++)
		    for (int k = 1; k < L - 1; k++) {
			double tmp = fabs(B(i,j,k) - A(i,j,k));
			eps = Max(tmp, eps);
			A(i,j,k) = B(i,j,k);
		    }
				
	    for (int i = 1; i < L - 1; i++)
		for (int j = 1; j < L - 1; j++)
		    for (int k = 1; k < L - 1; k++)
			B(i,j,k) = (A(i - 1,j,k) + A(i,j - 1,k) + A(i,j,k - 1) + A(i,j,k + 1) + A(i,j + 1,k) + A(i + 1,j,k)) / 6.0f;
				
	    // printf(" IT = %4i   EPS = %14.7E\n", it, eps);
	    if (eps < MAXEPS)
		break;
        }
			
	clock_t end_time = clock();
	time_taken_cpu = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

	printf(" CPU Completed.\n");
	printf(" CPU time = %12.2lf\n", time_taken_cpu);		
    }

    float time_taken_gpu = 0;
    double *gA = (double*)malloc(size);
    
    if (gpu || cpu_gpu) {
	double *d_A, *d_B;
	SAFE_CALL(cudaMalloc((void **)&d_A, size));
	SAFE_CALL(cudaMalloc((void **)&d_B, size));
		
	dim3 thread(16, 8, 8);
	dim3 block((L + thread.x - 1) / thread.x, 
		   (L + thread.y - 1) / thread.y, 
		   (L + thread.z - 1) / thread.z);
		
	initialKernel<<<block, thread>>>(d_A, d_B);
	SAFE_CALL(cudaGetLastError());	

	cudaEvent_t start_time, end_time;
        cudaEventCreate(&start_time);
        cudaEventCreate(&end_time);
		
	cudaEventRecord(start_time, 0);
		
	for (int it = 1; it <= ITMAX; it++) {
	    thrust::device_vector<double> diff(L*L*L);
	    double *ptrdiff = thrust::raw_pointer_cast(&diff[0]);
	    differenceKernel<<<block, thread>>>(d_A, d_B, ptrdiff);
	    SAFE_CALL(cudaGetLastError());
			
	    double eps = thrust::reduce(diff.begin(), diff.end(), 0.0, thrust::maximum<double>());
			
	    functionKernel<<<block, thread>>>(d_A, d_B);
	    SAFE_CALL(cudaGetLastError());
		   
	    // printf(" IT = %4i   EPS = %14.7E\n", it, eps);
	    if (eps < MAXEPS)
		break;
	}
		
	cudaEventRecord(end_time, 0);
	cudaEventSynchronize(end_time);
        cudaEventElapsedTime(&time_taken_gpu, start_time, end_time);
        cudaEventDestroy(start_time);
        cudaEventDestroy(end_time);
		
	time_taken_gpu = time_taken_gpu * 0.001;
		
	SAFE_CALL(cudaMemcpy(gA, d_A, size, cudaMemcpyDeviceToHost));

        SAFE_CALL(cudaFree(d_A));
        SAFE_CALL(cudaFree(d_B));

	printf(" GPU Completed.\n");
	printf(" Time in seconds = %12.2lf\n", time_taken_gpu);
    }

    if (cpu_gpu) {
        std::cout << "acceleration = " << time_taken_cpu / time_taken_gpu <<  std::endl;
        
        // print3DArray(A);
        // print3DArray(gA);
        std::cout << "max difference = " << compare_arrays(A, gA) << std::endl;
        // if max difference == 0 - same results
    }
    
    free(A);
    free(gA);
    free(B);

    return 0;
}
