#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <algorithm>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <iostream>

using namespace std;

#define ind(i, j, k) ((i) * ny * nz + (j) * nz + (k))

const int TILE_SIZE = 32;
const int ROWS = 8;

//#define nx 500
//#define ny 500
//#define nz 500

__global__ void kernel_step_1(double *A, int nx, int ny, int nz) {
  int k = blockIdx.y * blockDim.y + threadIdx.x;
  int j = blockIdx.x * blockDim.x + threadIdx.y;

  if (j > 0 && j < ny - 1) {
    if (k > 0 && k < nz - 1) {
      for (int i = 1; i < nx - 1; i++) {
        A[ind(i, j, k)] = (A[ind(i - 1, j, k)] + A[ind(i + 1, j, k)]) / 2;
      }
    }
  }
}

__global__ void kernel_step_2(double *A, int nx, int ny, int nz) {
  int k = blockIdx.y * blockDim.y + threadIdx.x;
  int i = blockIdx.x * blockDim.x + threadIdx.y;

  if (i > 0 && i < nx - 1) {
    if (k > 0 && k < nz - 1) {
      for (int j = 1; j < ny - 1; j++) {
        A[ind(i, j, k)] = (A[ind(i, j - 1, k)] + A[ind(i, j + 1, k)]) / 2;
      }
    }
  }
}

__global__ void transpose(const double *in, double *out, int nx, int ny, int nz) {
  const int dim = 32;
  const int rows = 8;

  __shared__ double tile[dim + 1][dim + 1];

  int k = blockIdx.z * dim + threadIdx.x;
  int j = blockIdx.y;
  int i = blockIdx.x * dim + threadIdx.y;

  if (k < nz) {
    for (int n = 0; n < dim; n += rows) {
      int row = i + n;

      if (row >= nx) 
        break;

      int linear_index = ind(row, j, k);
      tile[threadIdx.y + n][threadIdx.x] = in[linear_index];
      
    }
  }
  __syncthreads();

  int k_t = blockIdx.x * dim + threadIdx.x;
  int i_t = blockIdx.z * dim + threadIdx.y;

  if (k_t < nx) {
    for (int n = 0; n < dim; n += rows) {
      int row = i_t + n;

      if (row >= nz) 
        break;
        
      out[(i_t + n) * ny * nx + (j) * nx + (k_t)] = tile[threadIdx.x][threadIdx.y + n];
    }
  }
}

void init(double* a, int nx, int ny, int nz) {
  for (int i = 0; i < nx; i++) {
    for (int j = 0; j < ny; j++) {
      for (int k = 0; k < nz; k++) {
        if (k == 0 || k == nz - 1 || j == 0 || j == ny - 1 || i == 0 || i == nx - 1) {
          a[ind(i, j, k)] = 10.0 * i / (nx - 1) + 10.0 * j / (ny - 1) + 10.0 * k / (nz - 1);
        } else {
          a[ind(i, j, k)] = 0;
        }
      }
    }
  }
}

void init(thrust::host_vector<double> &A, int nx, int ny, int nz) {
  for (int i = 0; i < nx; i++) {
    for (int j = 0; j < ny; j++) {
      for (int k = 0; k < nz; k++) {
        if (k == 0 || k == nz - 1 || j == 0 || j == ny - 1 || i == 0 || i == nx - 1) {
          A[ind(i, j, k)] = 10.0 * i / (nx - 1) + 10.0 * j / (ny - 1) + 10.0 * k / (nz - 1);
        } else {
          A[ind(i, j, k)] = 0;
        }
      }
    }
  }
}

double compare_arrays(double *cpuArray, thrust::host_vector<double> &gpuArray, int nx, int ny, int nz) {
  double max_diff = 0;

  for (int i = 0; i < nx; ++i) {
    for (int j = 0; j < ny; ++j) {
      for (int k = 0; k < nz; ++k) {
        double tmp = std::abs(cpuArray[ind(i, j, k)] - gpuArray[ind(i, j, k)]);
        max_diff = std::max(tmp, max_diff);
      }
    }
  }
  return max_diff;
}


int main(int argc, char *argv[]) {
    const double maxeps = 0.01;
    const int itmax = 100;

    bool cpu = false;
    bool gpu = false;
    
    int nx, ny, nz;
    
    float time_taken_cpu = 0;
    float time_taken_gpu = 0;
    
    try {
	  nx = std::atoi(argv[2]);
	  ny = std::atoi(argv[3]);
	  nz = std::atoi(argv[4]);
	
	  if (strcmp(argv[1], "c") == 0) {
		cpu = true;
	  } else if (strcmp(argv[1], "g") == 0) {
		gpu = true;
	  } else if (strcmp(argv[1], "cg") == 0) {
		cpu = true;
		gpu = true;
	  }
    } catch (...) {
	std::cout << "format is ./adi MODE NX NY NZ" << std::endl;
	exit(1);
    };

    double* A = new double[nx * ny * nz];

    if (cpu) {
        init(A, nx, ny, nz);

        clock_t start_time = clock();

        for (int it = 1; it <= itmax; it++) {
            double eps = 0;

            for (int i = 1; i < nx - 1; i++) {
                for (int j = 1; j < ny - 1; j++) {
                    for (int k = 1; k < nz - 1; k++) {
                        A[ind(i, j, k)] = (A[ind(i - 1, j, k)] + A[ind(i + 1, j, k)]) / 2;
                    }
                }
            }

            for (int i = 1; i < nx - 1; i++) {
                for (int j = 1; j < ny - 1; j++) {
                    for (int k = 1; k < nz - 1; k++) {
                        A[ind(i, j, k)] = (A[ind(i, j - 1, k)] + A[ind(i, j + 1, k)]) / 2;
                    }
                }
            }

            for (int i = 1; i < nx - 1; i++) {
                for (int j = 1; j < ny - 1; j++) {
                    for (int k = 1; k < nz - 1; k++) {
                        double tmp1 = (A[ind(i, j, k - 1)] + A[ind(i, j, k + 1)]) / 2;
                        double tmp2 = std::abs(A[ind(i, j, k)] - tmp1);
                        eps = std::max(eps, tmp2);
                        A[ind(i, j, k)] = tmp1;
                    }
                }
            }

            if (eps < maxeps) {
                break;
            }
        }

        clock_t end_time = clock();
        time_taken_cpu = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

        printf(" CPU Completed.\n");
        printf(" CPU time = %12.2lf\n", time_taken_cpu);
        printf(" Size = %4d x %4d x %4d\n", nx, ny, nz);
    }

    thrust::host_vector<double> h_A(nx * ny * nz);
    thrust::device_vector<double> d_A(nx * ny * nz);
    thrust::device_vector<double> d_A2(nx * ny * nz);
    thrust::device_vector<double> A_trans(nx * ny * nz);
    thrust::device_vector<double> eps_diff(nx * ny * nz);

    if (gpu) {
        init(h_A, nx, ny, nz);
		
        d_A = h_A;
        
        
        dim3 block(TILE_SIZE, TILE_SIZE);
	dim3 grid_1((ny + block.x - 1) / block.x, (nz + block.y - 1) / block.y);
	dim3 grid_2((nx + block.x - 1) / block.x, (nz + block.y - 1) / block.y);

	dim3 bt(TILE_SIZE, ROWS);
	dim3 grid_transpose((nx + TILE_SIZE - 1) / TILE_SIZE, ny,
			    (nz + TILE_SIZE - 1) / TILE_SIZE);

	dim3 grid_1_in_transposed((ny + block.x - 1) / block.x,
				  (nx + block.y - 1) / block.y);

	dim3 grid_transpose_back((nz + TILE_SIZE - 1) / TILE_SIZE, ny,
			         (nx + TILE_SIZE - 1) / TILE_SIZE);

        cudaEvent_t start_time, end_time;
        cudaEventCreate(&start_time);
        cudaEventCreate(&end_time);
        cudaEventRecord(start_time, 0);

	double* A_ptr = thrust::raw_pointer_cast(d_A.data());
	double* A_trans_ptr = thrust::raw_pointer_cast(A_trans.data());
	double* A2_ptr = thrust::raw_pointer_cast(d_A2.data());
		
        for (int it = 1; it <= itmax; it++) {

            kernel_step_1<<<grid_1, block>>>(A_ptr, nx, ny, nz);
	    cudaDeviceSynchronize();

            kernel_step_2<<<grid_2, block>>>(A_ptr, nx, ny, nz);
	    cudaDeviceSynchronize();

	    transpose<<<grid_transpose, bt>>>(A_ptr, A_trans_ptr, nx, ny, nz);
	    cudaDeviceSynchronize();
	    kernel_step_1<<<grid_1_in_transposed, block>>>(A_trans_ptr, nz, ny, nx);
	    cudaDeviceSynchronize();
	    transpose<<<grid_transpose_back, bt>>>(A_trans_ptr, A2_ptr, nz, ny, nx);
	    cudaDeviceSynchronize();

	    thrust::transform(d_A.begin(), d_A.end(), d_A2.begin(), eps_diff.begin(), thrust::minus<double>());
            
            double eps = thrust::transform_reduce(eps_diff.begin(), eps_diff.end(),
						[] __device__(double x) { return x < 0.0 ? -x : x; }, 0.0,
						thrust::maximum<double>());

            d_A = d_A2;

            if (eps < maxeps) {
                break;
            }
        }

        cudaEventRecord(end_time, 0);
        cudaEventSynchronize(end_time);
        cudaEventElapsedTime(&time_taken_gpu, start_time, end_time);
        cudaEventDestroy(start_time);
        cudaEventDestroy(end_time);

        time_taken_gpu = time_taken_gpu * 0.001;
        h_A = d_A;

        printf(" GPU Completed.\n");
        printf(" Time in seconds = %12.2lf\n", time_taken_gpu);
        printf(" Size = %4d x %4d x %4d\n", nx, ny, nz);
    }

    if (cpu && gpu) {
        std::cout << "acceleration = " << time_taken_cpu / time_taken_gpu << std::endl;
        std::cout << "max difference = " << compare_arrays(A, h_A, nx, ny, nz) << std::endl;
    }

    delete[] A;

    return 0;
}
