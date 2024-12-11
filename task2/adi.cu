#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <iostream>

using namespace std;

#define Max(a, b) ((a) > (b) ? (a) : (b))
#define ind(i, j, k) (((i)*ny + (j))*nz + (k))

#define nx 500
#define ny 500
#define nz 500

__global__ void kernel_step_1(double *A) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (j > 0 && j < ny - 1) {
        if (k > 0 && k < nz - 1) {
            for (int i = 1; i < nx - 1; i++) {
                A[ind(i, j, k)] = (A[ind(i - 1, j, k)] + A[ind(i + 1, j, k)]) / 2;
            }
        }
    }
}

__global__ void kernel_step_2(double *A) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 0 && i < nx - 1) {
        if (k > 0 && k < nz - 1) {
            for (int j = 1; j < ny - 1; j++) {
                A[ind(i, j, k)] = (A[ind(i, j - 1, k)] + A[ind(i, j + 1, k)]) / 2;
            }
        }
    }
}

__global__ void transpose(const double *in, double *out) {
    const int dim = 32;
    const int rows = 8;

    __shared__ double tile[dim][dim];

    int k = blockIdx.x * dim + threadIdx.x;
    int j = blockIdx.y;
    int i = blockIdx.z * dim + threadIdx.y;

    for (int n = 0; n < dim; n += rows) {
        int index = ind(i + n, j, k);
        tile[threadIdx.y + n][threadIdx.x] = in[index];
    }
    __syncthreads();

    k = blockIdx.z * dim + threadIdx.x;
    i = blockIdx.x * dim + threadIdx.y;

    for (int n = 0; n < dim; n += rows) {
        out[ind(i + n, j, k)] = tile[threadIdx.x][threadIdx.y + n];
    }
}

void init(double (*a)[ny][nz]) {
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            for (int k = 0; k < nz; k++) {
                if (k == 0 || k == nz - 1 || j == 0 || j == ny - 1 || i == 0 || i == nx - 1) {
                    a[i][j][k] = 10.0 * i / (nx - 1) + 10.0 * j / (ny - 1) + 10.0 * k / (nz - 1);
                } else {
                    a[i][j][k] = 0;
                }
            }
        }
    }
}

void init(thrust::host_vector<double> &A) {
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

double compare_arrays(double (*cpuArray)[ny][nz], thrust::host_vector<double> &gpuArray) {
    double max_diff = 0;
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            for (int k = 0; k < nz; ++k) {
                double tmp = fabs(cpuArray[i][j][k] - gpuArray[ind(i, j, k)]);
                max_diff = Max(tmp, max_diff);
            }
        }
    }
    return max_diff;
}

int main(int argc, char *argv[]) {
    int i, j, k;
    double (*A)[ny][nz];
    double maxeps = 0.01;
    int itmax = 100;

    bool cpu = false;
    bool gpu = false;
    bool cpu_gpu = false;

    if (argc > 1) {
        if (strcmp(argv[1], "c") == 0) {
            cpu = true;
        } else if (strcmp(argv[1], "g") == 0) {
            gpu = true;
        } else if (strcmp(argv[1], "cg") == 0) {
            cpu_gpu = true;
        }
    }

    float time_taken_cpu = 0;
    float time_taken_gpu = 0;

    A = (double(*)[ny][nz])malloc(nx * ny * nz * sizeof(double));

    if (cpu || cpu_gpu) {
        init(A);

        clock_t start_time = clock();

        for (int it = 1; it <= itmax; it++) {
            double eps = 0;

            for (i = 1; i < nx - 1; i++) {
                for (j = 1; j < ny - 1; j++) {
                    for (k = 1; k < nz - 1; k++) {
                        A[i][j][k] = (A[i - 1][j][k] + A[i + 1][j][k]) / 2;
                    }
                }
            }

            for (i = 1; i < nx - 1; i++) {
                for (j = 1; j < ny - 1; j++) {
                    for (k = 1; k < nz - 1; k++) {
                        A[i][j][k] = (A[i][j - 1][k] + A[i][j + 1][k]) / 2;
                    }
                }
            }

            for (i = 1; i < nx - 1; i++) {
                for (j = 1; j < ny - 1; j++) {
                    for (k = 1; k < nz - 1; k++) {
                        double tmp1 = (A[i][j][k - 1] + A[i][j][k + 1]) / 2;
                        double tmp2 = fabs(A[i][j][k] - tmp1);
                        eps = Max(eps, tmp2);
                        A[i][j][k] = tmp1;
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

    if (gpu || cpu_gpu) {
        init(h_A);
        d_A = h_A;

        dim3 thread(32, 32);
        dim3 block_1((ny + thread.x - 1) / thread.x, (nz + thread.y - 1) / thread.y);
        dim3 block_2((nx + thread.x - 1) / thread.x, (nz + thread.y - 1) / thread.y);
        dim3 block_3((nx + thread.x - 1) / thread.x, (ny + thread.y - 1) / thread.y);

        dim3 thread2(32, 8, 1);
        dim3 block2(nz / 32, ny, nz / 32);

        cudaEvent_t start_time, end_time;
        cudaEventCreate(&start_time);
        cudaEventCreate(&end_time);
        cudaEventRecord(start_time, 0);

        for (int it = 1; it <= itmax; it++) {
            double *A_ptr = thrust::raw_pointer_cast(d_A.data());

            kernel_step_1<<<block_1, thread>>>(A_ptr);
            cudaDeviceSynchronize();

            kernel_step_2<<<block_2, thread>>>(A_ptr);
            cudaDeviceSynchronize();

            double *A_trans_ptr = thrust::raw_pointer_cast(A_trans.data());

            transpose<<<block2, thread2>>>(A_ptr, A_trans_ptr);
            cudaDeviceSynchronize();

            kernel_step_1<<<block_1, thread>>>(A_trans_ptr);
            cudaDeviceSynchronize();

            double *A2_ptr = thrust::raw_pointer_cast(d_A2.data());

            transpose<<<block2, thread2>>>(A_trans_ptr, A2_ptr);
            cudaDeviceSynchronize();

            thrust::transform(d_A.begin(), d_A.end(), d_A2.begin(), eps_diff.begin(), thrust::minus<double>());

            double eps = thrust::transform_reduce(eps_diff.begin(), eps_diff.end(),
                                                  [] __device__(double x) { return x < 0.0 ? -x : x; },
                                                  0.0, thrust::maximum<double>());

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

    if (cpu_gpu) {
        std::cout << "acceleration = " << time_taken_cpu / time_taken_gpu << std::endl;
        std::cout << "max difference = " << compare_arrays(A, h_A) << std::endl;
    }

    free(A);

    return 0;
}
