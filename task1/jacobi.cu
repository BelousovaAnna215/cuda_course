/* Jacobi-3 program */

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
using namespace std;

#define Max(a, b) ((a) > (b) ? (a) : (b))
#define ind(i, j, k) (((i)*L + (j))*L + (k))
#define L 384
#define ITMAX 100

double eps;
double MAXEPS = 0.5;

__global__ void functionKernel(const double *A, double *B) {
    int i = blockIdx.z * blockDim.z + threadIdx.z;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if (i > 0 && i < L - 1)
        if (j > 0 && j < L - 1)
            if (k > 0 && k < L - 1) {
                B[ind(i, j, k)] =
                    (A[ind(i - 1, j, k)] + A[ind(i, j - 1, k)] + A[ind(i, j, k - 1)] +
                     A[ind(i, j, k + 1)] + A[ind(i, j + 1, k)] + A[ind(i + 1, j, k)]) / 6.0;
            }
}

double compare_arrays(double *cpuArray, double *gpuArray) {
    double max_diff = 0;
    for (int i = 0; i < L * L * L; ++i) {
        double tmp = fabs(cpuArray[i] - gpuArray[i]);
        max_diff = Max(tmp, max_diff);
    }
    return max_diff;
}

void initial(double *A, double *B) {
    for (int i = 0; i < L; ++i)
        for (int j = 0; j < L; ++j)
            for (int k = 0; k < L; ++k) {
                A[ind(i, j, k)] = 0;
                if (i == 0 || j == 0 || k == 0 || i == L - 1 || j == L - 1 || k == L - 1)
                    B[ind(i, j, k)] = 0;
                else
                    B[ind(i, j, k)] = 4 + i + j + k;
            }
}

void print3DArray(double *A) {
    for (int i = 0; i < L; ++i) {
        std::cout << "Layer i = " << i << ":\n";
        for (int j = 0; j < L; ++j) {
            for (int k = 0; k < L; ++k) {
                std::cout << A[ind(i, j, k)] << " ";
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
    float time_taken_gpu = 0;

    size_t size = L * L * L;

    double *A = (double *)malloc(size * sizeof(double));
    double *B = (double *)malloc(size * sizeof(double));

    if (cpu || cpu_gpu) {
        initial(A, B);

        clock_t start_time = clock();

        for (int it = 1; it <= ITMAX; it++) {
            eps = 0;

            for (int i = 1; i < L - 1; i++)
                for (int j = 1; j < L - 1; j++)
                    for (int k = 1; k < L - 1; k++) {
                        double tmp = fabs(B[ind(i, j, k)] - A[ind(i, j, k)]);
                        eps = Max(tmp, eps);
                        A[ind(i, j, k)] = B[ind(i, j, k)];
                    }

            for (int i = 1; i < L - 1; i++)
                for (int j = 1; j < L - 1; j++)
                    for (int k = 1; k < L - 1; k++) {
                        B[ind(i, j, k)] =
                            (A[ind(i - 1, j, k)] + A[ind(i, j - 1, k)] + A[ind(i, j, k - 1)] +
                             A[ind(i, j, k + 1)] + A[ind(i, j + 1, k)] + A[ind(i + 1, j, k)]) /
                            6.0;
                    }

            if (eps < MAXEPS)
                break;
        }

        clock_t end_time = clock();
        time_taken_cpu = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

        printf(" CPU Completed.\n");
        printf(" CPU time = %12.2lf\n", time_taken_cpu);
    }

    thrust::host_vector<double> h_A(size), h_B(size);
    thrust::device_vector<double> d_A(size), d_B(size), diff_A_B(size);

    if (gpu || cpu_gpu) {
        initial(h_A.data(), h_B.data());

        thrust::copy(h_A.begin(), h_A.end(), d_A.begin());
        thrust::copy(h_B.begin(), h_B.end(), d_B.begin());

        dim3 thread(32, 4, 4);
        dim3 block((L + thread.x - 1) / thread.x,
                   (L + thread.y - 1) / thread.y,
                   (L + thread.z - 1) / thread.z);

        cudaEvent_t start_time, end_time;
        cudaEventCreate(&start_time);
        cudaEventCreate(&end_time);
        cudaEventRecord(start_time, 0);

        for (int it = 1; it <= ITMAX; it++) {
            thrust::transform(d_A.begin(), d_A.end(), d_B.begin(), diff_A_B.begin(),
                              thrust::minus<double>());

            double eps = thrust::transform_reduce(
                diff_A_B.begin(), diff_A_B.end(),
                [] __device__(double x) { return x < 0.0 ? -x : x; }, 0.0,
                thrust::maximum<double>());

            double *A_ptr = thrust::raw_pointer_cast(d_A.data());
            double *B_ptr = thrust::raw_pointer_cast(d_B.data());

            if (it % 2 == 1)
                functionKernel<<<block, thread>>>(B_ptr, A_ptr);
            else
                functionKernel<<<block, thread>>>(A_ptr, B_ptr);

            cudaStreamSynchronize(0);

            if (eps < MAXEPS)
                break;
        }

        cudaEventRecord(end_time, 0);
        cudaEventSynchronize(end_time);
        cudaEventElapsedTime(&time_taken_gpu, start_time, end_time);
        cudaEventDestroy(start_time);
        cudaEventDestroy(end_time);

        time_taken_gpu = time_taken_gpu * 0.001;

        h_A = d_A;
        h_B = d_B;

        printf(" GPU Completed.\n");
        printf(" Time in seconds = %12.2lf\n", time_taken_gpu);
    }

    if (cpu_gpu) {
        std::cout << "acceleration = " << time_taken_cpu / time_taken_gpu << std::endl;
        std::cout << "max difference = " << compare_arrays(A, h_A.data()) << std::endl;
    }

    free(A);
    free(B);

    return 0;
}
