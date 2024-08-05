#include <iostream>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cusolverDn.h>

#define MATRIX_DIM 300

__global__ void initializeMatrix(float* d_matrix, int matrixDim, int seed)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
    int index = idy * matrixDim + idx;

    // Initialize matrix with random int from 0-100
    if (idx < matrixDim && idy < matrixDim)
    {
        curandState state;
        curand_init(seed, index, 0, &state);
        d_matrix[index] = static_cast<float>(curand(&state) % 100);
    }
}

__global__ void setIdentityMatrix(float* matrix, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n * n) {
        int row = idx / n;
        int col = idx % n;
        matrix[idx] = (row == col) ? 1.0f : 0.0f;
    }
}


void printMatrix(const float* matrix, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            std::cout << matrix[i * size + j] << " ";
        }
        std::cout << std::endl;
    }
}

int main()
{
    const int matrixSize = MATRIX_DIM * MATRIX_DIM * sizeof(float);

    //Host
    float h_matrix[MATRIX_DIM*MATRIX_DIM];
    float h_invMatrix[MATRIX_DIM*MATRIX_DIM];
    
    // Device
    float* d_matrix;
    float* d_invMatrix;
    int* d_info;
    cudaMalloc(&d_matrix, matrixSize);
    cudaMalloc(&d_invMatrix, matrixSize);
    cudaMalloc(&d_info, sizeof(int));

    int threadsPerBlock = 16; 
    dim3 blockSize(threadsPerBlock, threadsPerBlock);
    dim3 gridSize((MATRIX_DIM + threadsPerBlock - 1) / threadsPerBlock, 
                (MATRIX_DIM + threadsPerBlock - 1) / threadsPerBlock);


    cudaEvent_t start, stop;
    float computation_time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);

    initializeMatrix<<<gridSize, blockSize>>>(d_matrix, MATRIX_DIM, time(NULL));
    cudaMemcpy(h_matrix, d_matrix, matrixSize, cudaMemcpyDeviceToHost);

    std::cout << "Current matrix:" << std::endl;
    //printMatrix((float*)h_matrix, MATRIX_DIM);

    cusolverDnHandle_t cussolverHandle;
    cusolverDnCreate(&cussolverHandle);
    
    // Workspace size query
    int workspace_size = 0;
    cusolverDnSgetrf_bufferSize(cussolverHandle, MATRIX_DIM, MATRIX_DIM, d_matrix, MATRIX_DIM, &workspace_size);
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    std::cout << "GPU memory: " << free / (1024*1024) << "MB free of " << total / (1024*1024) << "MB total" << std::endl;
    std::cout << "Workspace size:" << workspace_size << std::endl;
    std::cout << "Workspace size: " << workspace_size * sizeof(float) / (1024*1024) << " MB" << std::endl;

    float* d_work;
    cudaMalloc(&d_work, workspace_size * sizeof(float));

    // LU factorization
    int* d_ipiv;
    cudaMalloc(&d_ipiv, MATRIX_DIM * sizeof(int));
    cusolverDnSgetrf(cussolverHandle, MATRIX_DIM, MATRIX_DIM, d_matrix, MATRIX_DIM, d_work, d_ipiv, d_info);

    // Copy LU matrix back to host
    cudaMemcpy(h_matrix, d_matrix, matrixSize, cudaMemcpyDeviceToHost);
    std::cout << "LU matrix:" << std::endl;
    //printMatrix((float*)h_matrix, MATRIX_DIM);

    int h_info[1];
    cudaMemcpy(h_info,d_info,1*sizeof(int),cudaMemcpyDeviceToHost);

    for (int i = 0; i < 1; ++i)
    {
        std::cout << "h_info: " << i << " = " << h_info[i] << std::endl;
    }

    // Identity matrix
    float* d_identity;
    cudaMalloc(&d_identity, matrixSize);
    cudaMemset(d_identity, 0, matrixSize);
    int threadsPerBlock1D = 256;
    int numBlocks1D = (MATRIX_DIM * MATRIX_DIM + threadsPerBlock1D - 1) / threadsPerBlock1D;
    setIdentityMatrix<<<numBlocks1D, threadsPerBlock1D>>>(d_identity, MATRIX_DIM);

    // Invert the matrix
    cusolverDnSgetrs(cussolverHandle, CUBLAS_OP_N, MATRIX_DIM, MATRIX_DIM, d_matrix, MATRIX_DIM, d_ipiv, d_identity, MATRIX_DIM, d_info);

    // Copy inverted matrix back to host
    cudaMemcpy(h_invMatrix, d_identity, matrixSize, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&computation_time, start, stop);

    printf("Inverted Matrix:\n");
    //printMatrix(h_invMatrix, MATRIX_DIM);
    printf("Computation took %.10fms\n", computation_time);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);


    cudaFree(d_matrix);
    cudaFree(d_invMatrix);
    cudaFree(d_identity);
    cudaFree(d_info);
    cudaFree(d_work);
    cudaFree(d_ipiv);
    cusolverDnDestroy(cussolverHandle);

}