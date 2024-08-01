#include <iostream>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>


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
    const int matrixDim = 9;
    const int matrixSize = matrixDim * matrixDim * sizeof(float);
    float h_matrix[matrixDim][matrixDim];
    float h_invMatrix[matrixDim][matrixDim];
    
    float* d_matrix;
    cudaMalloc(&d_matrix, matrixSize);

    dim3 threadsPerBlock(matrixDim, matrixDim);
    dim3 numBlocks(1, 1);
    initializeMatrix<<<numBlocks, threadsPerBlock>>>(d_matrix, matrixDim, time(NULL));
    cudaMemcpy(h_matrix, d_matrix, matrixSize, cudaMemcpyDeviceToHost);

    std::cout << "Current matrix:" << std::endl;
    printMatrix((float*)h_matrix, matrixDim);
}