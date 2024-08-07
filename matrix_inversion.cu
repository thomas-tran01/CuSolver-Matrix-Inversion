#include <iostream>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <iomanip> 

#define MATRIX_DIM 5000
#define ITER 10

void createMatrix(float* h_matrix)
{
    for(int i = 0; i < MATRIX_DIM * MATRIX_DIM; i++)
    {
        h_matrix[i] = (rand()) % 100;
    }
}

void setIdentityMatrix(float* matrix, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            matrix[i*size+j] = (i == j) ? 1.0f : 0.0f;
        }
    }
}


void printMatrix(const float* matrix) {
    for (int i = 0; i < MATRIX_DIM; ++i) {
        for (int j = 0; j < MATRIX_DIM; ++j) {
            std::cout << matrix[i * MATRIX_DIM + j] << " ";
        }
        std::cout << std::endl;
    }
}

int main()
{
    const int matrixSize = MATRIX_DIM * MATRIX_DIM * sizeof(float);
    float res[ITER];
	float *curr = res;
    for (int i = 0; i < ITER; ++i)
    {
        //Host
        float h_matrix[MATRIX_DIM*MATRIX_DIM];
        float h_invMatrix[MATRIX_DIM*MATRIX_DIM];
        createMatrix(h_matrix);
        //printMatrix(h_matrix);
        
        // Device
        float* d_matrix;
        float* d_invMatrix;
        float* d_identity;
        int* d_info;
        float* d_work;
        int* d_ipiv;
        int workspace_size = 0;
        cudaMalloc(&d_matrix, matrixSize);
        cudaMalloc(&d_invMatrix, matrixSize);
        cudaMalloc(&d_identity, matrixSize);
        cudaMalloc(&d_info, sizeof(int));
        cudaMalloc(&d_work, workspace_size * sizeof(float));
        cudaMalloc(&d_ipiv, MATRIX_DIM * sizeof(int));
        cudaMemcpy(d_matrix, h_matrix, matrixSize, cudaMemcpyHostToDevice);


        cudaEvent_t start, stop;
        float computation_time;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start,0);
        

        cusolverDnHandle_t cussolverHandle;
        cusolverDnCreate(&cussolverHandle);

        cusolverDnSgetrf_bufferSize(cussolverHandle, MATRIX_DIM, MATRIX_DIM, d_matrix, MATRIX_DIM, &workspace_size);

        // LU factorization
        cusolverDnSgetrf(cussolverHandle, MATRIX_DIM, MATRIX_DIM, d_matrix, MATRIX_DIM, d_work, d_ipiv, d_info);
        cudaDeviceSynchronize();

        // Identity matrix
        float h_identity[MATRIX_DIM*MATRIX_DIM];
        setIdentityMatrix(h_identity, MATRIX_DIM);

        cudaMemcpy(d_identity, h_identity, matrixSize, cudaMemcpyHostToDevice);

        // Invert the matrix
        cusolverDnSgetrs(cussolverHandle, CUBLAS_OP_N, MATRIX_DIM, MATRIX_DIM, d_matrix, MATRIX_DIM, d_ipiv, d_identity, MATRIX_DIM, d_info);
        cudaDeviceSynchronize();
        cudaMemcpy(h_invMatrix, d_identity, matrixSize, cudaMemcpyDeviceToHost);

        cudaEventRecord(stop,0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&computation_time, start, stop);

        *curr = computation_time;
        ++curr;

        std::cout<< std::fixed << std::setprecision(5);
        std::cout<< "Computation took " << computation_time << "ms" << std::endl;
        //std::cout<< "Inverted Matrix:"<< std::endl;
        //std::cout<< h_invMatrix << std::endl;
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

    float total = 0.0;

	for (int x = 0; x < ITER; ++x)
	{
		total += res[x];
	}
	std::cout << "avg Cuda Time: " << total/ITER << "ms\n";\

}