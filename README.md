# EXP-6   Matrix-multiplication-using-cuBLAS-in-CUDA-C-
<h3>NAME:PRIYANKA A</h3> 
<h3>REGISTER NO:212222230113</h3> 
<h3>EX. NO:06</h3> 
<h3>DATE:13-11-2025</h3>

## Objective
To implement matrix multiplication on the GPU using the cuBLAS library in CUDA C, and analyze the performance improvement over CPU-based matrix multiplication by leveraging GPU acceleration.

# AIM:
To utilize the cuBLAS library for performing matrix multiplication on NVIDIA GPUs, enhancing the performance of matrix operations by parallelizing computations and utilizing efficient GPU memory access.

Code Overview
In this experiment, you will work with the provided CUDA C code that performs matrix multiplication using the cuBLAS library. The code initializes two matrices (A and B) on the host, transfers them to the GPU device, and uses cuBLAS functions to compute the matrix product (C). The resulting matrix C is then transferred back to the host for verification and output.

# EQUIPMENTS REQUIRED:
Hardware:
PC with NVIDIA GPU
Google Colab with NVCC compiler
Software:
CUDA Toolkit (with cuBLAS library)
NVCC (NVIDIA CUDA Compiler)
Sample datasets for matrix multiplication (e.g., random matrices)

# PROCEDURE:
Tasks:
Initialize Host Memory:

Allocate memory for matrices A, B, and C on the host (CPU). Use random values for matrices A and B.
Allocate Device Memory:

Allocate corresponding memory on the GPU device for matrices A, B, and C using cudaMalloc().
Transfer the host matrices A and B to the GPU device using cudaMemcpy().
Matrix Multiplication using cuBLAS:

Initialize the cuBLAS library using cublasCreate().
Use the cublasSgemm() function to perform single-precision matrix multiplication on the GPU. This function computes the matrix product C = alpha * A * B + beta * C.
Retrieve and Print Results:

Copy the resulting matrix C from the device back to the host memory using cudaMemcpy().
Print the matrices A, B, and C to verify the correctness of the multiplication.
Clean Up Resources:

Free the allocated host and device memory using free() and cudaFree().
Shutdown the cuBLAS library using cublasDestroy().

Performance Analysis:
Measure the execution time of matrix multiplication using the cuBLAS library with different matrix sizes (e.g., 256x256, 512x512, 1024x1024).
Experiment with varying block sizes (e.g., 16, 32, 64 threads per block) and analyze their effect on execution time.
Compare the performance of the GPU-based matrix multiplication using cuBLAS with a standard CPU-based matrix multiplication implementation.
# PROGRAM:
```python
%%writefile matrix_mul.cu
#include <stdlib.h>
#include <stdio.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <math.h>
#include <time.h>

#define index(i,j,ld) (((j)*(ld))+(i))

void initializeMatrix(float *matrix, int size) {
    for(int i=0;i<size;i++)
        for(int j=0;j<size;j++)
            matrix[index(i,j,size)] = (float)(i+j)/size;
}

void cpuMatrixMultiplication(float *A, float *B, float *C, int n) {
    for(int i=0;i<n;i++)
        for(int j=0;j<n;j++){
            C[index(i,j,n)] = 0.0f;
            for(int k=0;k<n;k++)
                C[index(i,j,n)] += A[index(i,k,n)] * B[index(k,j,n)];
        }
}

int main(){
    int sizes[] = {256, 512, 1024};
    int numSizes = 3;

    for(int s=0;s<numSizes;s++){
        int size = sizes[s];
        printf("\nMatrix size: %d x %d\n", size, size);

        float *A = (float*)malloc(size*size*sizeof(float));
        float *B = (float*)malloc(size*size*sizeof(float));
        float *C_cpu = (float*)malloc(size*size*sizeof(float));
        float *C_gpu = (float*)malloc(size*size*sizeof(float));

        initializeMatrix(A, size);
        initializeMatrix(B, size);

        clock_t start_cpu = clock();
        cpuMatrixMultiplication(A,B,C_cpu,size);
        clock_t end_cpu = clock();
        double time_cpu = ((double)(end_cpu - start_cpu))/CLOCKS_PER_SEC;
        printf("CPU Matrix Multiplication: %f sec\n", time_cpu);

        float *d_A, *d_B, *d_C;
        cudaMalloc((void**)&d_A, size*size*sizeof(float));
        cudaMalloc((void**)&d_B, size*size*sizeof(float));
        cudaMalloc((void**)&d_C, size*size*sizeof(float));

        cudaMemcpy(d_A,A,size*size*sizeof(float),cudaMemcpyHostToDevice);
        cudaMemcpy(d_B,B,size*size*sizeof(float),cudaMemcpyHostToDevice);

        cublasHandle_t handle;
        cublasCreate(&handle);
        float alpha=1.0f, beta=0.0f;

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, size, size, size,
                    &alpha, d_B, size, d_A, size, &beta, d_C, size);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float time_gpu;
        cudaEventElapsedTime(&time_gpu, start, stop);
        printf("GPU Matrix Multiplication (cuBLAS): %f ms\n", time_gpu);

        cudaMemcpy(C_gpu, d_C, size*size*sizeof(float), cudaMemcpyDeviceToHost);

        int errors=0;
        float max_rel_err=1e-4;
        for(int i=0;i<size*size;i++){
            float rel_err=fabs(C_cpu[i]-C_gpu[i])/fmax(fabs(C_cpu[i]),fabs(C_gpu[i]));
            if(rel_err>max_rel_err) errors++;
        }

        if(errors==0) printf("Results verified successfully for size %d x %d\n", size, size);
        else printf("Discrepancies found in size %d x %d\n", size, size);

        cublasDestroy(handle);
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        free(A); free(B); free(C_cpu); free(C_gpu);
    }
    return 0;
}
```

# OUTPUT:
<img width="443" height="265" alt="image" src="https://github.com/user-attachments/assets/7b2357c1-c3ce-4f76-b2d5-f7cbcde5469e" />


# RESULT:

Thus, the matrix multiplication has been successfully implemented using the cuBLAS library in CUDA C, demonstrating the enhanced performance of GPU-based computation over CPU-based approaches.
