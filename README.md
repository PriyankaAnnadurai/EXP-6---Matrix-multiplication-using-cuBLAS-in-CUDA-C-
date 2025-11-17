# EXP-6   Matrix-multiplication-using-cuBLAS-in-CUDA-C-
<h3>NAME:PRIYANKA A</h3> 
<h3>REGISTER NO:212222230113</h3> 
<h3>EX. NO:06</h3> 
<h3>DATE:13-11-2025</h3>

## Objective
To implement matrix multiplication on the GPU using the cuBLAS library in CUDA C, and analyze the performance improvement over CPU-based matrix multiplication by leveraging GPU acceleration.

## AIM:
To utilize the cuBLAS library for performing matrix multiplication on NVIDIA GPUs, enhancing the performance of matrix operations by parallelizing computations and utilizing efficient GPU memory access.

Code Overview
In this experiment, you will work with the provided CUDA C code that performs matrix multiplication using the cuBLAS library. The code initializes two matrices (A and B) on the host, transfers them to the GPU device, and uses cuBLAS functions to compute the matrix product (C). The resulting matrix C is then transferred back to the host for verification and output.

## EQUIPMENTS REQUIRED:
Hardware:
PC with NVIDIA GPU
Google Colab with NVCC compiler
Software:
CUDA Toolkit (with cuBLAS library)
NVCC (NVIDIA CUDA Compiler)
Sample datasets for matrix multiplication (e.g., random matrices)

## PROCEDURE:
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
## PROGRAM:
```py
code = """
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define MATRIX_SIZE 1024

void fillMatrix(float *matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = (float)(rand() % 100) / 10.0f;
    }
}

void printMatrix(float *matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%0.2f ", matrix[i * cols + j]);
        }
        printf("\\n");
    }
    printf("\\n");
}

int main() {
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;
    int size = MATRIX_SIZE * MATRIX_SIZE * sizeof(float);

    h_A = (float *)malloc(size);
    h_B = (float *)malloc(size);
    h_C = (float *)malloc(size);

    fillMatrix(h_A, MATRIX_SIZE, MATRIX_SIZE);
    fillMatrix(h_B, MATRIX_SIZE, MATRIX_SIZE);

    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, MATRIX_SIZE, MATRIX_SIZE, MATRIX_SIZE, 
                &alpha, d_A, MATRIX_SIZE, d_B, MATRIX_SIZE, &beta, d_C, MATRIX_SIZE);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    printf("Matrix A:\\n");
    printMatrix(h_A, MATRIX_SIZE, MATRIX_SIZE);
    printf("Matrix B:\\n");
    printMatrix(h_B, MATRIX_SIZE, MATRIX_SIZE);
    printf("Matrix C (Result):\\n");
    printMatrix(h_C, MATRIX_SIZE, MATRIX_SIZE);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    cublasDestroy(handle);

    return 0;
}
"""
with open('matrix_mul.cu', 'w') as f:
    f.write(code)
```
```
!nvcc matrix_mul.cu -o matrix_mul -lcublas
!./matrix_mul
```

## OUTPUT:
![image](https://github.com/user-attachments/assets/a53cb317-9475-4bce-b538-091fad8a2cfc)


## RESULT:

Thus, the matrix multiplication has been successfully implemented using the cuBLAS library in CUDA C, demonstrating the enhanced performance of GPU-based computation over CPU-based approaches.
