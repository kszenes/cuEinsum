/*  
 * Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
 * 
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *  - Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  - Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  - Neither the name(s) of the copyright holder(s) nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR 
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */  

#include <stdlib.h>
#include <stdio.h>


#include <cuda_runtime.h>

#include <cublas_v2.h>

#include "utils.h"
#include "timer.h"
#include "rmse.h"


int main()
{
    const int runs = 1;
    // --- Single precision ---
    // typedef float floatTypeA;
    // typedef float floatTypeB;
    // typedef float floatTypeC;
    // typedef float floatTypeCompute;
    // cudaDataType_t typeA = CUDA_R_32F;
    // cudaDataType_t typeB = CUDA_R_32F;
    // cudaDataType_t typeC = CUDA_R_32F;
    // cublasComputeType_t cublasComputeType = CUBLAS_COMPUTE_32F_FAST_TF32;

    // cublasComputeType_t cublasComputeType = CUBLAS_COMPUTE_32F;

    // --- Double precision ---
    typedef double floatTypeA;
    typedef double floatTypeB;
    typedef double floatTypeC;
    typedef double floatTypeCompute;
    cudaDataType_t typeA = CUDA_R_64F;
    cudaDataType_t typeB = CUDA_R_64F;
    cudaDataType_t typeC = CUDA_R_64F;
    cublasComputeType_t cublasComputeType = CUBLAS_COMPUTE_64F;
    // --- END ---

    floatTypeCompute alpha = (floatTypeCompute)1.4f;
    floatTypeCompute beta  = (floatTypeCompute)0.0f;

    printf("Include headers and define data types\n");

    const int size = 1 << 13;
    const int i = size;
    const int j = size;
    const int k = size;

    // computes FLOPS
    double tflops = (2.0 * i * j * k) /1e12;

    /**********************
     * Allocating data
     **********************/

    size_t elementsA = i * j;
    size_t elementsB = j * k;
    size_t elementsC = i * k;

    size_t sizeA = sizeof(floatTypeA) * elementsA;
    size_t sizeB = sizeof(floatTypeB) * elementsB;
    size_t sizeC = sizeof(floatTypeC) * elementsC;
    printf("Total memory: %.2f GiB\n", (sizeA + sizeB + sizeC)/1024./1024./1024);

    floatTypeA *A_d, *B_d, *C_d;
    CUDA_CHECK(cudaMalloc((void**) &A_d, sizeA));
    CUDA_CHECK(cudaMalloc((void**) &B_d, sizeB));
    CUDA_CHECK(cudaMalloc((void**) &C_d, sizeC));


    floatTypeA *A = (floatTypeA*) malloc(sizeof(floatTypeA) * elementsA);
    printf("A malloc successful\n");
    floatTypeB *B = (floatTypeB*) malloc(sizeof(floatTypeB) * elementsB);
    printf("B malloc successful\n");
    floatTypeC *C = (floatTypeC*) malloc(sizeof(floatTypeC) * elementsC);
    printf("C malloc successful\n");

    if (A == NULL || B == NULL || C == NULL)
    {
        printf("Error: Host allocation of A or C.\n");
        return -1;
    }

    printf("Allocate, initialize and transfer tensors\n");
    /*******************
     * Initialize data
     *******************/

    for (size_t i = 0; i < elementsA; i++)
        A[i] = (((floatTypeA) rand())/RAND_MAX - 0.5)*100;
    for (size_t i = 0; i < elementsB; i++)
        B[i] = (((floatTypeB) rand())/RAND_MAX - 0.5)*100;
    for (size_t i = 0; i < elementsC; i++)
        C[i] = (((floatTypeC) rand())/RAND_MAX - 0.5)*100;

    CUDA_CHECK(cudaMemcpy(A_d, A, sizeA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d, B, sizeB, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(C_d, C, sizeC, cudaMemcpyHostToDevice));

    // floatTypeA *C_gemm;
    // CUDA_CHECK(cudaMalloc((void**) &C_gemm, sizeC));
    // CUDA_CHECK(cudaMemcpy(C_gemm, C, sizeC, cudaMemcpyHostToDevice));

    /*************
     * Compute DGEMM CUBLAS
     ************/
    cudaStream_t s;
    cublasHandle_t cublas_handle;
    CUDA_CHECK(cudaStreamCreate(&s));
    CUDA_CHECK(cublasCreate(&cublas_handle));

    // void* work_cublas;
    // size_t cublas_worksize = 4*1024^3; // 4 MiB
    // // size_t cublas_worksize = 0;

    // CUDA_CHECK(cudaMalloc(&work_cublas, cublas_worksize));
    // CUDA_CHECK(cublasSetWorkspace(cublas_handle, &work_cublas, cublas_worksize));

    double av_time_cublas = 0.0;
    double min_time_cublas = 1e8;
    // CUDA_CHECK(cublasDgemm(
    //             cublas_handle,
    //             CUBLAS_OP_N, CUBLAS_OP_N,
    //             i, j, k,
    //             &alpha,
    //             A_d, i,
    //             B_d, k,
    //             &beta,
    //             C_gemm, i));
    for (int iter = 0; iter < runs; iter++) {
        GPUTimer timer;
        CUDA_CHECK(cublasGemmEx(
                    cublas_handle, 
                    CUBLAS_OP_N, CUBLAS_OP_N, 
                    i, j, k, 
                    &alpha, 
                    A_d, typeA, i, 
                    B_d, typeB, k, 
                    &beta, 
                    C_d, typeC, i,
                    cublasComputeType,
                    CUBLAS_GEMM_DEFAULT)); // warmup
        timer.start();
        CUDA_CHECK(cublasGemmEx(
                    cublas_handle, 
                    CUBLAS_OP_N, CUBLAS_OP_N, 
                    i, j, k, 
                    &alpha, 
                    A_d, typeA, i, 
                    B_d, typeB, k, 
                    &beta, 
                    C_d, typeC, i,
                    cublasComputeType,
                    CUBLAS_GEMM_DEFAULT)); // warmup
        auto time  = timer.seconds();
        min_time_cublas = (time  < min_time_cublas) ? time : min_time_cublas;
        av_time_cublas += time / runs;
    }


    double transferedBytes = sizeC + sizeA + sizeB;
    transferedBytes += ((float) beta != 0.f) ? sizeC : 0;
    transferedBytes /= 1e9;
    
    // auto myRMSE = rmse(elementsC, C_gemm, C_d);

    printf("\nRESULTS from %d runs:\n", runs);
    // printf("RMSE: %f\n", myRMSE);
    printf("Time cuBLAS[s]: Best: %.4f; Mean %.4f\n",
            min_time_cublas, av_time_cublas);
    printf("Compute [GFLOPS/s]: Best: %.4f;  Mean %.4f\n",
            tflops / min_time_cublas, tflops/ av_time_cublas);
    printf("Memory [GB/s]: Best: %.2f;  Mean: %.2f\n",
            transferedBytes / min_time_cublas, transferedBytes / av_time_cublas);

    if (A) free(A);
    if (B) free(B);
    if (C) free(C);
    if (A_d) cudaFree(A_d);
    if (B_d) cudaFree(B_d);
    if (C_d) cudaFree(C_d);
    // if (C_gemm) cudaFree(C_gemm);

    if (cublas_handle) CUDA_CHECK(cublasDestroy(cublas_handle));
    return 0;
}
