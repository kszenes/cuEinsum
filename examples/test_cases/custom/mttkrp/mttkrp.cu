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

#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>
#include <cutensor.h>

#include <cublas_v2.h>
#include "utils.h"
#include "timer.h"
#include "rmse.h"
#include <string>
#include <iostream>

template <typename T>
__global__
void khatriRao(const size_t xDim, const size_t yDim, const size_t nBatches, const T* x, const T* y, T* mat) {
    const int row = threadIdx.x + blockIdx.x * blockDim.x;
    const int col = threadIdx.y + blockIdx.y * blockDim.y;
    const int batch = threadIdx.z + blockIdx.z * blockDim.z;
    if (row < xDim && col < yDim && batch < nBatches) {
        mat[(batch * xDim * yDim) + row + col * xDim] = x[(batch * xDim) + row] * y[(batch * yDim) + col];
    }
}

// Computes: tmp[j,k,a] =  B[j,a] * C[k,a]
template <typename T>
void hostKRP(
    const size_t dimJ, const size_t dimK, const size_t dimA, const T* B, const T* C, T* tmp) {
        for (size_t j = 0; j < dimJ; j++)
            for (size_t k = 0; k < dimK; k++)
                for (size_t a = 0; a < dimA; a++)
                    tmp[j + k * dimJ + a * dimJ * dimK] += B[j + a * dimJ] * C[k + a * dimK];
}

// Computes: D[i,a] =  A[i,j,k] * tmp[j,k,a]
template <typename T>
void hostGEMM(
    const size_t M, const size_t N, const size_t K, const T* A, const T* tmp, T* D) {
        for (size_t m = 0; m < M; m++)
            for (size_t n = 0; n < N; n++)
                for (size_t k = 0; k < K; k++)
                    D[m + n * M] += A[m + k * M] * tmp[k + n * K];
}

// Computes: D[i,a] =  A[i,j,k] * B[j,a] * C[k,a]
template <typename T>
void hostMTTKRP(
    const size_t dimI, const size_t dimJ, const size_t dimK, const size_t dimA, const T* A, const T* B, const T* C, T* D) {
        for (size_t i = 0; i < dimI; i++) 
            for (size_t j = 0; j < dimJ; j++)
                for (size_t k = 0; k < dimK; k++)
                    for (size_t a = 0; a < dimA; a++)
                        D[i + a * dimI] += A[i + j * dimI + k * (dimI * dimJ)] 
                                          * B[j + a * dimJ] * C[k + a * dimK];
}

int main()
{
    // MTTKRP: 'ijk,ja,ka->ia'  A,B,C->D
    // 1) KRP: 'ja,ka->jka'     B,C->tmp
    // 2) MTT: 'ijk,jka->ia'    A,tmp->D

    // --- Parameters ---
    CUDA_CHECK(cudaSetDevice(1));
    const bool checkRMSE = false;

    // const size_t dim = 1 << 8;
    // const size_t dimI = dim + 27;
    // const size_t dimJ = dim - 53;
    // const size_t dimK = dim + 19;
    // const size_t dimA = dim - 41;

    const size_t dim = 1 << 10;
    const size_t dimI = dim;
    const size_t dimJ = dim;
    const size_t dimK = dim;
    const size_t dimA = dim;
    typedef double floatType;
    const floatType alpha = 1.;
    const floatType beta = 0.;

    double memTensors = (dimI * dimJ * dimK + dimJ * dimA + dimK * dimA + dimI * dimA) * sizeof(floatType) / 1e9;
    double memTmp = (dimJ * dimK * dimA) * sizeof(floatType) / 1e9;
    printf("=== Memory ===\n");
    printf("Tensors:\t%f GB\n", memTensors);
    printf("Temp:   \t%f GB\n", memTmp);

    GPUTimer timer;
    floatType *A_h, *B_h, *C_h, *D_h;

    CUDA_CHECK(cudaMallocHost(&A_h, dimI * dimJ * dimK * sizeof(floatType)));
    CUDA_CHECK(cudaMallocHost(&B_h, dimJ * dimA * sizeof(floatType)));
    CUDA_CHECK(cudaMallocHost(&C_h, dimK * dimA * sizeof(floatType)));
    CUDA_CHECK(cudaMallocHost(&D_h, dimI * dimA * sizeof(floatType)));


    floatType *tmp_h = nullptr;
    if (checkRMSE) CUDA_CHECK(cudaMallocHost(&tmp_h, dimJ * dimK * dimA * sizeof(floatType)));
    for (size_t i = 0; i < dimI * dimJ * dimK; i++) {
        A_h[i] = (((floatType) rand()) / RAND_MAX - 0.5) * 1.;
    }
    for (size_t i = 0; i < dimJ * dimA; i++) {
        B_h[i] = (((floatType) rand()) / RAND_MAX - 0.5) * 1.;
    }
    for (size_t i = 0; i < dimK * dimA; i++) {
        C_h[i] = (((floatType) rand()) / RAND_MAX - 0.5) * 1.;
    }
    for (size_t i = 0; i < dimI * dimA; i++) {
        D_h[i] = 0.;
    }

    if (checkRMSE) {
        hostKRP(dimJ, dimK, dimA, B_h, C_h, tmp_h);
        hostMTTKRP(dimI, dimJ, dimK, dimA, A_h, B_h, C_h, D_h);
    }

    floatType *A_d, *B_d, *C_d, *D_d;

    CUDA_CHECK(cudaMalloc(&A_d, dimI * dimJ * dimK * sizeof(floatType)));
    CUDA_CHECK(cudaMalloc(&B_d, dimJ * dimA * sizeof(floatType)));
    CUDA_CHECK(cudaMalloc(&C_d, dimK * dimA * sizeof(floatType)));
    CUDA_CHECK(cudaMalloc(&D_d, dimI * dimA * sizeof(floatType)));

    CUDA_CHECK(cudaMemcpy(A_d, A_h, dimI * dimJ * dimK * sizeof(floatType), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d, B_h, dimJ * dimA * sizeof(floatType), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(C_d, C_h, dimK * dimA * sizeof(floatType), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(D_d, 0, dimI * dimA * sizeof(floatType)));

    // === Khatri Rao ===

    floatType *tmp_d;
    CUDA_CHECK(cudaMalloc(&tmp_d, dimJ * dimK * dimA * sizeof(floatType)));

    dim3 dimThreads(16, 16, 4);
    dim3 dimBlocks(
        (dimJ + dimThreads.x - 1) / dimThreads.x,
        (dimK + dimThreads.y - 1) / dimThreads.y,
        (dimA + dimThreads.z - 1) / dimThreads.z
    );

    printf("=== Execution ===\n");
    timer.start();
    khatriRao<<<dimBlocks, dimThreads>>>(dimJ, dimK, dimA, B_d, C_d, tmp_d);
    auto krpTime = timer.seconds();
    printf("Khatri Rao:\t%f sec\n", krpTime);

    if (checkRMSE) {
        auto krpRMSE = hostRMSE(dimJ * dimK * dimA, tmp_h, tmp_d);
        printf("KRP RMSE: %f\n", krpRMSE);
    }

    // === GEMM === 'ijk,jka->ia'
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasStatus_t error;

    const size_t M = dimI;
    const size_t N = dimA;
    const size_t K = dimJ * dimK;

    cublasDataType_t cublasType = CUDA_R_64F;
    cublasComputeType_t cublasCompute = CUBLAS_COMPUTE_64F;

    timer.start();
    error = cublasGemmEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        M, N, K, &alpha, A_d, cublasType, M,
        tmp_d, cublasType, K, &beta,
        D_d, cublasType, M, cublasCompute,
        CUBLAS_GEMM_DEFAULT
    );
    auto gemmTime = timer.seconds();
    printf("GEMM:\t\t%f sec\n", gemmTime);
    if (error != CUBLAS_STATUS_SUCCESS) {
        printf("Error in GEMM call at line %d\n", __LINE__);
    }

    if (checkRMSE) {
        floatType *D_gemm;
        CUDA_CHECK(cudaMallocHost(&D_gemm, dimI * dimA * sizeof(floatType)));
        hostGEMM(dimI, dimA, dimJ * dimK, A_h, tmp_h, D_gemm);
        auto gemmRMSE = hostRMSE(dimI * dimA, D_gemm, D_d);
        printf("GEMM RMSE: %f\n", gemmRMSE);
        CUDA_CHECK(cudaFreeHost(D_gemm));
    }

    


    printf("Sum:\t\t%f sec\n", krpTime + gemmTime);
    if (checkRMSE) {
        auto myRMSE = hostRMSE(dimI * dimA, D_h, D_d);
        printf("=== RMSE = %f ===\n", myRMSE);
    }

    CUDA_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(tmp_d));

    CUDA_CHECK(cudaFree(D_d));
    CUDA_CHECK(cudaFree(C_d));
    CUDA_CHECK(cudaFree(B_d));
    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFreeHost(tmp_h));
    CUDA_CHECK(cudaFreeHost(D_h));
    CUDA_CHECK(cudaFreeHost(C_h));
    CUDA_CHECK(cudaFreeHost(B_h));
    CUDA_CHECK(cudaFreeHost(A_h));
}