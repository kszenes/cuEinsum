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

#include <cublasLt.h>
#include "utils.h"
#include "timer.h"
#include "rmse.h"


int main()
{
    printf("cuTENSOR version: %zu\n", cutensorGetVersion());

    const int runs = 1;

    // --- Single precision ---
    printf("Single precision\n");
    typedef float floatTypeA;
    typedef float floatTypeB;
    typedef float floatTypeC;
    typedef float floatTypeCompute;

    cudaDataType_t typeA = CUDA_R_32F;
    cudaDataType_t typeB = CUDA_R_32F;
    cudaDataType_t typeC = CUDA_R_32F;
    cutensorComputeType_t typeCompute = CUTENSOR_COMPUTE_TF32;
    cublasComputeType_t cublasComputeType = CUBLAS_COMPUTE_32F_FAST_TF32;

    // cutensorComputeType_t typeCompute = CUTENSOR_COMPUTE_32F;
    // cublasComputeType_t cublasComputeType = CUBLAS_COMPUTE_32F;

    // --- Double precision ---
    // printf("Double precision\n");
    // typedef double floatTypeA;
    // typedef double floatTypeB;
    // typedef double floatTypeC;
    // typedef double floatTypeCompute;

    // cudaDataType_t typeA = CUDA_R_64F;
    // cudaDataType_t typeB = CUDA_R_64F;
    // cudaDataType_t typeC = CUDA_R_64F;
    // cutensorComputeType_t typeCompute = CUTENSOR_COMPUTE_64F;
    // cublasComputeType_t cublasComputeType = CUBLAS_COMPUTE_64F;
    // // // --- END ---

    floatTypeCompute alpha = (floatTypeCompute)1.7f;
    floatTypeCompute beta  = (floatTypeCompute)0.f;

    printf("Include headers and define data types\n");

    /**********************
     * Computing: C_{m,u,n,v} = alpha * A_{m,h,k,n} B_{u,k,v,h} + beta * C_{m,u,n,v}
     **********************/

    // std::vector<int> modeC{'m','u','n','v'};
    // std::vector<int> modeA{'m','h','k','n'};
    // std::vector<int> modeB{'u','k','v','h'};
    // int nmodeA = modeA.size();
    // int nmodeB = modeB.size();
    // int nmodeC = modeC.size();

    // std::unordered_map<int, int64_t> extent;
    // extent['m'] = 96;
    // extent['n'] = 96;
    // extent['u'] = 96;
    // extent['v'] = 64;
    // extent['h'] = 64;
    // extent['k'] = 64;

    // extent['m'] = 255;
    // extent['n'] = 127;
    // extent['u'] = 129;
    // extent['v'] = 65;
    // extent['h'] = 62;
    // extent['k'] = 63;

    // double tflops = (2.0 * extent['m'] * extent['n'] * extent['u'] * extent['v'] * extent['k'] * extent['h']) /1e12;

    /**************
      MATMUL
    **************/
    std::vector<int> modeA{'i','j'};
    std::vector<int> modeB{'j','k'};
    std::vector<int> modeC{'i','k'};
    int nmodeA = modeA.size();
    int nmodeB = modeB.size();
    int nmodeC = modeC.size();

    std::unordered_map<int, int64_t> extent;
    int size = 1 << 14;
    const int i = size;
    const int j = size;
    const int k = size;
    printf("i = %d; j = %d; k = %d\n", i, j, k);

    extent['i'] = i;
    extent['j'] = j;
    extent['k'] = k;


    // // computes FLOPS
    double tflops = (2.0 * extent['i'] * extent['j'] * extent['k']) /1e12;

    std::vector<int64_t> extentC;
    for (auto mode : modeC)
        extentC.push_back(extent[mode]);
    std::vector<int64_t> extentA;
    for (auto mode : modeA)
        extentA.push_back(extent[mode]);
    std::vector<int64_t> extentB;
    for (auto mode : modeB)
        extentB.push_back(extent[mode]);

    printf("Define modes and extents\n");
    /**********************
     * Allocating data
     **********************/
    printf("cuTensor handle init\n");

    size_t elementsA = 1;
    for (auto mode : modeA)
        elementsA *= extent[mode];
    size_t elementsB = 1;
    for (auto mode : modeB)
        elementsB *= extent[mode];
    size_t elementsC = 1;
    for (auto mode : modeC)
        elementsC *= extent[mode];

    size_t sizeA = sizeof(floatTypeA) * elementsA;
    size_t sizeB = sizeof(floatTypeB) * elementsB;
    size_t sizeC = sizeof(floatTypeC) * elementsC;
    // printf("Elements A: %zu\n", elementsA);
    // printf("Elements B: %zu\n", elementsB);
    // printf("Elements C: %zu\n", elementsC);
    printf("Total memory: %.2f GiB\n", (sizeA + sizeB + sizeC)/1024./1024./1024);

    double transferedBytes = sizeC + sizeA + sizeB;
    transferedBytes += ((float) beta != 0.f) ? sizeC : 0;
    transferedBytes /= 1e9;

    floatTypeA *A_d, *B_d, *C_d;
    CUDA_CHECK(cudaMalloc((void**) &A_d, sizeA));
    CUDA_CHECK(cudaMalloc((void**) &B_d, sizeB));
    CUDA_CHECK(cudaMalloc((void**) &C_d, sizeC));


    floatTypeA *A = (floatTypeA*) malloc(sizeof(floatTypeA) * elementsA);
    // printf("A malloc successful\n");
    floatTypeB *B = (floatTypeB*) malloc(sizeof(floatTypeB) * elementsB);
    // printf("B malloc successful\n");
    floatTypeC *C = (floatTypeC*) malloc(sizeof(floatTypeC) * elementsC);
    // printf("C malloc successful\n");

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


    /*************
     * Compute GEMM CUBLAS
     ************/
    printf("Run CUBLAS baseline\n");
    floatTypeC *C_cublas;
    CUDA_CHECK(cudaMalloc((void**) &C_cublas, sizeC));
    CUDA_CHECK(cudaMemcpy(C_cublas, C, sizeC, cudaMemcpyHostToDevice));


    cudaStream_t s;
    cublasLtHandle_t lt_handle;
    CUDA_CHECK(cudaStreamCreate(&s));
    CUDA_CHECK(cublasLtCreate(&lt_handle));
    // CUDA_CHECK(cublasLoggerConfigure(1, 1, 0, nullptr));

    int returnedResults                             = 0;
    cublasLtMatmulHeuristicResult_t heuristicResult = {};

    void* cublas_work;
    size_t cublas_worksize = 4*1024*1024; // 4 MiB
    // size_t cublas_worksize = 0;
    CUDA_CHECK(cudaMalloc(&cublas_work, cublas_worksize));

    cublasLtMatmulDesc_t operationDesc = NULL;
    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
    cublasLtMatmulPreference_t preference = NULL;

    // create operation desciriptor; see cublasLtMatmulDescAttributes_t for details about defaults; here we just need to
    // set the transforms for A and B
    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;
    CUDA_CHECK(cublasLtMatmulDescCreate(&operationDesc, cublasComputeType, typeA));
    CUDA_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    CUDA_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transa)));

    // create matrix descriptors, we are good with the details here so no need to set any extra attributes
    CUDA_CHECK(cublasLtMatrixLayoutCreate(&Adesc, typeA, transa == CUBLAS_OP_N ? i : j, transa == CUBLAS_OP_N ? j : i, i));
    CUDA_CHECK(cublasLtMatrixLayoutCreate(&Bdesc, typeB, transb == CUBLAS_OP_N ? j : k, transb == CUBLAS_OP_N ? k : j, j));
    CUDA_CHECK(cublasLtMatrixLayoutCreate(&Cdesc, typeC, i, k, i));

    // create preference handle; here we could use extra attributes to disable tensor ops or to make sure algo selected
    // will work with badly aligned A, B, C; here for simplicity we just assume A,B,C are always well aligned (e.g.
    // directly come from cudaMalloc)
    CUDA_CHECK(cublasLtMatmulPreferenceCreate(&preference));
    CUDA_CHECK(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &cublas_work, sizeof(cublas_worksize)));
    CUDA_CHECK(cublasLtMatmulAlgoGetHeuristic(lt_handle, operationDesc, Adesc, Bdesc, Cdesc, Cdesc, preference, 1, &heuristicResult, &returnedResults));

    if (returnedResults == 0) {
        CUDA_CHECK(CUBLAS_STATUS_NOT_SUPPORTED);
    }

    double av_time_cublas = 0.0;
    double min_time_cublas = 1e8;
    for (int iter = 0; iter < runs; iter++) {
        GPUTimer timer;

    CUDA_CHECK(cublasLtMatmul(lt_handle,
                                     operationDesc,
                                     &alpha,
                                     A,
                                     Adesc,
                                     B,
                                     Bdesc,
                                     &beta,
                                     C,
                                     Cdesc,
                                     C,
                                     Cdesc,
                                     &heuristicResult.algo,
                                     cublas_work,
                                     cublas_worksize,
                                     s));

    timer.start();
    CUDA_CHECK(cublasLtMatmul(lt_handle,
                                     operationDesc,
                                     &alpha,
                                     A,
                                     Adesc,
                                     B,
                                     Bdesc,
                                     &beta,
                                     C,
                                     Cdesc,
                                     C,
                                     Cdesc,
                                     &heuristicResult.algo,
                                     cublas_work,
                                     cublas_worksize,
                                     s));
        auto time  = timer.seconds();
        min_time_cublas = (time  < min_time_cublas) ? time : min_time_cublas;
        av_time_cublas += time / runs;
    }
    printf("CUBLAS: %.2f GB/s %.2f TFLOP/s\n", transferedBytes / av_time_cublas, tflops / av_time_cublas);
    return 0;


    /*************************
     * cuTENSOR
     *************************/ 
    cutensorHandle_t handle;
    CUDA_CHECK(cutensorInit(&handle));

    /**********************
     * Create Tensor Descriptors
     **********************/

    cutensorTensorDescriptor_t descA;
    CUDA_CHECK(cutensorInitTensorDescriptor(&handle,
                 &descA,
                 nmodeA,
                 extentA.data(),
                 NULL,/*stride*/
                 typeA, CUTENSOR_OP_IDENTITY));

    cutensorTensorDescriptor_t descB;
    CUDA_CHECK(cutensorInitTensorDescriptor(&handle,
                 &descB,
                 nmodeB,
                 extentB.data(),
                 NULL,/*stride*/
                 typeB, CUTENSOR_OP_IDENTITY));

    cutensorTensorDescriptor_t descC;
    CUDA_CHECK(cutensorInitTensorDescriptor( &handle,
                 &descC,
                 nmodeC,
                 extentC.data(),
                 NULL,/*stride*/
                 typeC, CUTENSOR_OP_IDENTITY));

    printf("Initialize cuTENSOR and tensor descriptors\n");
    /**********************************************
     * Retrieve the memory alignment for each tensor
     **********************************************/ 

     uint32_t alignmentRequirementA;
     CUDA_CHECK(cutensorGetAlignmentRequirement(&handle,
                  A_d,
                  &descA,
                  &alignmentRequirementA));

     uint32_t alignmentRequirementB;
     CUDA_CHECK(cutensorGetAlignmentRequirement(&handle,
                  B_d,
                  &descB,
                  &alignmentRequirementB));

     uint32_t alignmentRequirementC;
     CUDA_CHECK(cutensorGetAlignmentRequirement(&handle,
                  C_d,
                  &descC, 
                  &alignmentRequirementC));

    printf("Query best alignment requirement for our pointers\n");
    /*******************************
     * Create Contraction Descriptor
     *******************************/

    cutensorContractionDescriptor_t desc;
    CUDA_CHECK(cutensorInitContractionDescriptor(&handle, 
                 &desc,
                 &descA, modeA.data(), alignmentRequirementA,
                 &descB, modeB.data(), alignmentRequirementB,
                 &descC, modeC.data(), alignmentRequirementC,
                 &descC, modeC.data(), alignmentRequirementC,
                 typeCompute));

    printf("Initialize contraction descriptor\n");
    /**************************
    * Set the algorithm to use
    ***************************/

    cutensorContractionFind_t find;

    CUDA_CHECK(cutensorInitContractionFind( 
                 &handle, &find, 
                 CUTENSOR_ALGO_DEFAULT_PATIENT));

    // CUDA_CHECK(cutensorInitContractionFind( 
    //              &handle, &find, 
    //              (cutensorAlgo_t) -6)); // 1 is usually best for matmul

    printf("Initialize settings to find algorithm\n");
    /**********************
     * Query workspace
     **********************/

    uint64_t worksize = 0;
    CUDA_CHECK(cutensorContractionGetWorkspaceSize(&handle,
                 &desc,
                 &find,
                 CUTENSOR_WORKSPACE_MAX, &worksize));

    worksize = 1024*1024*1024;

    void *work = nullptr;
    if (worksize > 0)
    {
        if (cudaSuccess != cudaMalloc(&work, worksize))
        {
            work = nullptr;
            worksize = 0;
        }
    } 
    int32_t maxAlgosTC = 0;
    cutensorContractionMaxAlgos(&maxAlgosTC);
    double bestTime = 1e100;
    int bestAlgo = -1;

    printf("Query recommended workspace size (%zu MiB) and allocate it\n", worksize/1024/1024);
    /**************************
     * Create Contraction Plan
     **************************/

    cutensorContractionPlan_t plan;
    CUDA_CHECK(cutensorInitContractionPlan(&handle,
                 &plan,
                 &desc,
                 &find,
                 worksize));

    printf("Create plan for contraction\n");
    /**********************
     * Run
     **********************/

    cutensorStatus_t err;
    for (int algo = (int) CUTENSOR_ALGO_DEFAULT_PATIENT; algo < 6; algo++) {
        double minTimeCUTENSOR = 1e100;
        double avTime = 0;
        for (int i=0; i < runs; ++i)
        {
            CUDA_CHECK(cudaMemcpy(C_d, C, sizeC, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaDeviceSynchronize());

            cutensorContractionFind_t find;
            err = cutensorInitContractionFind(&handle, &find, (cutensorAlgo_t) algo);

            if (err == CUTENSOR_STATUS_SUCCESS) {
                cutensorContractionPlan_t plan;
                err = cutensorInitContractionPlan(&handle,
                                                  &plan,
                                                  &desc,
                                                  &find,
                                                  worksize);

                if (err == CUTENSOR_STATUS_SUCCESS)
                {
                    // Set up timing
                    GPUTimer timer;
                    err = cutensorContraction(&handle,
                                            &plan,
                                            (void*) &alpha, A_d, B_d,
                                            (void*) &beta,  C_d, C_d, 
                                            work, worksize, 0 /* stream */);
                    timer.start();

                    err = cutensorContraction(&handle,
                                            &plan,
                                            (void*) &alpha, A_d, B_d,
                                            (void*) &beta,  C_d, C_d, 
                                            work, worksize, 0 /* stream */);

                    // Synchronize and measure timing
                    auto time = timer.seconds();
                    // printf("Time: %f\n", time);

                    if (err != CUTENSOR_STATUS_SUCCESS)
                    {
                        printf("ERROR: %s in line %d\n", cutensorGetErrorString(err), __LINE__);
                    }
                    avTime += time / runs;
                    minTimeCUTENSOR = (minTimeCUTENSOR < time) ? minTimeCUTENSOR : time;
                }

            }
        }
        if (err != CUTENSOR_STATUS_NOT_SUPPORTED)
        {
            printf("cuTensor: %d algo %.2f GB/s %.2f TFLOP/s\n", algo, transferedBytes / minTimeCUTENSOR, tflops / avTime);
        }

        if (bestTime > minTimeCUTENSOR)
        {
            bestTime = minTimeCUTENSOR;
            bestAlgo = algo;
        }

    }

    floatTypeA rmse_diff = rmse(j*k, C_d, C_cublas);
    printf("RMSE = %f\n", rmse_diff);

    printf("Execute contraction from plan\n");
    /*************************/

    printf("\nRESULTS from %d runs:\n", runs);
    // printf("Time cuTENSOR[s]: Best: %.4f; Mean %.4f\n",
    //         minTimeCUTENSOR, avTime);
    printf("Time cuBLAS[s]: Best: %.4f; Mean %.4f\n",
            min_time_cublas, av_time_cublas);
    printf("Compute cuBLAS [TFLOPS/s]: Best: %.4f;  Mean %.4f\n",
            tflops / min_time_cublas, tflops/ av_time_cublas);
    printf("CUTENSOR best: %d algo %.2f GB/s %.2f TFLOP/s\n", bestAlgo, transferedBytes / bestTime, tflops / bestTime);
    // printf("Compute cuTENSOR [TFLOPS/s]: Best: %.4f;  Mean %.4f\n",
    //         tflops / minTimeCUTENSOR, tflops/ avTime);
    // printf("Memory [GB/s]: Best: %.2f;  Mean: %.2f\n",
    //         transferedBytes / minTimeCUTENSOR, transferedBytes / avTime);

    if (A) free(A);
    if (B) free(B);
    if (C) free(C);
    if (A_d) cudaFree(A_d);
    if (B_d) cudaFree(B_d);
    if (C_d) cudaFree(C_d);
    if (work) cudaFree(work);

    if (C_cublas) CUDA_CHECK(cudaFree(C_cublas));
    if (lt_handle) CUDA_CHECK(cublasLtDestroy(lt_handle));
    // descriptors are no longer needed as all GPU work was already enqueued
    if (preference) CUDA_CHECK(cublasLtMatmulPreferenceDestroy(preference));
    if (Cdesc) CUDA_CHECK(cublasLtMatrixLayoutDestroy(Cdesc));
    if (Bdesc) CUDA_CHECK(cublasLtMatrixLayoutDestroy(Bdesc));
    if (Adesc) CUDA_CHECK(cublasLtMatrixLayoutDestroy(Adesc));
    if (operationDesc) CUDA_CHECK(cublasLtMatmulDescDestroy(operationDesc));
    if (cublas_work) CUDA_CHECK(cudaFree(cublas_work));

    return 0;
}
