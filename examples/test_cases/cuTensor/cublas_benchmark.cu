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


int main()
{
    // --- Parameters ---
    #define DOUBLE
    const int runs = 3;
    const int worksizePref = 3; // 0: 0[Mib]; 1: MIN; 2: RECOMMENDED; 3: MAX
    const bool printDebug = false;
    const bool allAlgos = false;
    const bool checkRMSE = false;
    const int deviceId = 1;

    printf("Workspace preference: ");
    switch (worksizePref) {
        case 0: printf("0 [MB]\n"); break;
        case 1: printf("MIN\n"); break;
        case 2: printf("RECOMMENDED\n"); break;
        case 3: printf("MAX\n"); break;
        default: printf("Unsupported worksizePref: %d", worksizePref); exit(-1);
    }
    CUDA_CHECK(cudaSetDevice(deviceId));

    if (printDebug) printf("cuTENSOR version: %zu\n", cutensorGetVersion());

    #if defined(FLOAT)
    printf("Single precision\n");
    typedef float floatTypeA;
    typedef float floatTypeB;
    typedef float floatTypeC;
    typedef float floatTypeCompute;

    cudaDataType_t typeA = CUDA_R_32F;
    cudaDataType_t typeB = CUDA_R_32F;
    cudaDataType_t typeC = CUDA_R_32F;
    cutensorComputeType_t typeCompute = CUTENSOR_COMPUTE_32F;
    cublasComputeType_t cublasComputeType = CUBLAS_COMPUTE_32F;
    #undef TENSOR
    #elif defined(TENSOR)
    printf("Tensor float precision\n");
    typedef float floatTypeA;
    typedef float floatTypeB;
    typedef float floatTypeC;
    typedef float floatTypeCompute;
    cudaDataType_t typeA = CUDA_R_32F;
    cudaDataType_t typeB = CUDA_R_32F;
    cudaDataType_t typeC = CUDA_R_32F;
    cutensorComputeType_t typeCompute = CUTENSOR_COMPUTE_TF32;
    cublasComputeType_t cublasComputeType = CUBLAS_COMPUTE_32F_FAST_TF32;
    #undef TENSOR
    #elif defined(DOUBLE)
    printf("Double precision\n");
    typedef double floatTypeA;
    typedef double floatTypeB;
    typedef double floatTypeC;
    typedef double floatTypeCompute;

    cudaDataType_t typeA = CUDA_R_64F;
    cudaDataType_t typeB = CUDA_R_64F;
    cudaDataType_t typeC = CUDA_R_64F;
    cutensorComputeType_t typeCompute = CUTENSOR_COMPUTE_64F;
    cublasComputeType_t cublasComputeType = CUBLAS_COMPUTE_64F;
    #undef DOUBLE
    #endif

    floatTypeCompute alpha = (floatTypeCompute)1.7f;
    floatTypeCompute beta  = (floatTypeCompute)0.f;

    if (printDebug) printf("Include headers and define data types\n");

    /**************
      MATMUL
    **************/
    std::string A_string = "ik";
    std::string B_string = "kj";
    std::string C_string = "ij";

    std::vector<int> modeA;
    for (char c : A_string) {
        modeA.push_back(c);
    }
    std::vector<int> modeB;
    for (char c : B_string) {
        modeB.push_back(c);
    }
    std::vector<int> modeC;
    for (char c : C_string) {
        modeC.push_back(c);
    }
    
    int nmodeA = modeA.size();
    int nmodeB = modeB.size();
    int nmodeC = modeC.size();

    std::unordered_map<int, int64_t> extent;
    // const int size = 1 << 14;
    const size_t size = 4096 * 8;
    const size_t i = size;
    const size_t j = size;
    const size_t k = size;
    printf("i = %zu; j = %zu; k = %zu\n", i, j, k);

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

    if (printDebug) printf("Define modes and extents\n");
    /**********************
     * Allocating data
     **********************/
    if (printDebug) printf("cuTensor handle init\n");

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
    if (printDebug) printf("Elements A: %zu\n", elementsA);
    if (printDebug) printf("Elements B: %zu\n", elementsB);
    if (printDebug) printf("Elements C: %zu\n", elementsC);
    printf("Total memory: %.2f GiB\n", (sizeA + sizeB + sizeC)/1024./1024./1024);

    double transferedBytes = sizeC + sizeA + sizeB;
    transferedBytes += ((float) beta != 0.f) ? sizeC : 0;
    transferedBytes /= 1e9;

    floatTypeA *A_d, *B_d, *C_d;
    CUDA_CHECK(cudaMalloc((void**) &A_d, sizeA));
    CUDA_CHECK(cudaMalloc((void**) &B_d, sizeB));
    CUDA_CHECK(cudaMalloc((void**) &C_d, sizeC));


    floatTypeA *A = (floatTypeA*) malloc(sizeof(floatTypeA) * elementsA);
    if (printDebug) printf("A malloc successful\n");
    floatTypeB *B = (floatTypeB*) malloc(sizeof(floatTypeB) * elementsB);
    if (printDebug) printf("B malloc successful\n");
    floatTypeC *C = (floatTypeC*) malloc(sizeof(floatTypeC) * elementsC);
    if (printDebug) printf("C malloc successful\n");

    if (A == NULL || B == NULL || C == NULL)
    {
        printf("Error: Host allocation of A or C.\n");
        return -1;
    }

    if (printDebug) printf("Allocate, initialize and transfer tensors\n");
    /*******************
     * Initialize data
     *******************/

    if (checkRMSE) {
        for (size_t i = 0; i < elementsA; i++)
            A[i] = (((floatTypeA) rand())/RAND_MAX - 0.5)*100;
        for (size_t i = 0; i < elementsB; i++)
            B[i] = (((floatTypeB) rand())/RAND_MAX - 0.5)*100;
        for (size_t i = 0; i < elementsC; i++)
            C[i] = (((floatTypeC) rand())/RAND_MAX - 0.5)*100;
        CUDA_CHECK(cudaMemcpy(A_d, A, sizeA, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(B_d, B, sizeB, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(C_d, C, sizeC, cudaMemcpyHostToDevice));
    }

    /*************
     * Compute GEMM CUBLAS
     ************/
    printf("=== cuBLAS ===\n");
    floatTypeC *C_cublas;
    CUDA_CHECK(cudaMalloc((void**) &C_cublas, sizeC));
    CUDA_CHECK(cudaMemcpy(C_cublas, C, sizeC, cudaMemcpyHostToDevice));

    cudaStream_t s;
    cublasHandle_t cublas_handle;
    CUDA_CHECK(cudaStreamCreate(&s));
    CUDA_CHECK(cublasCreate(&cublas_handle));
    // CUDA_CHECK(cublasLoggerConfigure(1, 1, 0, nullptr));

    void* work_cublas;
    // size_t cublas_worksize = 4*1024*1024; // 4 MiB
    size_t cublas_worksize = 0;
    CUDA_CHECK(cudaMalloc(&work_cublas, cublas_worksize));
    CUDA_CHECK(cublasSetWorkspace(cublas_handle, &work_cublas, cublas_worksize));


    double av_time_cublas = 0.0;
    double min_time_cublas = 1e8;
    for (int iter = 0; iter < runs; iter++) {
        GPUTimer timer;
        CUDA_CHECK(cublasGemmEx(
                    cublas_handle, 
                    CUBLAS_OP_N, CUBLAS_OP_N, 
                    i, k, j, 
                    &alpha, 
                    A_d, typeA, i, 
                    B_d, typeB, j, 
                    &beta, 
                    C_cublas, typeC, i,
                    cublasComputeType,
                    CUBLAS_GEMM_DEFAULT)); // warmup
        timer.start();
        CUDA_CHECK(cublasGemmEx(
                    cublas_handle, 
                    CUBLAS_OP_N, CUBLAS_OP_N, 
                    i, k, j, 
                    &alpha, 
                    A_d, typeA, i, 
                    B_d, typeB, j, 
                    &beta, 
                    C_cublas, typeC, i,
                    cublasComputeType,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP)); // warmup
        auto time  = timer.seconds();
        min_time_cublas = (time  < min_time_cublas) ? time : min_time_cublas;
        av_time_cublas += time / runs;
    }
    printf("CUBLAS: %.2f GB/s %.2f TFLOP/s\n", transferedBytes / av_time_cublas, tflops / av_time_cublas);

    /*************************
     * cuTENSOR
     *************************/ 
    cutensorHandle_t handle;
    CUDA_CHECK(cutensorInit(&handle));

    /**********************
     * Create Tensor Descriptors
     **********************/
    printf("=== cuTENSOR ===\n");

    GPUTimer timer;
    timer.start();
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

    auto time = timer.seconds();
    if (printDebug) printf("Initialize cuTENSOR and tensor descriptors (%f sec)\n", time);
    /**********************************************
     * Retrieve the memory alignment for each tensor
     **********************************************/ 

    timer.start();
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

    time = timer.seconds();
    if (printDebug) printf("Query best alignment requirement for our pointers (%f sec)\n", time);
    /*******************************
     * Create Contraction Descriptor
     *******************************/

    timer.start();
    cutensorContractionDescriptor_t desc;
    CUDA_CHECK(cutensorInitContractionDescriptor(&handle, 
                 &desc,
                 &descA, modeA.data(), alignmentRequirementA,
                 &descB, modeB.data(), alignmentRequirementB,
                 &descC, modeC.data(), alignmentRequirementC,
                 &descC, modeC.data(), alignmentRequirementC,
                 typeCompute));

    time = timer.seconds();
    if (printDebug) printf("Initialize contraction descriptor (%f sec)\n", time);
    /**************************
    * Set the algorithm to use
    ***************************/

    timer.start();
    cutensorContractionFind_t find;

    CUDA_CHECK(cutensorInitContractionFind( 
                 &handle, &find, 
                 CUTENSOR_ALGO_DEFAULT_PATIENT));

    // CUDA_CHECK(cutensorInitContractionFind( 
    //              &handle, &find, 
    //              (cutensorAlgo_t) -6)); // 1 is usually best for matmul

    time = timer.seconds();
    if (printDebug) printf("Initialize settings to find algorithm (%f sec)\n", time);
    /**********************
     * Query workspace
     **********************/
    size_t freeMem, totalMem;
    HANDLE_CUDA_ERROR( cudaMemGetInfo(&freeMem, &totalMem) );
    if (printDebug) printf("Free Memory: %zu; Total Memory: %zu\n", freeMem/1024/1024/1024, totalMem/1024/1024/1024);

    timer.start();
    uint64_t worksize = 0;
    if (worksizePref) {
        if (worksizePref == 4) {
            CUDA_CHECK(cutensorContractionGetWorkspaceSize(&handle,
                        &desc,
                        &find,
                        (cutensorWorksizePreference_t) CUTENSOR_WORKSPACE_MAX, &worksize));
        } else {
            CUDA_CHECK(cutensorContractionGetWorkspaceSize(&handle,
                        &desc,
                        &find,
                        (cutensorWorksizePreference_t) worksizePref, &worksize));
        }
    }

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

    time = timer.seconds();
    printf("Sizes[MiB]: A = %zu ; B = %zu; C = %zu\n", sizeA/1024/1024, sizeB/1024/1024, sizeC/1024/1024);
    printf("Query recommended workspace size (%zu MiB) and allocate it (%f sec)\n", worksize/1024/1024, time);
    /**************************
     * Create Contraction Plan
     **************************/

    timer.start();
    cutensorContractionPlan_t plan;
    CUDA_CHECK(cutensorInitContractionPlan(&handle,
                 &plan,
                 &desc,
                 &find,
                 worksize));

    time = timer.seconds();
    if (printDebug) printf("Create plan for contraction (%f sec)\n", time);
    /**********************
     * Run
     **********************/
    cutensorStatus_t err;
    const int algoToTry = allAlgos ? 6 : -5; // only try default patient
    for (int algo = (int) CUTENSOR_ALGO_DEFAULT_PATIENT; algo < algoToTry; algo++) {
        double minTimeCUTENSOR = 1e100;
        double avTime = 0;
        for (int iter=0; iter < runs; iter++)
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
                // Set up timing
                if (err == CUTENSOR_STATUS_SUCCESS) {
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
            if (checkRMSE) {
                floatTypeA rmse_diff = rmse(i*j, C_d, C_cublas);
                printf("RMSE = %f\t", rmse_diff);

            } 
            printf("cuTensor: %d algo %.2f GB/s %.2f TFLOP/s\n", algo, transferedBytes / minTimeCUTENSOR, tflops / avTime);
        }

        if (bestTime > minTimeCUTENSOR)
        {
            bestTime = minTimeCUTENSOR;
            bestAlgo = algo;
        }

    }

    if (printDebug) printf("Execute contraction from plan\n");
    /*************************/

    printf("\nRESULTS from %d runs:\n", runs);
    printf("Time cuBLAS[s]: Best: %.4f; Mean %.4f\n",
            min_time_cublas, av_time_cublas);
    printf("Compute cuBLAS [TFLOPS/s]: Best: %.4f;  Mean %.4f\n",
            tflops / min_time_cublas, tflops/ av_time_cublas);
    printf("CUTENSOR best: %d algo %.2f GB/s %.2f TFLOP/s\n", bestAlgo, transferedBytes / bestTime, tflops / bestTime);


    if (A) free(A);
    if (B) free(B);
    if (C) free(C);
    if (A_d) cudaFree(A_d);
    if (B_d) cudaFree(B_d);
    if (C_d) cudaFree(C_d);
    if (work) cudaFree(work);

    if (C_cublas) CUDA_CHECK(cudaFree(C_cublas));
    if (cublas_handle) CUDA_CHECK(cublasDestroy(cublas_handle));

    return 0;
}
