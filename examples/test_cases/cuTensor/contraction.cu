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
#include <tuple>
#include <omp.h>

int findCommonIndices(std::string&, std::string&);


int main()
{
    // --- Parameters ---
    #define TENSOR
    const int runs = 1;
    const int worksizePref = 4; // 0: 0[Mib]; 1: MIN; 2: RECOMMENDED; 3: MAX
    const bool printDebug = false;
    const bool allAlgos = false;
    const bool cublasFlag = false;
    const bool checkRMSE = true;

    std::string A_string = "abcd";
    std::string B_string = "fdage";
    std::string C_string = "bcfge";
    const size_t size = 1 << 5;

    printf("Workspace size:\t\t");
    switch (worksizePref) {
        case 0: printf("0 [MB]\n"); break;
        case 1: printf("MIN\n"); break;
        case 2: printf("RECOMMENDED\n"); break;
        case 3: printf("MAX\n"); break;
        case 4: printf("ALL\n"); break;
        default: printf("Unsupported worksizePref: %d", worksizePref); exit(-1);
    }

    if (printDebug) printf("cuTENSOR version: %zu\n", cutensorGetVersion());

    #if defined(FLOAT)
    printf("Precision:\t\tF32\n");
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
    printf("Precision:\t\tTF32\n");
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
    printf("Precision:\t\tF64\n");
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

    // double tflops = (2.0 * extent['m'] * extent['n'] * extent['u'] * extent['v'] * extent['h'] * extent['k']) /1e12;

    /**************
      MATMUL
    **************/

    std::cout << "Contraction:\t\t" << A_string << ',' << B_string << "->" << C_string << '\n';

    const int contractedIndices = findCommonIndices(A_string, B_string);
    const size_t M = std::pow(size, A_string.length() - contractedIndices);
    const size_t K = std::pow(size, contractedIndices);
    const size_t N = std::pow(size, C_string.length() - (A_string.length() - contractedIndices));

    std::cout << "Num Indices:\t\tM: " << A_string.length() - contractedIndices << "; N: " << C_string.length() - (A_string.length() - contractedIndices) << "; K: " << contractedIndices << '\n';

    std::vector<int> modeA;
    std::vector<int> modeB;
    std::vector<int> modeC;
    for (char c : A_string) {
        modeA.push_back(c);
    }
    for (char c : B_string) {
        modeB.push_back(c);
    }
    for (char c : C_string) {
        modeC.push_back(c);
    }
    int nmodeA = modeA.size();
    int nmodeB = modeB.size();
    int nmodeC = modeC.size();

    std::unordered_map<int, int64_t> extent;
    for (char c : A_string) {
        extent[c] = size;
    }
    for (char c : B_string) {
        extent[c] = size;
    }
    for (char c : C_string) {
        extent[c] = size;
    }
    // const int n = size;
    // const int o = size;
    // const int p = size;

    // extent['n'] = n;
    // extent['o'] = o;
    // extent['p'] = p;

    // computes FLOPS
    double tflops = 2.0;
    for (auto &e : extent) {
        tflops *= e.second;
    }
    tflops /= 1e12;

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

    printf("Size [MiB]:\t\tA = %zu ; B = %zu; C = %zu\n", sizeA/1024/1024, sizeB/1024/1024, sizeC/1024/1024);
    printf("Total memory:\t\t%.2f GiB\n", (sizeA + sizeB + sizeC)/1024./1024./1024);
    double transferedBytes = sizeC + sizeA + sizeB;
    transferedBytes += ((float) beta != 0.f) ? sizeC : 0;
    transferedBytes /= 1e9;
    if (printDebug) printf("TFLOPS: %f;\nBytes: %f\n", tflops * 1e12, transferedBytes * 1e9);
    printf("Arithmetic Intensity:\t%.2f FLOP/Byte\n", tflops * 1e3 / transferedBytes);

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

    for (size_t i = 0; i < elementsA; i++)
        A[i] = (((floatTypeA) rand())/RAND_MAX - 0.5)*100;
    for (size_t i = 0; i < elementsB; i++)
        B[i] = (((floatTypeB) rand())/RAND_MAX - 0.5)*100;
    for (size_t i = 0; i < elementsC; i++)
        C[i] = (((floatTypeC) rand())/RAND_MAX - 0.5)*100;

    printf("init Done");

    CUDA_CHECK(cudaMemcpy(A_d, A, sizeA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d, B, sizeB, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(C_d, C, sizeC, cudaMemcpyHostToDevice));

    /*************
     * Compute GEMM CUBLAS
     ************/
    double av_time_cublas = 0.0;
    double min_time_cublas = 1e8;
    cublasHandle_t cublas_handle;
    floatTypeC *C_cublas = NULL;
    if (cublasFlag) {
        if (printDebug) printf("Run CUBLAS baseline\n");
        CUDA_CHECK(cudaMalloc((void**) &C_cublas, sizeC));
        CUDA_CHECK(cudaMemcpy(C_cublas, C, sizeC, cudaMemcpyHostToDevice));

        cudaStream_t s;
        CUDA_CHECK(cudaStreamCreate(&s));
        CUDA_CHECK(cublasCreate(&cublas_handle));
        // CUDA_CHECK(cublasSetMathMode(cublas_handle, CUBLAS_PEDANTIC_MATH));

        void* work_cublas;
        // size_t cublas_worksize = 4*1024*1024; // 4 MiB
        size_t cublas_worksize = 0;
        CUDA_CHECK(cudaMalloc(&work_cublas, cublas_worksize));
        CUDA_CHECK(cublasSetWorkspace(cublas_handle, &work_cublas, cublas_worksize));

        av_time_cublas = 0.0;
        min_time_cublas = 1e8;
        for (int iter = 0; iter < runs; iter++) {
            GPUTimer timer;
            CUDA_CHECK(cublasGemmEx(
                        cublas_handle, 
                        CUBLAS_OP_N, CUBLAS_OP_N, 
                        M, N, K, 
                        &alpha, 
                        A_d, typeA, M,
                        B_d, typeB, K,
                        &beta, 
                        C_cublas, typeC, M,
                        cublasComputeType,
                        CUBLAS_GEMM_DEFAULT)); // warmup
            timer.start();
            CUDA_CHECK(cublasGemmEx(
                        cublas_handle, 
                        CUBLAS_OP_N, CUBLAS_OP_N, 
                        M, N, K, 
                        &alpha, 
                        A_d, typeA, M,
                        B_d, typeB, K,
                        &beta, 
                        C_cublas, typeC, M,
                        cublasComputeType,
                        CUBLAS_GEMM_DEFAULT)); // warmup
            auto time  = timer.seconds();
            min_time_cublas = (time  < min_time_cublas) ? time : min_time_cublas;
            av_time_cublas += time / runs;
        }
        printf("=== CUBLAS ===\n");
        printf("CUBLAS: %.2f GB/s %.2f TFLOP/s\n", transferedBytes / av_time_cublas, tflops / av_time_cublas);
    }


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

    if (printDebug) printf("Initialize cuTENSOR and tensor descriptors\n");
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

    if (printDebug) printf("Query best alignment requirement for our pointers\n");
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

    if (printDebug) printf("Initialize contraction descriptor\n");
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

    if (printDebug) printf("Initialize settings to find algorithm\n");
    /**********************
     * Query workspace
     **********************/

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

    printf("=== CUTENSOR ===\n");
    printf("Query recommended workspace size (%zu MB) and allocate it\n", worksize/1024/1024);
    /**************************
     * Create Contraction Plan
     **************************/

    cutensorContractionPlan_t plan;
    CUDA_CHECK(cutensorInitContractionPlan(&handle,
                 &plan,
                 &desc,
                 &find,
                 worksize));

    if (printDebug) printf("Create plan for contraction\n");
    /**********************
     * Run
     **********************/

    cutensorStatus_t err;
    const int algoToTry = allAlgos ? 6 : -5; // only try default patient
    for (int algo = (int) CUTENSOR_ALGO_DEFAULT_PATIENT; algo < algoToTry; algo++) {
        printf("Algo: %d\n", algo);
        if (worksizePref == 4) {
            for (int worksize_iter = 1; worksize_iter <= (int) CUTENSOR_WORKSPACE_MAX; worksize_iter++) {
                double minTimeCUTENSOR = 1e100;
                double avTime = 0;
                CUDA_CHECK(cutensorContractionGetWorkspaceSize(&handle,
                            &desc,
                            &find,
                            (cutensorWorksizePreference_t) worksize_iter, &worksize));
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
                    if (checkRMSE && cublasFlag) {
                        floatTypeA rmse_diff = rmse(M*N, C_d, C_cublas);
                        printf("RMSE = %f\t", rmse_diff);
                    } 

                    printf("  Worksize: %zu MB\t\t%.2f GB/s %.2f TFLOP/s\n", worksize/1024/1024, transferedBytes / avTime, tflops / avTime);
                }

                if (bestTime > minTimeCUTENSOR)
                {
                    bestTime = minTimeCUTENSOR;
                    bestAlgo = algo;
                }


            }
        } else {
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
                if (checkRMSE && cublasFlag) {
                    floatTypeA rmse_diff = rmse(M*N, C_d, C_cublas);
                    printf("RMSE = %f\t", rmse_diff);
                } 

                printf("cuTensor: %d algo %.2f GB/s %.2f TFLOP/s\n", algo, transferedBytes / avTime, tflops / avTime);
            }

            if (bestTime > minTimeCUTENSOR)
            {
                bestTime = minTimeCUTENSOR;
                bestAlgo = algo;
            }

        }

    }

    // floatTypeA rmse_diff = rmse(j*k, C_d, C_cublas);
    // printf("RMSE = %f\n", rmse_diff);

    if (printDebug) printf("Execute contraction from plan\n");
    /*************************/

    printf("\nRESULTS from %d runs:\n", runs);
    if (cublasFlag) {
        printf("Time cuBLAS[s]: Best: %.4f; Mean %.4f\n",
                min_time_cublas, av_time_cublas);
        printf("CUBLAS:\t\t%.2f GB/s;  %.2f TFLOP/s\n",
                transferedBytes / av_time_cublas, tflops/ av_time_cublas);
    }
    printf("CUTENSOR:\t%.2f GB/s;  %.2f TFLOP/s\t(algo %d)\n", transferedBytes / bestTime, tflops / bestTime, bestAlgo);

    if (A) free(A);
    if (B) free(B);
    if (C) free(C);
    if (A_d) cudaFree(A_d);
    if (B_d) cudaFree(B_d);
    if (C_d) cudaFree(C_d);
    if (work) cudaFree(work);
    if (C_cublas) cudaFree(C_cublas);

    return 0;
}

int findCommonIndices(std::string& A_string, std::string& C_string) {
    int M = 0;
    for (size_t i = 0; i < A_string.length(); i++) {
        for (size_t j = 0; j < C_string.length(); j++) {
            if (A_string[i] == C_string[j]) {
                M++;
            }
        }
    }
    return M;
}