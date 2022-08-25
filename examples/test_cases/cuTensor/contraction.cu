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


int main()
{
    #define TENSOR
    const int runs = 1;
    const int worksizePref = 3; // 0: 0[Mib]; 1: MIN; 2: RECOMMENDED; 3: MAX
    const bool printDebug = false;
    const bool allAlgos = false;
    // const bool cublasFlag = true;
    // const bool checkRMSE = false;

    printf("Workspace preference: ");
    switch (worksizePref) {
        case 0: printf("0 [MB]\n"); break;
        case 1: printf("MIN\n"); break;
        case 2: printf("RECOMMENDED\n"); break;
        case 3: printf("MAX\n"); break;
        default: printf("Unsupported worksizePref: %d", worksizePref); exit(-1);
    }

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
    // cublasComputeType_t cublasComputeType = CUBLAS_COMPUTE_32F;
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
    // cublasComputeType_t cublasComputeType = CUBLAS_COMPUTE_32F_FAST_TF32;
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
    // cublasComputeType_t cublasComputeType = CUBLAS_COMPUTE_64F;
    #undef DOUBLE
    #endif

    floatTypeCompute alpha = (floatTypeCompute)1.7f;
    floatTypeCompute beta  = (floatTypeCompute)0.f;

    if (printDebug) printf("Include headers and define data types\n");

    /**********************
     * Computing: C_{m,u,n,v} = alpha * A_{m,h,k,n} B_{u,k,v,h} + beta * C_{m,u,n,v}
     **********************/

    std::vector<int> modeC{'m','u','n','v'};
    std::vector<int> modeA{'m','h','k','n'};
    std::vector<int> modeB{'u','k','v','h'};
    int nmodeA = modeA.size();
    int nmodeB = modeB.size();
    int nmodeC = modeC.size();

    std::unordered_map<int, int64_t> extent;
    extent['m'] = 96;
    extent['n'] = 96;
    extent['u'] = 2 * 96;
    extent['v'] = 64;
    extent['h'] = 4 * 64;
    extent['k'] = 64;

    // double tflops = (2.0 * extent['m'] * extent['n'] * extent['u'] * extent['v'] * extent['h'] * extent['k']) /1e12;

    /******************
       ijn, jmk -> inkm
    ******************/

    // std::vector<int> modeC{'i','j',,'n'};
    // std::vector<int> modeA{'j','m','k'};
    // std::vector<int> modeB{'i','n','k','m'};
    // int nmodeA = modeA.size();
    // int nmodeB = modeB.size();
    // int nmodeC = modeC.size();

    // std::unordered_map<int, int64_t> extent;
    // extent['i'] = 255;
    // extent['j'] = 127;
    // extent['k'] = 129;
    // extent['n'] = 65;
    // extent['m'] = 62;
    // extent['l'] = 63;

    // double tflops = (2.0 * extent['i'] * extent['j'] * extent['k'] * extent['n'] * extent['m'], extent['l']) /1e12;

    /**************
      MATMUL
    **************/
    // std::vector<int> modeA{'i','j'};
    // std::vector<int> modeB{'j','k'};
    // std::vector<int> modeC{'i','k'};
    // int nmodeA = modeA.size();
    // int nmodeB = modeB.size();
    // int nmodeC = modeC.size();

    // std::unordered_map<int, int64_t> extent;
    // const int size = 1 << 14;
    // const int i = size;
    // const int j = size;
    // const int k = size;

    // extent['i'] = i;
    // extent['j'] = j;
    // extent['k'] = k;

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

    printf("Size [MiB]: A = %zu ; B = %zu; C = %zu\n", sizeA/1024/1024, sizeB/1024/1024, sizeC/1024/1024);
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

    for (size_t i = 0; i < elementsA; i++)
        A[i] = (((floatTypeA) rand())/RAND_MAX - 0.5)*100;
    for (size_t i = 0; i < elementsB; i++)
        B[i] = (((floatTypeB) rand())/RAND_MAX - 0.5)*100;
    for (size_t i = 0; i < elementsC; i++)
        C[i] = (((floatTypeC) rand())/RAND_MAX - 0.5)*100;

    CUDA_CHECK(cudaMemcpy(A_d, A, sizeA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d, B, sizeB, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(C_d, C, sizeC, cudaMemcpyHostToDevice));



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
        CUDA_CHECK(cutensorContractionGetWorkspaceSize(&handle,
                    &desc,
                    &find,
                    (cutensorWorksizePreference_t) worksizePref, &worksize));
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
            // if (checkRMSE && cublasFlag) {
            //     floatTypeA rmse_diff = rmse(i*j*l, C_d, C_cublas);
            //     printf("RMSE = %f\t", rmse_diff);

            // } 
            printf("cuTensor: %d algo %.2f GB/s %.2f TFLOP/s\n", algo, transferedBytes / minTimeCUTENSOR, tflops / avTime);
        }

        if (bestTime > minTimeCUTENSOR)
        {
            bestTime = minTimeCUTENSOR;
            bestAlgo = algo;
        }

    }

    // floatTypeA rmse_diff = rmse(j*k, C_d, C_cublas);
    // printf("RMSE = %f\n", rmse_diff);

    if (printDebug) printf("Execute contraction from plan\n");
    /*************************/

    printf("\nRESULTS from %d runs:\n", runs);
    printf("CUTENSOR best: %d algo %.2f GB/s %.2f TFLOP/s\n", bestAlgo, transferedBytes / bestTime, tflops / bestTime);

    if (A) free(A);
    if (B) free(B);
    if (C) free(C);
    if (A_d) cudaFree(A_d);
    if (B_d) cudaFree(B_d);
    if (C_d) cudaFree(C_d);
    if (work) cudaFree(work);

    return 0;
}
