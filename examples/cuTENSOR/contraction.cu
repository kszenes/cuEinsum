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

double rmse(const int, const double*, const double*);

#define HANDLE_ERROR(x)                                               \
{ const auto err = x;                                                 \
  if( err != CUTENSOR_STATUS_SUCCESS )                                \
  { printf("Error: %s\n", cutensorGetErrorString(err)); return err; } \
};

#define HANDLE_CUDA_ERROR(x)                                      \
{ const auto err = x;                                             \
  if( err != cudaSuccess )                                        \
  { printf("Error: %s\n", cudaGetErrorString(err)); return err; } \
};

struct GPUTimer
{
    GPUTimer() 
    {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
        cudaEventRecord(start_, 0);
    }

    ~GPUTimer() 
    {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    void start() 
    {
        cudaEventRecord(start_, 0);
    }

    float seconds() 
    {
        cudaEventRecord(stop_, 0);
        cudaEventSynchronize(stop_);
        float time;
        cudaEventElapsedTime(&time, start_, stop_);
        return time * 1e-3;
    }
    private:
    cudaEvent_t start_, stop_;
};

int main()
{
    // --- Single precision ---
    // typedef float floatTypeA;
    // typedef float floatTypeB;
    // typedef float floatTypeC;
    // typedef float floatTypeCompute;

    // cudaDataType_t typeA = CUDA_R_32F;
    // cudaDataType_t typeB = CUDA_R_32F;
    // cudaDataType_t typeC = CUDA_R_32F;
    // cutensorComputeType_t typeCompute = CUTENSOR_COMPUTE_32F;

    // --- Double precision ---
    typedef double floatTypeA;
    typedef double floatTypeB;
    typedef double floatTypeC;
    typedef double floatTypeCompute;

    cudaDataType_t typeA = CUDA_R_64F;
    cudaDataType_t typeB = CUDA_R_64F;
    cudaDataType_t typeC = CUDA_R_64F;
    cutensorComputeType_t typeCompute = CUTENSOR_COMPUTE_64F;
    // --- END ---

    floatTypeCompute alpha = (floatTypeCompute)0.f;
    floatTypeCompute beta  = (floatTypeCompute)2.3f;

    printf("Include headers and define data types\n");

    /**********************
     * Computing: C_{m,u,n,v} = alpha * A_{m,h,k,n} B_{u,k,v,h} + beta * C_{m,u,n,v}
     **********************/

    /* std::vector<int> modeC{'m','u','n','v'}; */
    /* std::vector<int> modeA{'m','h','k','n'}; */
    /* std::vector<int> modeB{'u','k','v','h'}; */
    /* int nmodeA = modeA.size(); */
    /* int nmodeB = modeB.size(); */
    /* int nmodeC = modeC.size(); */

    /* std::unordered_map<int, int64_t> extent; */
    /* extent['m'] = 96; */
    /* extent['n'] = 96; */
    /* extent['u'] = 96; */
    /* extent['v'] = 64; */
    /* extent['h'] = 64; */
    /* extent['k'] = 64; */

    /* extent['m'] = 255; */
    /* extent['n'] = 127; */
    /* extent['u'] = 129; */
    /* extent['v'] = 65; */
    /* extent['h'] = 62; */
    /* extent['k'] = 63; */

    /* double tflops = (2.0 * extent['m'] * extent['n'] * extent['u'] * extent['v'] * extent['k'] * extent['h']) /1e12; */

    /**************
      MATMUL
    **************/
    std::vector<int> modeC{'i','k'};
    std::vector<int> modeA{'i','j'};
    std::vector<int> modeB{'j','k'};
    int nmodeA = modeA.size();
    int nmodeB = modeB.size();
    int nmodeC = modeC.size();

    std::unordered_map<int, int64_t> extent;
    const int i = 1 << 13;
    const int j = 1 << 11;
    const int k = 1 << 12;

    extent['i'] = i;
    extent['j'] = j;
    extent['k'] = k;


    // computes FLOPS
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
    printf("Total memory: %.2f GiB\n", (sizeA + sizeB + sizeC)/1024./1024./1024);

    floatTypeA *A_d, *B_d, *C_d;
    HANDLE_CUDA_ERROR(cudaMalloc((void**) &A_d, sizeA));
    HANDLE_CUDA_ERROR(cudaMalloc((void**) &B_d, sizeB));
    HANDLE_CUDA_ERROR(cudaMalloc((void**) &C_d, sizeC));


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

    for (int64_t i = 0; i < elementsA; i++)
        A[i] = (((floatTypeA) rand())/RAND_MAX - 0.5)*100;
    for (int64_t i = 0; i < elementsB; i++)
        B[i] = (((floatTypeB) rand())/RAND_MAX - 0.5)*100;
    for (int64_t i = 0; i < elementsC; i++)
        C[i] = (((floatTypeC) rand())/RAND_MAX - 0.5)*100;

    HANDLE_CUDA_ERROR(cudaMemcpy(A_d, A, sizeA, cudaMemcpyHostToDevice));
    HANDLE_CUDA_ERROR(cudaMemcpy(B_d, B, sizeB, cudaMemcpyHostToDevice));
    HANDLE_CUDA_ERROR(cudaMemcpy(C_d, C, sizeC, cudaMemcpyHostToDevice));


    /*************
     * Compute DGEMM CUBLAS
     ************/
    floatTypeC *C_cublas;
    HANDLE_CUDA_ERROR(cudaMalloc((void**) &C_cublas, sizeC));
    HANDLE_CUDA_ERROR(cudaMemcpy(C_cublas, C, sizeC, cudaMemcpyHostToDevice));

    cudaStream_t s;
    cublasHandle_t cublas_handle;
    HANDLE_CUDA_ERROR(cudaStreamCreate(&s));
    CUDA_CHECK(cublasCreate(&cublas_handle));

    GPUTimer timer_cublas;
    timer_cublas.start();
    CUDA_CHECK(cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, i, j, k, &alpha, A_d, i, B_d, k, &beta, C_cublas, i));
    auto time_cublas = timer_cublas.seconds();


    /*************************
     * cuTENSOR
     *************************/ 

    cutensorHandle_t handle;
    HANDLE_ERROR(cutensorInit(&handle));
    /**********************
     * Create Tensor Descriptors
     **********************/

    cutensorTensorDescriptor_t descA;
    HANDLE_ERROR(cutensorInitTensorDescriptor(&handle,
                 &descA,
                 nmodeA,
                 extentA.data(),
                 NULL,/*stride*/
                 typeA, CUTENSOR_OP_IDENTITY));

    cutensorTensorDescriptor_t descB;
    HANDLE_ERROR(cutensorInitTensorDescriptor(&handle,
                 &descB,
                 nmodeB,
                 extentB.data(),
                 NULL,/*stride*/
                 typeB, CUTENSOR_OP_IDENTITY));

    cutensorTensorDescriptor_t descC;
    HANDLE_ERROR(cutensorInitTensorDescriptor( &handle,
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
     HANDLE_ERROR(cutensorGetAlignmentRequirement(&handle,
                  A_d,
                  &descA,
                  &alignmentRequirementA));

     uint32_t alignmentRequirementB;
     HANDLE_ERROR(cutensorGetAlignmentRequirement(&handle,
                  B_d,
                  &descB,
                  &alignmentRequirementB));

     uint32_t alignmentRequirementC;
     HANDLE_ERROR(cutensorGetAlignmentRequirement(&handle,
                  C_d,
                  &descC, 
                  &alignmentRequirementC));

    printf("Query best alignment requirement for our pointers\n");
    /*******************************
     * Create Contraction Descriptor
     *******************************/

    cutensorContractionDescriptor_t desc;
    HANDLE_ERROR(cutensorInitContractionDescriptor(&handle, 
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
    HANDLE_ERROR(cutensorInitContractionFind( 
                 &handle, &find, 
                 CUTENSOR_ALGO_DEFAULT));

    printf("Initialize settings to find algorithm\n");
    /**********************
     * Query workspace
     **********************/

    uint64_t worksize = 0;
    HANDLE_ERROR(cutensorContractionGetWorkspace(&handle,
                 &desc,
                 &find,
                 CUTENSOR_WORKSPACE_RECOMMENDED, &worksize));

    void *work = nullptr;
    if (worksize > 0)
    {
        if (cudaSuccess != cudaMalloc(&work, worksize))
        {
            work = nullptr;
            worksize = 0;
        }
    } 

    printf("Query recommended workspace size and allocate it\n");
    /**************************
     * Create Contraction Plan
     **************************/

    cutensorContractionPlan_t plan;
    HANDLE_ERROR(cutensorInitContractionPlan(&handle,
                 &plan,
                 &desc,
                 &find,
                 worksize));

    printf("Create plan for contraction\n");
    /**********************
     * Run
     **********************/

    double minTimeCUTENSOR = 1e100;
    double avTime = 0;
    const int runs = 1;
    cutensorStatus_t err;
    for (int i=0; i < runs; ++i)
    {
        cudaMemcpy(C_d, C, sizeC, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();

        // Set up timing
        GPUTimer timer;
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

    double rmse_diff = rmse(j*k, C_d, C_cublas);
    printf("RMSE = %f\n", rmse_diff);

    printf("Execute contraction from plan\n");
    /*************************/

    double transferedBytes = sizeC + sizeA + sizeB;
    transferedBytes += ((float) beta != 0.f) ? sizeC : 0;
    transferedBytes /= 1e9;
    printf("\nRESULTS from %d runs:\n", runs);
    printf("Time cuTensor[s]: Best: %.4f; Mean %.4f\n",
            minTimeCUTENSOR, avTime);
    printf("Time cuBLAS[s]: Best: %.4f; Mean %.4f\n",
            time_cublas, time_cublas);
    printf("Compute [GFLOPS/s]: Best: %.4f;  Mean %.4f\n",
            tflops / minTimeCUTENSOR, tflops/ avTime);
    printf("Memory [GB/s]: Best: %.2f;  Mean: %.2f\n",
            transferedBytes / minTimeCUTENSOR, transferedBytes / avTime);

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

double rmse(const int n, const double* dVec1, const double* dVec2)
{
  double* hVec1;
  double* hVec2;
  HANDLE_CUDA_ERROR(cudaMallocHost(&hVec1, n*sizeof(double)));
  HANDLE_CUDA_ERROR(cudaMallocHost(&hVec2, n*sizeof(double)));
  
  HANDLE_CUDA_ERROR(cudaMemcpy(hVec1, dVec1, n*sizeof(double), cudaMemcpyDeviceToHost));
  HANDLE_CUDA_ERROR(cudaMemcpy(hVec2, dVec2, n*sizeof(double), cudaMemcpyDeviceToHost));

  double rmse = 0.;
  for (int i(0); i < n; ++i)
  {
    double diff = hVec1[i] - hVec2[i];
    rmse += (diff*diff);
  }

  HANDLE_CUDA_ERROR(cudaFreeHost(hVec1));
  HANDLE_CUDA_ERROR(cudaFreeHost(hVec2));

  return std::sqrt(rmse/n);
}