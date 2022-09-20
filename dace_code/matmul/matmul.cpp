/* DaCe AUTO-GENERATED FILE. DO NOT MODIFY */
#include <dace/dace.h>
#include "../../include/hash.h"

struct matmul_t {
    dace::cuda::Context *gpu_context;
};

DACE_EXPORTED void __dace_runkernel_freduce_init_map_0_1_0(matmul_t *__state, double * __restrict__ gpu_C, int M, int N);
DACE_EXPORTED void __dace_runkernel_matmul_39_0_0_4(matmul_t *__state, const double * __restrict__ gpu_A, const double * __restrict__ gpu_B, double * __restrict__ gpu_C, int K, int M, int N);
void __program_matmul_internal(matmul_t *__state, double * __restrict__ A, double * __restrict__ B, double * __restrict__ C, int K, int M, int N)
{
    double * gpu_A;
    cudaMalloc((void**)&gpu_A, (K * M) * sizeof(double));
    double * gpu_B;
    cudaMalloc((void**)&gpu_B, (K * N) * sizeof(double));
    double * gpu_C;
    cudaMalloc((void**)&gpu_C, (M * N) * sizeof(double));

    {

        cudaMemcpyAsync(gpu_A, A, (K * M) * sizeof(double), cudaMemcpyHostToDevice, __state->gpu_context->streams[0]);
        cudaMemcpyAsync(gpu_B, B, (K * N) * sizeof(double), cudaMemcpyHostToDevice, __state->gpu_context->streams[1]);
        cudaMemcpyAsync(gpu_C, C, (M * N) * sizeof(double), cudaMemcpyHostToDevice, __state->gpu_context->streams[2]);
        cudaStreamSynchronize(__state->gpu_context->streams[0]);
        cudaStreamSynchronize(__state->gpu_context->streams[1]);
        cudaStreamSynchronize(__state->gpu_context->streams[2]);


    }
    {

        __dace_runkernel_freduce_init_map_0_1_0(__state, gpu_C, M, N);


    }
    {

        __dace_runkernel_matmul_39_0_0_4(__state, gpu_A, gpu_B, gpu_C, K, M, N);
        cudaMemcpyAsync(C, gpu_C, (M * N) * sizeof(double), cudaMemcpyDeviceToHost, __state->gpu_context->streams[0]);
        cudaStreamSynchronize(__state->gpu_context->streams[0]);


    }
    cudaFree(gpu_A);
    cudaFree(gpu_B);
    cudaFree(gpu_C);
}

DACE_EXPORTED void __program_matmul(matmul_t *__state, double * __restrict__ A, double * __restrict__ B, double * __restrict__ C, int K, int M, int N)
{
    __program_matmul_internal(__state, A, B, C, K, M, N);
}
DACE_EXPORTED int __dace_init_cuda(matmul_t *__state, int K, int M, int N);
DACE_EXPORTED int __dace_exit_cuda(matmul_t *__state);

DACE_EXPORTED matmul_t *__dace_init_matmul(int K, int M, int N)
{
    int __result = 0;
    matmul_t *__state = new matmul_t;


    __result |= __dace_init_cuda(__state, K, M, N);

    if (__result) {
        delete __state;
        return nullptr;
    }
    return __state;
}

DACE_EXPORTED void __dace_exit_matmul(matmul_t *__state)
{
    __dace_exit_cuda(__state);
    delete __state;
}

