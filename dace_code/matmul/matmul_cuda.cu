
#include <cuda_runtime.h>
#include <dace/dace.h>


struct matmul_t {
    dace::cuda::Context *gpu_context;
};



DACE_EXPORTED int __dace_init_cuda(matmul_t *__state, int K, int M, int N);
DACE_EXPORTED void __dace_exit_cuda(matmul_t *__state);

DACE_DFI void nested_MapState_0_0_6(const double * gpu_A, const double * gpu_B, double * gpu_C, int K, int M, int N) {
    __shared__ double trans_gpu_A[((128 * (63 / 8)) + 128)];
    __shared__ double trans_gpu_B[((64 * (63 / 4)) + 64)];
    double trans_trans_gpu_A[8]  DACE_ALIGN(64);
    double trans_trans_gpu_B[4]  DACE_ALIGN(64);
    double trans_gpu_C[32]  DACE_ALIGN(64) = {0};
    int tile_k;

    {

        dace::GlobalToShared2D<double, 16, 8, 1, ((8 * (63 / 8)) + 8), 8, 8, 1, true>(gpu_A, K, 1, trans_gpu_A);
        dace::GlobalToShared2D<double, 16, 8, 1, 8, ((4 * (63 / 4)) + 4), ((4 * (63 / 4)) + 4), 1, true>(gpu_B, N, 1, trans_gpu_B);

    }


    for (tile_k = 0; (tile_k < (K - 8)); tile_k = tile_k + 8) {
        {

            {
                {
                    {
                        __syncthreads();
                        int tile1_j = (4 * threadIdx.x);
                        int tile1_i = (8 * threadIdx.y);
                        {
                            {
                                {
                                    for (auto k = 0; k < 8; k += 1) {

                                        dace::CopyND<double, 1, false, 8>::template ConstDst<1>::Copy(
                                        trans_gpu_A + ((k + (8 * tile1_i)) + (((64 * (63 / 8)) + 64) * ((tile_k / 8) % 2))), trans_trans_gpu_A, 8);

                                        dace::CopyND<double, 1, false, 4>::template ConstDst<1>::Copy(
                                        trans_gpu_B + (((k * ((4 * (63 / 4)) + 4)) + tile1_j) + (((32 * (63 / 4)) + 32) * ((tile_k / 8) % 2))), trans_trans_gpu_B, 1);
                                        {
                                            #pragma unroll
                                            for (auto i = 0; i < 8; i += 1) {
                                                #pragma unroll
                                                for (auto j = 0; j < 4; j += 1) {
                                                    {
                                                        double in_A = trans_trans_gpu_A[i];
                                                        double in_B = trans_trans_gpu_B[j];
                                                        double out;

                                                        ///////////////////
                                                        // Tasklet code (matmul_40)
                                                        out = (in_A * in_B);
                                                        ///////////////////

                                                        dace::wcr_fixed<dace::ReductionType::Sum, double>::reduce(trans_gpu_C + ((4 * i) + j), out);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            dace::GlobalToShared2D<double, 16, 8, 1, ((8 * (63 / 8)) + 8), 8, 8, 1, true>(gpu_A + (tile_k + 8), K, 1, trans_gpu_A + (((64 * (63 / 8)) + 64) * (((tile_k / 8) + 1) % 2)));
            dace::GlobalToShared2D<double, 16, 8, 1, 8, ((4 * (63 / 4)) + 4), ((4 * (63 / 4)) + 4), 1, true>(gpu_B + (N * (tile_k + 8)), N, 1, trans_gpu_B + (((32 * (63 / 4)) + 32) * (((tile_k / 8) + 1) % 2)));

        }

    }
    {

        {
            {
                {
                    __syncthreads();
                    int tile1_j = (4 * threadIdx.x);
                    int tile1_i = (8 * threadIdx.y);
                    {
                        {
                            {
                                for (auto k = 0; k < 8; k += 1) {

                                    dace::CopyND<double, 1, false, 8>::template ConstDst<1>::Copy(
                                    trans_gpu_A + ((k + (8 * tile1_i)) + (((64 * (63 / 8)) + 64) * ((tile_k / 8) % 2))), trans_trans_gpu_A, 8);

                                    dace::CopyND<double, 1, false, 4>::template ConstDst<1>::Copy(
                                    trans_gpu_B + (((k * ((4 * (63 / 4)) + 4)) + tile1_j) + (((32 * (63 / 4)) + 32) * ((tile_k / 8) % 2))), trans_trans_gpu_B, 1);
                                    {
                                        #pragma unroll
                                        for (auto i = 0; i < 8; i += 1) {
                                            #pragma unroll
                                            for (auto j = 0; j < 4; j += 1) {
                                                {
                                                    double in_A = trans_trans_gpu_A[i];
                                                    double in_B = trans_trans_gpu_B[j];
                                                    double out;

                                                    ///////////////////
                                                    // Tasklet code (matmul_40)
                                                    out = (in_A * in_B);
                                                    ///////////////////

                                                    dace::wcr_fixed<dace::ReductionType::Sum, double>::reduce(trans_gpu_C + ((4 * i) + j), out);
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            dace::CopyND<double, 1, false, 8, 4>::template ConstSrc<4, 1>::Accumulate(
                            trans_gpu_C, gpu_C + ((N * tile1_i) + tile1_j), [] (const double& a, const double& b) { return (a + b); }, N, 1);
                        }
                    }
                }
            }
        }

    }
    
}



int __dace_init_cuda(matmul_t *__state, int K, int M, int N) {
    int count;

    // Check that we are able to run cuda code
    if (cudaGetDeviceCount(&count) != cudaSuccess)
    {
        printf("ERROR: GPU drivers are not configured or cuda-capable device "
               "not found\n");
        return 1;
    }
    if (count == 0)
    {
        printf("ERROR: No cuda-capable devices found\n");
        return 2;
    }

    // Initialize cuda before we run the application
    float *dev_X;
    cudaMalloc((void **) &dev_X, 1);
    cudaFree(dev_X);

    

    __state->gpu_context = new dace::cuda::Context(3, 1);

    // Create cuda streams and events
    for(int i = 0; i < 3; ++i) {
        cudaStreamCreateWithFlags(&__state->gpu_context->streams[i], cudaStreamNonBlocking);
    }
    for(int i = 0; i < 1; ++i) {
        cudaEventCreateWithFlags(&__state->gpu_context->events[i], cudaEventDisableTiming);
    }

    

    return 0;
}

void __dace_exit_cuda(matmul_t *__state) {
    

    // Destroy cuda streams and events
    for(int i = 0; i < 3; ++i) {
        cudaStreamDestroy(__state->gpu_context->streams[i]);
    }
    for(int i = 0; i < 1; ++i) {
        cudaEventDestroy(__state->gpu_context->events[i]);
    }

    delete __state->gpu_context;
}

__global__ void freduce_init_map_0_1_0(double * __restrict__ gpu_C, int M, int N) {
    {
        {
            int o1 = (blockIdx.x * 32 + threadIdx.x);
            int o0 = (blockIdx.y * 1 + threadIdx.y);
            if (o1 < N) {
                {
                    {
                        double __out;

                        ///////////////////
                        // Tasklet code (freduce_init)
                        __out = 0;
                        ///////////////////

                        gpu_C[((N * o0) + o1)] = __out;
                    }
                }
            }
        }
    }
}


DACE_EXPORTED void __dace_runkernel_freduce_init_map_0_1_0(matmul_t *__state, double * __restrict__ gpu_C, int M, int N);
void __dace_runkernel_freduce_init_map_0_1_0(matmul_t *__state, double * __restrict__ gpu_C, int M, int N)
{

    void  *freduce_init_map_0_1_0_args[] = { (void *)&gpu_C, (void *)&M, (void *)&N };
    cudaLaunchKernel((void*)freduce_init_map_0_1_0, dim3(int_ceil(int_ceil(N, 1), 32), int_ceil(int_ceil(M, 1), 1), 1), dim3(32, 1, 1), freduce_init_map_0_1_0_args, 0, __state->gpu_context->streams[0]);
}
__global__ void matmul_39_0_0_4(const double * __restrict__ gpu_A, const double * __restrict__ gpu_B, double * __restrict__ gpu_C, int K, int M, int N) {
    {
        {
            int tile_j = (64 * blockIdx.x);
            int tile_i = (64 * blockIdx.y);
            nested_MapState_0_0_6(&gpu_A[(K * tile_i)], &gpu_B[tile_j], &gpu_C[((N * tile_i) + tile_j)], K, M, N);
        }
    }
}


DACE_EXPORTED void __dace_runkernel_matmul_39_0_0_4(matmul_t *__state, const double * __restrict__ gpu_A, const double * __restrict__ gpu_B, double * __restrict__ gpu_C, int K, int M, int N);
void __dace_runkernel_matmul_39_0_0_4(matmul_t *__state, const double * __restrict__ gpu_A, const double * __restrict__ gpu_B, double * __restrict__ gpu_C, int K, int M, int N)
{

    void  *matmul_39_0_0_4_args[] = { (void *)&gpu_A, (void *)&gpu_B, (void *)&gpu_C, (void *)&K, (void *)&M, (void *)&N };
    cudaLaunchKernel((void*)matmul_39_0_0_4, dim3(int_ceil(N, 64), int_ceil(M, 64), 1), dim3(16, 8, 1), matmul_39_0_0_4_args, 0, __state->gpu_context->streams[0]);
}

