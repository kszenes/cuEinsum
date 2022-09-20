
#include <cuda_runtime.h>
#include <dace/dace.h>

#include "../include/dace_cutensor.h"

struct mttkrp_order_3_mode_1_compute_t {
    dace::cuda::Context *gpu_context;
    dace::linalg::CuTensorHandle cutensor_handle;
};



DACE_EXPORTED int __dace_init_cuda(mttkrp_order_3_mode_1_compute_t *__state, int R, int S0, int S1, int S2);
DACE_EXPORTED void __dace_exit_cuda(mttkrp_order_3_mode_1_compute_t *__state);

DACE_DFI void mttkrp_order_3_mode_1_compute_204_4_0_0_2(const double * __tmp_206_25_r, const double * __tmp_206_40_r, double * __tmp_206_12_w, int R, int S0, int S1, long long a, long long j) {
    long long i;


    for (i = 0; (i < S0); i = i + 1) {
        {
            double __tmp4[1]  DACE_ALIGN(64);

            {
                double __in1 = __tmp_206_25_r[((R * S1) * i)];
                double __in2 = __tmp_206_40_r[(R * i)];
                double __out;

                ///////////////////
                // Tasklet code (_Mult_)
                __out = (__in1 * __in2);
                ///////////////////

                __tmp4[0] = __out;
            }
            {
                double __in1 = __tmp_206_12_w[0];
                double __in2 = __tmp4[0];
                double __out;

                ///////////////////
                // Tasklet code (augassign_206_12)
                __out = (__in1 + __in2);
                ///////////////////

                __tmp_206_12_w[0] = __out;
            }

        }

    }
    
}



int __dace_init_cuda(mttkrp_order_3_mode_1_compute_t *__state, int R, int S0, int S1, int S2) {
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

    __state->gpu_context = new dace::cuda::Context(4, 1);

    // Create cuda streams and events
    for(int i = 0; i < 4; ++i) {
        cudaStreamCreateWithFlags(&__state->gpu_context->streams[i], cudaStreamNonBlocking);
    }
    for(int i = 0; i < 1; ++i) {
        cudaEventCreateWithFlags(&__state->gpu_context->events[i], cudaEventDisableTiming);
    }

    

    return 0;
}

void __dace_exit_cuda(mttkrp_order_3_mode_1_compute_t *__state) {
    

    // Destroy cuda streams and events
    for(int i = 0; i < 4; ++i) {
        cudaStreamDestroy(__state->gpu_context->streams[i]);
    }
    for(int i = 0; i < 1; ++i) {
        cudaEventDestroy(__state->gpu_context->events[i]);
    }

    delete __state->gpu_context;
}

__global__ void assign_203_4_map_0_1_7(double * __restrict__ gpu_out, int R, int S1) {
    {
        {
            int __i1 = (blockIdx.x * 32 + threadIdx.x);
            int __i0 = (blockIdx.y * 1 + threadIdx.y);
            if (__i1 < R) {
                {
                    {
                        double __out;

                        ///////////////////
                        // Tasklet code (assign_203_4)
                        __out = 0;
                        ///////////////////

                        gpu_out[((R * __i0) + __i1)] = __out;
                    }
                }
            }
        }
    }
}


DACE_EXPORTED void __dace_runkernel_assign_203_4_map_0_1_7(mttkrp_order_3_mode_1_compute_t *__state, double * __restrict__ gpu_out, int R, int S1);
void __dace_runkernel_assign_203_4_map_0_1_7(mttkrp_order_3_mode_1_compute_t *__state, double * __restrict__ gpu_out, int R, int S1)
{

    void  *assign_203_4_map_0_1_7_args[] = { (void *)&gpu_out, (void *)&R, (void *)&S1 };
    cudaLaunchKernel((void*)assign_203_4_map_0_1_7, dim3(int_ceil(int_ceil(R, 1), 32), int_ceil(int_ceil(S1, 1), 1), 1), dim3(32, 1, 1), assign_203_4_map_0_1_7_args, 0, __state->gpu_context->streams[0]);
}
__global__ void mttkrp_order_3_mode_1_compute_204_0_0_0(const double * __restrict__ gpu_IM, double * __restrict__ gpu_out, const double * __restrict__ tmp, int R, int S0, int S1) {
    {
        {
            int a = (blockIdx.x * 32 + threadIdx.x);
            int j = (blockIdx.y * 1 + threadIdx.y);
            if (a < R) {
                {
                    mttkrp_order_3_mode_1_compute_204_4_0_0_2(&tmp[((R * j) + a)], &gpu_IM[a], &gpu_out[((R * j) + a)], R, S0, S1, a, j);
                }
            }
        }
    }
}


DACE_EXPORTED void __dace_runkernel_mttkrp_order_3_mode_1_compute_204_0_0_0(mttkrp_order_3_mode_1_compute_t *__state, const double * __restrict__ gpu_IM, double * __restrict__ gpu_out, const double * __restrict__ tmp, int R, int S0, int S1);
void __dace_runkernel_mttkrp_order_3_mode_1_compute_204_0_0_0(mttkrp_order_3_mode_1_compute_t *__state, const double * __restrict__ gpu_IM, double * __restrict__ gpu_out, const double * __restrict__ tmp, int R, int S0, int S1)
{

    void  *mttkrp_order_3_mode_1_compute_204_0_0_0_args[] = { (void *)&gpu_IM, (void *)&gpu_out, (void *)&tmp, (void *)&R, (void *)&S0, (void *)&S1 };
    cudaLaunchKernel((void*)mttkrp_order_3_mode_1_compute_204_0_0_0, dim3(int_ceil(int_ceil(R, 1), 32), int_ceil(int_ceil(S1, 1), 1), 1), dim3(32, 1, 1), mttkrp_order_3_mode_1_compute_204_0_0_0_args, 0, __state->gpu_context->streams[0]);
}

