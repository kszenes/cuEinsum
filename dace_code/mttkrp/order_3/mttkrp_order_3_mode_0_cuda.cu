
#include <cuda_runtime.h>
#include <dace/dace.h>

#include "../include/cuda_mpi_interop.h"
#include "../include/dace_cutensor.h"

struct mttkrp_order_3_mode_0_t {
    dace::cuda::Context *gpu_context;
    MPI_Comm __pgrid_0_comm;
    MPI_Group __pgrid_0_group;
    int __pgrid_0_coords[4];
    int __pgrid_0_dims[4];
    int __pgrid_0_rank;
    int __pgrid_0_size;
    bool __pgrid_0_valid;
    MPI_Comm __pgrid_1_comm;
    MPI_Group __pgrid_1_group;
    int __pgrid_1_coords[2];
    int __pgrid_1_dims[2];
    int __pgrid_1_rank;
    int __pgrid_1_size;
    bool __pgrid_1_valid;
    dace::linalg::CuTensorHandle cutensor_handle;
};



DACE_EXPORTED int __dace_init_cuda(mttkrp_order_3_mode_0_t *__state, int P0, int P1, int P2, int PR, int R, int S0, int S1, int S2);
DACE_EXPORTED void __dace_exit_cuda(mttkrp_order_3_mode_0_t *__state);



int __dace_init_cuda(mttkrp_order_3_mode_0_t *__state, int P0, int P1, int P2, int PR, int R, int S0, int S1, int S2) {
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

void __dace_exit_cuda(mttkrp_order_3_mode_0_t *__state) {
    

    // Destroy cuda streams and events
    for(int i = 0; i < 4; ++i) {
        cudaStreamDestroy(__state->gpu_context->streams[i]);
    }
    for(int i = 0; i < 1; ++i) {
        cudaEventDestroy(__state->gpu_context->events[i]);
    }

    delete __state->gpu_context;
}

__global__ void mttkrp_order_3_mode_0_132_0_0_2(const double * __restrict__ gpu_JM, const double * __restrict__ gpu_KM, double * __restrict__ tmp, int R, int S1, int S2) {
    {
        {
            {
                int a = (blockIdx.x * 32 + threadIdx.x);
                int k = (blockIdx.y * 1 + threadIdx.y);
                int j = (blockIdx.z * 1 + threadIdx.z);
                double __tmp2;
                if (a < R) {
                    {
                        {
                            {
                                double __in1 = gpu_JM[((R * j) + a)];
                                double __in2 = gpu_KM[((R * k) + a)];
                                double __out;

                                ///////////////////
                                // Tasklet code (_Mult_)
                                __out = (__in1 * __in2);
                                ///////////////////

                                __tmp2 = __out;
                            }
                            {
                                double __inp = __tmp2;
                                double __out;

                                ///////////////////
                                // Tasklet code (assign_133_8)
                                __out = __inp;
                                ///////////////////

                                tmp[((((R * S2) * j) + (R * k)) + a)] = __out;
                            }
                        }
                    }
                }
            }
        }
    }
}


DACE_EXPORTED void __dace_runkernel_mttkrp_order_3_mode_0_132_0_0_2(mttkrp_order_3_mode_0_t *__state, const double * __restrict__ gpu_JM, const double * __restrict__ gpu_KM, double * __restrict__ tmp, int R, int S1, int S2);
void __dace_runkernel_mttkrp_order_3_mode_0_132_0_0_2(mttkrp_order_3_mode_0_t *__state, const double * __restrict__ gpu_JM, const double * __restrict__ gpu_KM, double * __restrict__ tmp, int R, int S1, int S2)
{

    void  *mttkrp_order_3_mode_0_132_0_0_2_args[] = { (void *)&gpu_JM, (void *)&gpu_KM, (void *)&tmp, (void *)&R, (void *)&S1, (void *)&S2 };
    cudaLaunchKernel((void*)mttkrp_order_3_mode_0_132_0_0_2, dim3(int_ceil(int_ceil(R, 1), 32), int_ceil(int_ceil(S2, 1), 1), int_ceil(int_ceil(S1, 1), 1)), dim3(32, 1, 1), mttkrp_order_3_mode_0_132_0_0_2_args, 0, __state->gpu_context->streams[2]);
}

