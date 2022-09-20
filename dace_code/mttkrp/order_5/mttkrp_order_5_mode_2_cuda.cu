
#include <cuda_runtime.h>
#include <dace/dace.h>

#include "../include/cuda_mpi_interop.h"
#include "../include/dace_cutensor.h"

struct mttkrp_order_5_mode_2_t {
    dace::cuda::Context *gpu_context;
    MPI_Comm __pgrid_0_comm;
    MPI_Group __pgrid_0_group;
    int __pgrid_0_coords[6];
    int __pgrid_0_dims[6];
    int __pgrid_0_rank;
    int __pgrid_0_size;
    bool __pgrid_0_valid;
    MPI_Comm __pgrid_1_comm;
    MPI_Group __pgrid_1_group;
    int __pgrid_1_coords[4];
    int __pgrid_1_dims[4];
    int __pgrid_1_rank;
    int __pgrid_1_size;
    bool __pgrid_1_valid;
    dace::linalg::CuTensorHandle cutensor_handle;
};



DACE_EXPORTED int __dace_init_cuda(mttkrp_order_5_mode_2_t *__state, int P0, int P1, int P2, int P3, int P4, int PR, int R, int S0, int S1, int S2, int S3, int S4);
DACE_EXPORTED void __dace_exit_cuda(mttkrp_order_5_mode_2_t *__state);

DACE_DFI void mttkrp_order_5_mode_2_216_4_0_1_2(const double * __tmp_219_29_r, const double * __tmp_219_48_r, double * __tmp_219_16_w, int R, int S0, int S1, int S2, long long a, long long k) {
    long long i;
    long long j;


    for (i = 0; (i < S0); i = i + 1) {

        for (j = 0; (j < S1); j = j + 1) {
            {
                double __tmp6[1]  DACE_ALIGN(64);

                {
                    double __in1 = __tmp_219_29_r[((((R * S1) * S2) * i) + ((R * S2) * j))];
                    double __in2 = __tmp_219_48_r[(((R * S1) * i) + (R * j))];
                    double __out;

                    ///////////////////
                    // Tasklet code (_Mult_)
                    __out = (__in1 * __in2);
                    ///////////////////

                    __tmp6[0] = __out;
                }
                {
                    double __in1 = __tmp_219_16_w[0];
                    double __in2 = __tmp6[0];
                    double __out;

                    ///////////////////
                    // Tasklet code (augassign_219_16)
                    __out = (__in1 + __in2);
                    ///////////////////

                    __tmp_219_16_w[0] = __out;
                }

            }

        }

    }
    
}



int __dace_init_cuda(mttkrp_order_5_mode_2_t *__state, int P0, int P1, int P2, int P3, int P4, int PR, int R, int S0, int S1, int S2, int S3, int S4) {
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

    __state->gpu_context = new dace::cuda::Context(6, 1);

    // Create cuda streams and events
    for(int i = 0; i < 6; ++i) {
        cudaStreamCreateWithFlags(&__state->gpu_context->streams[i], cudaStreamNonBlocking);
    }
    for(int i = 0; i < 1; ++i) {
        cudaEventCreateWithFlags(&__state->gpu_context->events[i], cudaEventDisableTiming);
    }

    

    return 0;
}

void __dace_exit_cuda(mttkrp_order_5_mode_2_t *__state) {
    

    // Destroy cuda streams and events
    for(int i = 0; i < 6; ++i) {
        cudaStreamDestroy(__state->gpu_context->streams[i]);
    }
    for(int i = 0; i < 1; ++i) {
        cudaEventDestroy(__state->gpu_context->events[i]);
    }

    delete __state->gpu_context;
}

__global__ void mttkrp_order_5_mode_2_202_0_0_2(const double * __restrict__ gpu_LM, const double * __restrict__ gpu_MM, double * __restrict__ tmp, int R, int S3, int S4) {
    {
        {
            {
                int a = (blockIdx.x * 32 + threadIdx.x);
                int m = (blockIdx.y * 1 + threadIdx.y);
                int l = (blockIdx.z * 1 + threadIdx.z);
                double __tmp2;
                if (a < R) {
                    {
                        {
                            {
                                double __in1 = gpu_LM[((R * l) + a)];
                                double __in2 = gpu_MM[((R * m) + a)];
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
                                // Tasklet code (assign_203_8)
                                __out = __inp;
                                ///////////////////

                                tmp[((((R * S4) * l) + (R * m)) + a)] = __out;
                            }
                        }
                    }
                }
            }
        }
    }
}


DACE_EXPORTED void __dace_runkernel_mttkrp_order_5_mode_2_202_0_0_2(mttkrp_order_5_mode_2_t *__state, const double * __restrict__ gpu_LM, const double * __restrict__ gpu_MM, double * __restrict__ tmp, int R, int S3, int S4);
void __dace_runkernel_mttkrp_order_5_mode_2_202_0_0_2(mttkrp_order_5_mode_2_t *__state, const double * __restrict__ gpu_LM, const double * __restrict__ gpu_MM, double * __restrict__ tmp, int R, int S3, int S4)
{

    void  *mttkrp_order_5_mode_2_202_0_0_2_args[] = { (void *)&gpu_LM, (void *)&gpu_MM, (void *)&tmp, (void *)&R, (void *)&S3, (void *)&S4 };
    cudaLaunchKernel((void*)mttkrp_order_5_mode_2_202_0_0_2, dim3(int_ceil(int_ceil(R, 1), 32), int_ceil(int_ceil(S4, 1), 1), int_ceil(int_ceil(S3, 1), 1)), dim3(32, 1, 1), mttkrp_order_5_mode_2_202_0_0_2_args, 0, __state->gpu_context->streams[3]);
}
__global__ void mttkrp_order_5_mode_2_210_0_0_9(const double * __restrict__ gpu_IM, const double * __restrict__ gpu_JM, double * __restrict__ tmp3, int R, int S0, int S1) {
    {
        {
            {
                int a = (blockIdx.x * 32 + threadIdx.x);
                int j = (blockIdx.y * 1 + threadIdx.y);
                int i = (blockIdx.z * 1 + threadIdx.z);
                double __tmp4;
                if (a < R) {
                    {
                        {
                            {
                                double __in1 = gpu_IM[((R * i) + a)];
                                double __in2 = gpu_JM[((R * j) + a)];
                                double __out;

                                ///////////////////
                                // Tasklet code (_Mult_)
                                __out = (__in1 * __in2);
                                ///////////////////

                                __tmp4 = __out;
                            }
                            {
                                double __inp = __tmp4;
                                double __out;

                                ///////////////////
                                // Tasklet code (assign_211_8)
                                __out = __inp;
                                ///////////////////

                                tmp3[((((R * S1) * i) + (R * j)) + a)] = __out;
                            }
                        }
                    }
                }
            }
        }
    }
}


DACE_EXPORTED void __dace_runkernel_mttkrp_order_5_mode_2_210_0_0_9(mttkrp_order_5_mode_2_t *__state, const double * __restrict__ gpu_IM, const double * __restrict__ gpu_JM, double * __restrict__ tmp3, int R, int S0, int S1);
void __dace_runkernel_mttkrp_order_5_mode_2_210_0_0_9(mttkrp_order_5_mode_2_t *__state, const double * __restrict__ gpu_IM, const double * __restrict__ gpu_JM, double * __restrict__ tmp3, int R, int S0, int S1)
{

    void  *mttkrp_order_5_mode_2_210_0_0_9_args[] = { (void *)&gpu_IM, (void *)&gpu_JM, (void *)&tmp3, (void *)&R, (void *)&S0, (void *)&S1 };
    cudaLaunchKernel((void*)mttkrp_order_5_mode_2_210_0_0_9, dim3(int_ceil(int_ceil(R, 1), 32), int_ceil(int_ceil(S1, 1), 1), int_ceil(int_ceil(S0, 1), 1)), dim3(32, 1, 1), mttkrp_order_5_mode_2_210_0_0_9_args, 0, __state->gpu_context->streams[4]);
}
__global__ void assign_215_4_map_0_0_14(double * __restrict__ gpu_out, int R, int S2) {
    {
        {
            int __i1 = (blockIdx.x * 32 + threadIdx.x);
            int __i0 = (blockIdx.y * 1 + threadIdx.y);
            if (__i1 < R) {
                {
                    {
                        double __out;

                        ///////////////////
                        // Tasklet code (assign_215_4)
                        __out = 0;
                        ///////////////////

                        gpu_out[((R * __i0) + __i1)] = __out;
                    }
                }
            }
        }
    }
}


DACE_EXPORTED void __dace_runkernel_assign_215_4_map_0_0_14(mttkrp_order_5_mode_2_t *__state, double * __restrict__ gpu_out, int R, int S2);
void __dace_runkernel_assign_215_4_map_0_0_14(mttkrp_order_5_mode_2_t *__state, double * __restrict__ gpu_out, int R, int S2)
{

    void  *assign_215_4_map_0_0_14_args[] = { (void *)&gpu_out, (void *)&R, (void *)&S2 };
    cudaLaunchKernel((void*)assign_215_4_map_0_0_14, dim3(int_ceil(int_ceil(R, 1), 32), int_ceil(int_ceil(S2, 1), 1), 1), dim3(32, 1, 1), assign_215_4_map_0_0_14_args, 0, __state->gpu_context->streams[0]);
}
__global__ void mttkrp_order_5_mode_2_216_0_1_0(double * __restrict__ gpu_out, const double * __restrict__ tmp2, const double * __restrict__ tmp3, int R, int S0, int S1, int S2) {
    {
        {
            int a = (blockIdx.x * 32 + threadIdx.x);
            int k = (blockIdx.y * 1 + threadIdx.y);
            if (a < R) {
                {
                    mttkrp_order_5_mode_2_216_4_0_1_2(&tmp2[((R * k) + a)], &tmp3[a], &gpu_out[((R * k) + a)], R, S0, S1, S2, a, k);
                }
            }
        }
    }
}


DACE_EXPORTED void __dace_runkernel_mttkrp_order_5_mode_2_216_0_1_0(mttkrp_order_5_mode_2_t *__state, double * __restrict__ gpu_out, const double * __restrict__ tmp2, const double * __restrict__ tmp3, int R, int S0, int S1, int S2);
void __dace_runkernel_mttkrp_order_5_mode_2_216_0_1_0(mttkrp_order_5_mode_2_t *__state, double * __restrict__ gpu_out, const double * __restrict__ tmp2, const double * __restrict__ tmp3, int R, int S0, int S1, int S2)
{

    void  *mttkrp_order_5_mode_2_216_0_1_0_args[] = { (void *)&gpu_out, (void *)&tmp2, (void *)&tmp3, (void *)&R, (void *)&S0, (void *)&S1, (void *)&S2 };
    cudaLaunchKernel((void*)mttkrp_order_5_mode_2_216_0_1_0, dim3(int_ceil(int_ceil(R, 1), 32), int_ceil(int_ceil(S2, 1), 1), 1), dim3(32, 1, 1), mttkrp_order_5_mode_2_216_0_1_0_args, 0, __state->gpu_context->streams[0]);
}

