
#include <cuda_runtime.h>
#include <dace/dace.h>

#include "../include/dace_cublas.h"
#include "../include/cuda_mpi_interop.h"
#include "../include/dace_cusparse.h"

struct vanilla_dace_cupy_t {
    dace::cuda::Context *gpu_context;
    MPI_Comm __pgrid_0_comm;
    MPI_Group __pgrid_0_group;
    int __pgrid_0_coords[2];
    int __pgrid_0_dims[2];
    int __pgrid_0_rank;
    int __pgrid_0_size;
    bool __pgrid_0_valid;
    MPI_Comm __pgrid_1_comm;
    MPI_Group __pgrid_1_group;
    int __pgrid_1_coords[1];
    int __pgrid_1_dims[1];
    int __pgrid_1_rank;
    int __pgrid_1_size;
    bool __pgrid_1_valid;
    dace::blas::CublasHandle cublas_handle;
    dace::sparse::CusparseHandle cusparse_handle;
};



DACE_EXPORTED int __dace_init_cuda(vanilla_dace_cupy_t *__state, int LAcols, int LAnnz, int LArows, int LHcols, int LWcols, int Px, int Py);
DACE_EXPORTED void __dace_exit_cuda(vanilla_dace_cupy_t *__state);

DACE_DFI void vanilla_dace_115_4_118_8_0_0_27(const double* __tmp_119_24_r, const double* __tmp_119_34_r, const int* __tmp_119_37_r, double * __tmp_119_55_r, double * __tmp_119_12_w, int LAcols, int LHcols, long long i, int j, long long k) {
    int __sym___tmp_119_37_r;

    __sym___tmp_119_37_r = __tmp_119_37_r[0];
    {
        double __tmp7[1]  DACE_ALIGN(64);
        double __tmp8[1]  DACE_ALIGN(64);

        {
            double __in1 = __tmp_119_24_r[0];
            double __in2 = __tmp_119_34_r[((LHcols * __sym___tmp_119_37_r) + k)];
            double __out;

            ///////////////////
            // Tasklet code (_Mult_)
            __out = (__in1 * __in2);
            ///////////////////

            __tmp7[0] = __out;
        }
        {
            double __in2 = __tmp_119_55_r[0];
            double __in1 = __tmp7[0];
            double __out;

            ///////////////////
            // Tasklet code (_Add_)
            __out = (__in1 + __in2);
            ///////////////////

            __tmp8[0] = __out;
        }
        {
            double __inp = __tmp8[0];
            double __out;

            ///////////////////
            // Tasklet code (assign_119_12)
            __out = __inp;
            ///////////////////

            __tmp_119_12_w[0] = __out;
        }

    }
    
}



int __dace_init_cuda(vanilla_dace_cupy_t *__state, int LAcols, int LAnnz, int LArows, int LHcols, int LWcols, int Px, int Py) {
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

    __state->gpu_context = new dace::cuda::Context(5, 3);

    // Create cuda streams and events
    for(int i = 0; i < 5; ++i) {
        cudaStreamCreateWithFlags(&__state->gpu_context->streams[i], cudaStreamNonBlocking);
    }
    for(int i = 0; i < 3; ++i) {
        cudaEventCreateWithFlags(&__state->gpu_context->events[i], cudaEventDisableTiming);
    }

    

    return 0;
}

void __dace_exit_cuda(vanilla_dace_cupy_t *__state) {
    

    // Destroy cuda streams and events
    for(int i = 0; i < 5; ++i) {
        cudaStreamDestroy(__state->gpu_context->streams[i]);
    }
    for(int i = 0; i < 3; ++i) {
        cudaEventDestroy(__state->gpu_context->events[i]);
    }

    delete __state->gpu_context;
}

__global__ void _numpy_full__map_0_0_5(double * __restrict__ values, int LAnnz) {
    {
        int __i0 = (blockIdx.x * 32 + threadIdx.x);
        if (__i0 < LAnnz) {
            {
                double __out;

                ///////////////////
                // Tasklet code (_numpy_full_)
                __out = 0.0;
                ///////////////////

                values[__i0] = __out;
            }
        }
    }
}


DACE_EXPORTED void __dace_runkernel__numpy_full__map_0_0_5(vanilla_dace_cupy_t *__state, double * __restrict__ values, int LAnnz);
void __dace_runkernel__numpy_full__map_0_0_5(vanilla_dace_cupy_t *__state, double * __restrict__ values, int LAnnz)
{

    void  *_numpy_full__map_0_0_5_args[] = { (void *)&values, (void *)&LAnnz };
    cudaLaunchKernel((void*)_numpy_full__map_0_0_5, dim3(int_ceil(int_ceil(LAnnz, 1), 32), 1, 1), dim3(32, 1, 1), _numpy_full__map_0_0_5_args, 0, __state->gpu_context->streams[0]);
}
__global__ void vanilla_dace_115_0_0_9(const int * __restrict__ A_colidx, const int * __restrict__ A_rowptr, const double * __restrict__ H1, const double * __restrict__ H2, double * __restrict__ values, int LAcols, int LAnnz, int LArows, int LHcols) {
    {
        int i = (blockIdx.x * 32 + threadIdx.x);
        int start;
        int finish;
        if (i < LArows) {
            {
                int __inp = A_rowptr[i];
                int __out;

                ///////////////////
                // Tasklet code (assign_116_8)
                __out = __inp;
                ///////////////////

                start = __out;
            }
            {
                int __inp = A_rowptr[(i + 1)];
                int __out;

                ///////////////////
                // Tasklet code (assign_117_8)
                __out = __inp;
                ///////////////////

                finish = __out;
            }
            {
                int __map_118_b0 = start;
                int __map_118_e1 = finish;
                for (auto k = 0; k < LHcols; k += 1) {
                    for (auto j = __map_118_b0; j < __map_118_e1; j += 1) {
                        vanilla_dace_115_4_118_8_0_0_27(&H1[((LHcols * i) + k)], &H2[0], &A_colidx[j], &values[j], &values[j], LAcols, LHcols, i, j, k);
                    }
                }
            }
        }
    }
}


DACE_EXPORTED void __dace_runkernel_vanilla_dace_115_0_0_9(vanilla_dace_cupy_t *__state, const int * __restrict__ A_colidx, const int * __restrict__ A_rowptr, const double * __restrict__ H1, const double * __restrict__ H2, double * __restrict__ values, int LAcols, int LAnnz, int LArows, int LHcols);
void __dace_runkernel_vanilla_dace_115_0_0_9(vanilla_dace_cupy_t *__state, const int * __restrict__ A_colidx, const int * __restrict__ A_rowptr, const double * __restrict__ H1, const double * __restrict__ H2, double * __restrict__ values, int LAcols, int LAnnz, int LArows, int LHcols)
{

    void  *vanilla_dace_115_0_0_9_args[] = { (void *)&A_colidx, (void *)&A_rowptr, (void *)&H1, (void *)&H2, (void *)&values, (void *)&LAcols, (void *)&LAnnz, (void *)&LArows, (void *)&LHcols };
    cudaLaunchKernel((void*)vanilla_dace_115_0_0_9, dim3(int_ceil(int_ceil(LArows, 1), 32), 1, 1), dim3(32, 1, 1), vanilla_dace_115_0_0_9_args, 0, __state->gpu_context->streams[3]);
}
__global__ void _numpy_maximum__map_0_0_17(double * __restrict__ __return, const double * __restrict__ out, int LArows, int LWcols) {
    {
        {
            int __i1 = (blockIdx.x * 32 + threadIdx.x);
            int __i0 = (blockIdx.y * 1 + threadIdx.y);
            if (__i1 < LWcols) {
                {
                    {
                        double __in1 = out[((LWcols * __i0) + __i1)];
                        double __out;

                        ///////////////////
                        // Tasklet code (_numpy_maximum_)
                        __out = max(__in1, dace::float64(0));
                        ///////////////////

                        __return[((LWcols * __i0) + __i1)] = __out;
                    }
                }
            }
        }
    }
}


DACE_EXPORTED void __dace_runkernel__numpy_maximum__map_0_0_17(vanilla_dace_cupy_t *__state, double * __restrict__ __return, const double * __restrict__ out, int LArows, int LWcols);
void __dace_runkernel__numpy_maximum__map_0_0_17(vanilla_dace_cupy_t *__state, double * __restrict__ __return, const double * __restrict__ out, int LArows, int LWcols)
{

    void  *_numpy_maximum__map_0_0_17_args[] = { (void *)&__return, (void *)&out, (void *)&LArows, (void *)&LWcols };
    cudaLaunchKernel((void*)_numpy_maximum__map_0_0_17, dim3(int_ceil(int_ceil(LWcols, 1), 32), int_ceil(int_ceil(LArows, 1), 1), 1), dim3(32, 1, 1), _numpy_maximum__map_0_0_17_args, 0, __state->gpu_context->streams[3]);
}

