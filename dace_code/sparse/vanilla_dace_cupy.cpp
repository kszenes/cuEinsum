/* DaCe AUTO-GENERATED FILE. DO NOT MODIFY */
#include <dace/dace.h>
#include "../../include/hash.h"
#include "../include/dace_cublas.h"
#include "mpi.h"
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

DACE_EXPORTED void __dace_runkernel__numpy_full__map_0_0_5(vanilla_dace_cupy_t *__state, double * __restrict__ values, int LAnnz);
DACE_EXPORTED void __dace_runkernel_vanilla_dace_115_0_0_9(vanilla_dace_cupy_t *__state, const int * __restrict__ A_colidx, const int * __restrict__ A_rowptr, const double * __restrict__ H1, const double * __restrict__ H2, double * __restrict__ values, int LAcols, int LAnnz, int LArows, int LHcols);
DACE_EXPORTED void __dace_runkernel__numpy_maximum__map_0_0_17(vanilla_dace_cupy_t *__state, double * __restrict__ __return, const double * __restrict__ out, int LArows, int LWcols);
void __program_vanilla_dace_cupy_internal(vanilla_dace_cupy_t *__state, int * __restrict__ A_colidx, double * __restrict__ A_data, int * __restrict__ A_rowptr, double * __restrict__ H1, double * __restrict__ H2, double * __restrict__ W, double * __restrict__ __return, int LAcols, int LAnnz, int LArows, int LHcols, int LWcols, int Px, int Py)
{
    double * HW;
    cudaMalloc((void**)&HW, (LAcols * LWcols) * sizeof(double));
    double * values;
    cudaMalloc((void**)&values, LAnnz * sizeof(double));
    double * out;
    cudaMalloc((void**)&out, (LArows * LWcols) * sizeof(double));

    {
        int parent_grid;
        int reduce_grid;

        {
            double* _a = &H2[0];
            double* _b = &W[0];
            double* _c = HW;

            ///////////////////
            int __dace_current_stream_id = 4;
            cudaStream_t __dace_current_stream = __state->gpu_context->streams[__dace_current_stream_id];
            const int __dace_cuda_device = 0;
            cublasHandle_t &__dace_cublas_handle = __state->cublas_handle.Get(__dace_cuda_device);
            cublasSetStream(__dace_cublas_handle, __dace_current_stream);
            cublasDgemm(__dace_cublas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            LWcols, LAcols, LHcols,
            __state->cublas_handle.Constants(__dace_cuda_device).DoublePone(),
            (double*)_b, LWcols,
            (double*)_a, LHcols,
            __state->cublas_handle.Constants(__dace_cuda_device).DoubleZero(),
            (double*)_c, LWcols);
            cudaEventRecord(__state->gpu_context->events[2], __state->gpu_context->streams[4]);
            cudaStreamWaitEvent(__state->gpu_context->streams[3], __state->gpu_context->events[2], 0);
            ///////////////////

        }
        __dace_runkernel__numpy_full__map_0_0_5(__state, values, LAnnz);
        cudaEventRecord(__state->gpu_context->events[1], __state->gpu_context->streams[0]);
        cudaStreamWaitEvent(__state->gpu_context->streams[3], __state->gpu_context->events[1], 0);
        __dace_runkernel_vanilla_dace_115_0_0_9(__state, A_colidx, A_rowptr, H1, H2, values, LAcols, LAnnz, LArows, LHcols);
        {
            int* _a_cols = &A_colidx[0];
            double * _a_vals = &values[0];
            int* _a_rows = &A_rowptr[0];
            double * _b = &HW[0];
            double* _c = out;

            ///////////////////
            int __dace_current_stream_id = 3;
            cudaStream_t __dace_current_stream = __state->gpu_context->streams[__dace_current_stream_id];
            const int __dace_cuda_device = 0;
            cusparseHandle_t &__dace_cusparse_handle = __state->cusparse_handle.Get(__dace_cuda_device);
            cusparseSetStream(__dace_cusparse_handle, __dace_current_stream);
            cusparseSetPointerMode(__dace_cusparse_handle, CUSPARSE_POINTER_MODE_HOST);
            double alpha = double(1);
            double beta = double(0);

            cusparseSpMatDescr_t matA;
            cusparseDnMatDescr_t matB, matC;
            void*                dBuffer    = NULL;
            size_t               bufferSize = 0;
            // Create sparse matrix A in CSR format
            dace::sparse::CheckCusparseError( cusparseCreateCsr(&matA, LArows, LAcols, LAnnz,
            _a_rows, _a_cols, _a_vals,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F) );
            // Create dense matrix B
            dace::sparse::CheckCusparseError( cusparseCreateDnMat(&matB, LAcols, LWcols, LWcols, _b,
            CUDA_R_64F, CUSPARSE_ORDER_ROW) );
            // Create dense matrix C
            dace::sparse::CheckCusparseError( cusparseCreateDnMat(&matC, LArows, LWcols, LWcols, _c,
            CUDA_R_64F, CUSPARSE_ORDER_ROW) );
            // allocate an external buffer if needed
            dace::sparse::CheckCusparseError( cusparseSpMM_bufferSize(
            __dace_cusparse_handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            (double *)&alpha, matA, matB, (double *)&beta, matC, CUDA_R_64F,
            CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize) );
            cudaMalloc(&dBuffer, bufferSize);

            // execute SpMM
            dace::sparse::CheckCusparseError( cusparseSpMM(__dace_cusparse_handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            (double *)&alpha, matA, matB, (double *)&beta, matC, CUDA_R_64F,
            CUSPARSE_SPMM_ALG_DEFAULT, dBuffer) );

            // destroy matrix/vector descriptors
            dace::sparse::CheckCusparseError( cusparseDestroySpMat(matA) );
            dace::sparse::CheckCusparseError( cusparseDestroyDnMat(matB) );
            dace::sparse::CheckCusparseError( cusparseDestroyDnMat(matC) );
            cudaFree(dBuffer);
            cusparseSetPointerMode(__dace_cusparse_handle, CUSPARSE_POINTER_MODE_DEVICE);
            ///////////////////

        }
        {
            double * _inbuffer = &out[0];
            double* _outbuffer = out;

            ///////////////////
            int __dace_current_stream_id = 3;
            cudaStream_t __dace_current_stream = __state->gpu_context->streams[__dace_current_stream_id];
            if (__state->__pgrid_1_size > 1) {
                MPI_Allreduce(MPI_IN_PLACE, _outbuffer, LArows*LWcols, MPI_DOUBLE, MPI_SUM, __state->__pgrid_1_comm);
            }
            ///////////////////

        }
        __dace_runkernel__numpy_maximum__map_0_0_17(__state, __return, out, LArows, LWcols);
        {
            int __out;

            ///////////////////
            // Tasklet code (__pgrid_0)
            ///////////////////

            parent_grid = __out;
        }
        {
            int __out;

            ///////////////////
            // Tasklet code (__pgrid_1)
            ///////////////////

            reduce_grid = __out;
        }
        cudaStreamSynchronize(__state->gpu_context->streams[3]);


    }
    cudaFree(HW);
    cudaFree(values);
    cudaFree(out);
}

DACE_EXPORTED void __program_vanilla_dace_cupy(vanilla_dace_cupy_t *__state, int * __restrict__ A_colidx, double * __restrict__ A_data, int * __restrict__ A_rowptr, double * __restrict__ H1, double * __restrict__ H2, double * __restrict__ W, double * __restrict__ __return, int LAcols, int LAnnz, int LArows, int LHcols, int LWcols, int Px, int Py)
{
    __program_vanilla_dace_cupy_internal(__state, A_colidx, A_data, A_rowptr, H1, H2, W, __return, LAcols, LAnnz, LArows, LHcols, LWcols, Px, Py);
}
DACE_EXPORTED int __dace_init_cuda(vanilla_dace_cupy_t *__state, int LAcols, int LAnnz, int LArows, int LHcols, int LWcols, int Px, int Py);
DACE_EXPORTED int __dace_exit_cuda(vanilla_dace_cupy_t *__state);

DACE_EXPORTED vanilla_dace_cupy_t *__dace_init_vanilla_dace_cupy(int LAcols, int LAnnz, int LArows, int LHcols, int LWcols, int Px, int Py)
{
    int __result = 0;
    vanilla_dace_cupy_t *__state = new vanilla_dace_cupy_t;


    __result |= __dace_init_cuda(__state, LAcols, LAnnz, LArows, LHcols, LWcols, Px, Py);
    {  // Environment: MPI
        int t; MPI_Initialized(&t);  if (!t) MPI_Init(NULL, NULL);
    }
    __state->__pgrid_0_dims[0] = Px;
    __state->__pgrid_0_dims[1] = Py;

    int __pgrid_0_periods[2] = {0};
    MPI_Cart_create(MPI_COMM_WORLD, 2, __state->__pgrid_0_dims, __pgrid_0_periods, 0, &__state->__pgrid_0_comm);
    if (__state->__pgrid_0_comm != MPI_COMM_NULL) {
        MPI_Comm_group(__state->__pgrid_0_comm, &__state->__pgrid_0_group);
        MPI_Comm_rank(__state->__pgrid_0_comm, &__state->__pgrid_0_rank);
        MPI_Comm_size(__state->__pgrid_0_comm, &__state->__pgrid_0_size);
        MPI_Cart_coords(__state->__pgrid_0_comm, __state->__pgrid_0_rank, 2, __state->__pgrid_0_coords);
        __state->__pgrid_0_valid = true;
    } else {
        __state->__pgrid_0_group = MPI_GROUP_NULL;
        __state->__pgrid_0_rank = MPI_PROC_NULL;
        __state->__pgrid_0_size = 0;
        __state->__pgrid_0_valid = false;
    }
    __state->__pgrid_1_dims[0] = Py;

    __state->__pgrid_1_valid = false;
    if (__state->__pgrid_0_valid) {
        int __pgrid_1_remain[2] = {0, 1};
        MPI_Cart_sub(__state->__pgrid_0_comm, __pgrid_1_remain, &__state->__pgrid_1_comm);
        MPI_Comm_group(__state->__pgrid_1_comm, &__state->__pgrid_1_group);
        MPI_Comm_rank(__state->__pgrid_1_comm, &__state->__pgrid_1_rank);
        MPI_Comm_size(__state->__pgrid_1_comm, &__state->__pgrid_1_size);
        MPI_Cart_coords(__state->__pgrid_1_comm, __state->__pgrid_1_rank, 1, __state->__pgrid_1_coords);

        __state->__pgrid_1_valid = true;
    }


    if (__result) {
        delete __state;
        return nullptr;
    }
    return __state;
}

DACE_EXPORTED void __dace_exit_vanilla_dace_cupy(vanilla_dace_cupy_t *__state)
{

    if (__state->__pgrid_0_valid) {
        MPI_Group_free(&__state->__pgrid_0_group);
        MPI_Comm_free(&__state->__pgrid_0_comm);
    }

    if (__state->__pgrid_1_valid) {
        MPI_Group_free(&__state->__pgrid_1_group);
        MPI_Comm_free(&__state->__pgrid_1_comm);
    }

    __dace_exit_cuda(__state);
    {  // Environment: MPI
        // MPI_Finalize();
    }
    delete __state;
}

