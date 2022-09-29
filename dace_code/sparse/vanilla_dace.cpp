/* DaCe AUTO-GENERATED FILE. DO NOT MODIFY */
#include <dace/dace.h>
#include "../../include/hash.h"
#include "mkl.h"
#include "mkl_spblas.h"
#include "mkl.h"
#include "../include/dace_blas.h"
#include "mpi.h"

struct vanilla_dace_t {
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
};

inline void vanilla_dace_115_4_118_8_0_0_27(vanilla_dace_t *__state, const double& __tmp_119_24_r, double* __tmp_119_34_r, const int& __tmp_119_37_r, const double& __tmp_119_55_r, double& __tmp_119_12_w, int LAcols, int LHcols, long long i, int j, long long k) {
    int __sym___tmp_119_37_r;

    __sym___tmp_119_37_r = __tmp_119_37_r;
    {
        double __tmp7[1]  DACE_ALIGN(64);
        double __tmp8[1]  DACE_ALIGN(64);

        {
            double __in1 = __tmp_119_24_r;
            double __in2 = __tmp_119_34_r[((LHcols * __sym___tmp_119_37_r) + k)];
            double __out;

            ///////////////////
            // Tasklet code (_Mult_)
            __out = (__in1 * __in2);
            ///////////////////

            __tmp7[0] = __out;
        }
        {
            double __in2 = __tmp_119_55_r;
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

            __tmp_119_12_w = __out;
        }

    }
    
}

void __program_vanilla_dace_internal(vanilla_dace_t *__state, int * __restrict__ A_colidx, double * __restrict__ A_data, int * __restrict__ A_rowptr, double * __restrict__ H1, double * __restrict__ H2, double * __restrict__ W, double * __restrict__ __return, int LAcols, int LAnnz, int LArows, int LHcols, int LWcols, int Px, int Py)
{

    {
        int parent_grid;
        int reduce_grid;
        double *HW;
        HW = new double DACE_ALIGN(64)[(LAcols * LWcols)];
        double *values;
        values = new double DACE_ALIGN(64)[LAnnz];
        double *out;
        out = new double DACE_ALIGN(64)[(LArows * LWcols)];

        {
            double* _a = &H2[0];
            double* _b = &W[0];
            double* _c = HW;

            ///////////////////
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, LWcols, LAcols, LHcols, double(1.0), _b, LWcols, _a, LHcols, double(0.0), _c, LWcols);
            ///////////////////

        }
        {
            #pragma omp parallel for
            for (auto __i0 = 0; __i0 < LAnnz; __i0 += 1) {
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
        {
            #pragma omp parallel for
            for (auto i = 0; i < LArows; i += 1) {
                int start;
                int finish;
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
                            vanilla_dace_115_4_118_8_0_0_27(__state, H1[((LHcols * i) + k)], &H2[0], A_colidx[j], values[j], values[j], LAcols, LHcols, i, j, k);
                        }
                    }
                }
            }
        }
        {
            int* _a_cols = &A_colidx[0];
            double* _a_vals = &values[0];
            int* _a_rows = &A_rowptr[0];
            double* _b = &HW[0];
            double* _c = out;

            ///////////////////

            sparse_matrix_t __csrA;
            mkl_sparse_d_create_csr(&__csrA, SPARSE_INDEX_BASE_ZERO, LArows, LAcols, _a_rows, _a_rows + 1, _a_cols, _a_vals);
            struct matrix_descr __descrA;
            __descrA.type = SPARSE_MATRIX_TYPE_GENERAL;
            __descrA.mode = SPARSE_FILL_MODE_UPPER;
            __descrA.diag = SPARSE_DIAG_NON_UNIT;

            mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, double(1), __csrA, __descrA, SPARSE_LAYOUT_ROW_MAJOR, _b, LWcols, LWcols, double(0), _c, LWcols);

            ///////////////////

        }
        {
            double* _inbuffer = &out[0];
            double* _outbuffer = out;

            ///////////////////
            if (__state->__pgrid_1_size > 1) {
                MPI_Allreduce(MPI_IN_PLACE, _outbuffer, LArows*LWcols, MPI_DOUBLE, MPI_SUM, __state->__pgrid_1_comm);
            }
            ///////////////////

        }
        {
            #pragma omp parallel for
            for (auto __i0 = 0; __i0 < LArows; __i0 += 1) {
                for (auto __i1 = 0; __i1 < LWcols; __i1 += 1) {
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
        delete[] HW;
        delete[] values;
        delete[] out;

    }
}

DACE_EXPORTED void __program_vanilla_dace(vanilla_dace_t *__state, int * __restrict__ A_colidx, double * __restrict__ A_data, int * __restrict__ A_rowptr, double * __restrict__ H1, double * __restrict__ H2, double * __restrict__ W, double * __restrict__ __return, int LAcols, int LAnnz, int LArows, int LHcols, int LWcols, int Px, int Py)
{
    __program_vanilla_dace_internal(__state, A_colidx, A_data, A_rowptr, H1, H2, W, __return, LAcols, LAnnz, LArows, LHcols, LWcols, Px, Py);
}

DACE_EXPORTED vanilla_dace_t *__dace_init_vanilla_dace(int LAcols, int LAnnz, int LArows, int LHcols, int LWcols, int Px, int Py)
{
    int __result = 0;
    vanilla_dace_t *__state = new vanilla_dace_t;


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

DACE_EXPORTED void __dace_exit_vanilla_dace(vanilla_dace_t *__state)
{

    if (__state->__pgrid_0_valid) {
        MPI_Group_free(&__state->__pgrid_0_group);
        MPI_Comm_free(&__state->__pgrid_0_comm);
    }

    if (__state->__pgrid_1_valid) {
        MPI_Group_free(&__state->__pgrid_1_group);
        MPI_Comm_free(&__state->__pgrid_1_comm);
    }

    {  // Environment: MPI
        // MPI_Finalize();
    }
    delete __state;
}

