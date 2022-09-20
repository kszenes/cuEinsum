/* DaCe AUTO-GENERATED FILE. DO NOT MODIFY */
#include <dace/dace.h>
#include "../../include/hash.h"
#include "mpi.h"
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

DACE_EXPORTED void __dace_runkernel_mttkrp_order_5_mode_2_202_0_0_2(mttkrp_order_5_mode_2_t *__state, const double * __restrict__ gpu_LM, const double * __restrict__ gpu_MM, double * __restrict__ tmp, int R, int S3, int S4);
DACE_EXPORTED void __dace_runkernel_mttkrp_order_5_mode_2_210_0_0_9(mttkrp_order_5_mode_2_t *__state, const double * __restrict__ gpu_IM, const double * __restrict__ gpu_JM, double * __restrict__ tmp3, int R, int S0, int S1);
DACE_EXPORTED void __dace_runkernel_assign_215_4_map_0_0_14(mttkrp_order_5_mode_2_t *__state, double * __restrict__ gpu_out, int R, int S2);
DACE_EXPORTED void __dace_runkernel_mttkrp_order_5_mode_2_216_0_1_0(mttkrp_order_5_mode_2_t *__state, double * __restrict__ gpu_out, const double * __restrict__ tmp2, const double * __restrict__ tmp3, int R, int S0, int S1, int S2);
void __program_mttkrp_order_5_mode_2_internal(mttkrp_order_5_mode_2_t *__state, double * __restrict__ IM, double * __restrict__ JM, double * __restrict__ LM, double * __restrict__ MM, double * __restrict__ X, double * __restrict__ out, int P0, int P1, int P2, int P3, int P4, int PR, int R, int S0, int S1, int S2, int S3, int S4)
{
    double * tmp;
    cudaMalloc((void**)&tmp, ((R * S3) * S4) * sizeof(double));
    double * tmp2;
    cudaMalloc((void**)&tmp2, (((R * S0) * S1) * S2) * sizeof(double));
    double * tmp3;
    cudaMalloc((void**)&tmp3, ((R * S0) * S1) * sizeof(double));
    double * gpu_MM;
    cudaMalloc((void**)&gpu_MM, (R * S4) * sizeof(double));
    double * gpu_LM;
    cudaMalloc((void**)&gpu_LM, (R * S3) * sizeof(double));
    double * gpu_out;
    cudaMalloc((void**)&gpu_out, (R * S2) * sizeof(double));
    double * gpu_IM;
    cudaMalloc((void**)&gpu_IM, (R * S0) * sizeof(double));
    double * gpu_X;
    cudaMalloc((void**)&gpu_X, ((((S0 * S1) * S2) * S3) * S4) * sizeof(double));
    double * gpu_JM;
    cudaMalloc((void**)&gpu_JM, (R * S1) * sizeof(double));

    {

        cudaMemcpyAsync(gpu_LM, LM, (R * S3) * sizeof(double), cudaMemcpyHostToDevice, __state->gpu_context->streams[0]);
        cudaMemcpyAsync(gpu_MM, MM, (R * S4) * sizeof(double), cudaMemcpyHostToDevice, __state->gpu_context->streams[1]);
        cudaMemcpyAsync(gpu_X, X, ((((S0 * S1) * S2) * S3) * S4) * sizeof(double), cudaMemcpyHostToDevice, __state->gpu_context->streams[2]);
        cudaMemcpyAsync(gpu_IM, IM, (R * S0) * sizeof(double), cudaMemcpyHostToDevice, __state->gpu_context->streams[3]);
        cudaMemcpyAsync(gpu_JM, JM, (R * S1) * sizeof(double), cudaMemcpyHostToDevice, __state->gpu_context->streams[4]);
        cudaMemcpyAsync(gpu_out, out, (R * S2) * sizeof(double), cudaMemcpyHostToDevice, __state->gpu_context->streams[5]);
        cudaStreamSynchronize(__state->gpu_context->streams[0]);
        cudaStreamSynchronize(__state->gpu_context->streams[1]);
        cudaStreamSynchronize(__state->gpu_context->streams[2]);
        cudaStreamSynchronize(__state->gpu_context->streams[3]);
        cudaStreamSynchronize(__state->gpu_context->streams[4]);
        cudaStreamSynchronize(__state->gpu_context->streams[5]);


    }
    {
        int parent_grid;
        int reduce_grid;

        __dace_runkernel_mttkrp_order_5_mode_2_202_0_0_2(__state, gpu_LM, gpu_MM, tmp, R, S3, S4);
        {
            double * _left_tensor = &gpu_X[0];
            double * _right_tensor = &tmp[0];
            double* _out_tensor = tmp2;

            ///////////////////
            int __dace_current_stream_id = 3;
            cudaStream_t __dace_current_stream = __state->gpu_context->streams[__dace_current_stream_id];
            const int __dace_cuda_device = 0;
            cutensorHandle_t &__dace_cutensor_handle = __state->cutensor_handle.Get(__dace_cuda_device);
            // cutensorSetStream(__dace_cutensor_handle, __dace_current_stream);

            double alpha = (double)1.0;
            double beta = (double)0.0;

            std::vector<int> modeA{0,1,2,3,4};
            std::vector<int> modeB{3,4,7};
            std::vector<int> modeC{0,1,2,7};
            std::unordered_map<int, int64_t> extent;
            extent[0] = S0;
            extent[1] = S1;
            extent[2] = S2;
            extent[3] = S3;
            extent[4] = S4;
            extent[3] = S3;
            extent[4] = S4;
            extent[7] = R;

            std::vector<int64_t> extentA;
            for (auto mode : modeA) extentA.push_back(extent[mode]);
            std::vector<int64_t> extentB;
            for (auto mode : modeB) extentB.push_back(extent[mode]);
            std::vector<int64_t> extentC;
            for (auto mode : modeC) extentC.push_back(extent[mode]);

            std::vector<int64_t> stridesA{S1*S2*S3*S4,S2*S3*S4,S3*S4,S4,1};
            std::vector<int64_t> stridesB{R*S4,R,1};
            std::vector<int64_t> stridesC{R*S1*S2,R*S2,R,1};

            cutensorTensorDescriptor_t descA, descB, descC;
            dace::linalg::CheckCuTensorError(cutensorInitTensorDescriptor(
            &__dace_cutensor_handle, &descA, modeA.size(), extentA.data(), stridesA.data(), CUDA_R_64F, CUTENSOR_OP_IDENTITY));
            dace::linalg::CheckCuTensorError(cutensorInitTensorDescriptor(
            &__dace_cutensor_handle, &descB, modeB.size(), extentB.data(), stridesB.data(), CUDA_R_64F, CUTENSOR_OP_IDENTITY));
            dace::linalg::CheckCuTensorError(cutensorInitTensorDescriptor(
            &__dace_cutensor_handle, &descC, modeC.size(), extentC.data(), stridesC.data(), CUDA_R_64F, CUTENSOR_OP_IDENTITY));
            // printf("Tensor descriptors created!\n");

            uint32_t alignmentRequirementA, alignmentRequirementB, alignmentRequirementC;
            dace::linalg::CheckCuTensorError(cutensorGetAlignmentRequirement(&__dace_cutensor_handle, _left_tensor, &descA, &alignmentRequirementA));
            dace::linalg::CheckCuTensorError(cutensorGetAlignmentRequirement(&__dace_cutensor_handle, _right_tensor, &descB, &alignmentRequirementB));
            dace::linalg::CheckCuTensorError(cutensorGetAlignmentRequirement(&__dace_cutensor_handle, _out_tensor, &descC, &alignmentRequirementC));
            cutensorContractionDescriptor_t desc;
            dace::linalg::CheckCuTensorError(cutensorInitContractionDescriptor(
            &__dace_cutensor_handle, &desc,
            &descA, modeA.data(), alignmentRequirementA,
            &descB, modeB.data(), alignmentRequirementB,
            &descC, modeC.data(), alignmentRequirementC,
            &descC, modeC.data(), alignmentRequirementC,
            CUTENSOR_COMPUTE_64F));
            // printf("Memory alignment and coontraction descriptor created!\n");

            cutensorContractionFind_t find;
            dace::linalg::CheckCuTensorError(cutensorInitContractionFind(&__dace_cutensor_handle, &find, CUTENSOR_ALGO_DEFAULT));
            size_t worksize = 0;
            dace::linalg::CheckCuTensorError(cutensorContractionGetWorkspace(
            &__dace_cutensor_handle, &desc, &find, CUTENSOR_WORKSPACE_RECOMMENDED, &worksize));
            void *work = nullptr;
            if (worksize > 0) cudaMalloc(&work, worksize);
            // printf("Workspace created!\n");

            cutensorContractionPlan_t plan;
            dace::linalg::CheckCuTensorError(cutensorInitContractionPlan(&__dace_cutensor_handle, &plan, &desc, &find, worksize));
            cutensorStatus_t err;
            err = cutensorContraction(
            &__dace_cutensor_handle, &plan,
            (void*)&alpha, _left_tensor, _right_tensor, (void*)&beta, _out_tensor, _out_tensor,
            work, worksize, __dace_current_stream);
            cudaStreamSynchronize(__dace_current_stream);
            if(err != CUTENSOR_STATUS_SUCCESS) {
                printf("ERROR: %s\n", cutensorGetErrorString(err));
            }
            if (work) cudaFree(work);
            // printf("Contraction executed!\n");

            ///////////////////

        }
        __dace_runkernel_mttkrp_order_5_mode_2_210_0_0_9(__state, gpu_IM, gpu_JM, tmp3, R, S0, S1);
        __dace_runkernel_assign_215_4_map_0_0_14(__state, gpu_out, R, S2);
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
        cudaStreamSynchronize(__state->gpu_context->streams[0]);
        cudaStreamSynchronize(__state->gpu_context->streams[3]);
        cudaStreamSynchronize(__state->gpu_context->streams[4]);


    }
    {

        __dace_runkernel_mttkrp_order_5_mode_2_216_0_1_0(__state, gpu_out, tmp2, tmp3, R, S0, S1, S2);
        {
            double * _inbuffer = &gpu_out[0];
            double* _outbuffer = gpu_out;

            ///////////////////
            int __dace_current_stream_id = 0;
            cudaStream_t __dace_current_stream = __state->gpu_context->streams[__dace_current_stream_id];
            if (__state->__pgrid_1_size > 1) {
                MPI_Allreduce(MPI_IN_PLACE, _outbuffer, S2*R, MPI_DOUBLE, MPI_SUM, __state->__pgrid_1_comm);
            }
            ///////////////////

        }
        cudaMemcpyAsync(out, gpu_out, (R * S2) * sizeof(double), cudaMemcpyDeviceToHost, __state->gpu_context->streams[0]);
        cudaStreamSynchronize(__state->gpu_context->streams[0]);


    }
    cudaFree(tmp);
    cudaFree(tmp2);
    cudaFree(tmp3);
    cudaFree(gpu_MM);
    cudaFree(gpu_LM);
    cudaFree(gpu_out);
    cudaFree(gpu_IM);
    cudaFree(gpu_X);
    cudaFree(gpu_JM);
}

DACE_EXPORTED void __program_mttkrp_order_5_mode_2(mttkrp_order_5_mode_2_t *__state, double * __restrict__ IM, double * __restrict__ JM, double * __restrict__ LM, double * __restrict__ MM, double * __restrict__ X, double * __restrict__ out, int P0, int P1, int P2, int P3, int P4, int PR, int R, int S0, int S1, int S2, int S3, int S4)
{
    __program_mttkrp_order_5_mode_2_internal(__state, IM, JM, LM, MM, X, out, P0, P1, P2, P3, P4, PR, R, S0, S1, S2, S3, S4);
}
DACE_EXPORTED int __dace_init_cuda(mttkrp_order_5_mode_2_t *__state, int P0, int P1, int P2, int P3, int P4, int PR, int R, int S0, int S1, int S2, int S3, int S4);
DACE_EXPORTED int __dace_exit_cuda(mttkrp_order_5_mode_2_t *__state);

DACE_EXPORTED mttkrp_order_5_mode_2_t *__dace_init_mttkrp_order_5_mode_2(int P0, int P1, int P2, int P3, int P4, int PR, int R, int S0, int S1, int S2, int S3, int S4)
{
    int __result = 0;
    mttkrp_order_5_mode_2_t *__state = new mttkrp_order_5_mode_2_t;


    __result |= __dace_init_cuda(__state, P0, P1, P2, P3, P4, PR, R, S0, S1, S2, S3, S4);
    {  // Environment: MPI
        int t; MPI_Initialized(&t);  if (!t) MPI_Init(NULL, NULL);
    }
    __state->__pgrid_0_dims[0] = P0;
    __state->__pgrid_0_dims[1] = P1;
    __state->__pgrid_0_dims[2] = P2;
    __state->__pgrid_0_dims[3] = P3;
    __state->__pgrid_0_dims[4] = P4;
    __state->__pgrid_0_dims[5] = PR;

    int __pgrid_0_periods[6] = {0};
    MPI_Cart_create(MPI_COMM_WORLD, 6, __state->__pgrid_0_dims, __pgrid_0_periods, 0, &__state->__pgrid_0_comm);
    if (__state->__pgrid_0_comm != MPI_COMM_NULL) {
        MPI_Comm_group(__state->__pgrid_0_comm, &__state->__pgrid_0_group);
        MPI_Comm_rank(__state->__pgrid_0_comm, &__state->__pgrid_0_rank);
        MPI_Comm_size(__state->__pgrid_0_comm, &__state->__pgrid_0_size);
        MPI_Cart_coords(__state->__pgrid_0_comm, __state->__pgrid_0_rank, 6, __state->__pgrid_0_coords);
        __state->__pgrid_0_valid = true;
    } else {
        __state->__pgrid_0_group = MPI_GROUP_NULL;
        __state->__pgrid_0_rank = MPI_PROC_NULL;
        __state->__pgrid_0_size = 0;
        __state->__pgrid_0_valid = false;
    }
    __state->__pgrid_1_dims[0] = P0;
    __state->__pgrid_1_dims[1] = P1;
    __state->__pgrid_1_dims[2] = P3;
    __state->__pgrid_1_dims[3] = P4;

    __state->__pgrid_1_valid = false;
    if (__state->__pgrid_0_valid) {
        int __pgrid_1_remain[6] = {1, 1, 0, 1, 1, 0};
        MPI_Cart_sub(__state->__pgrid_0_comm, __pgrid_1_remain, &__state->__pgrid_1_comm);
        MPI_Comm_group(__state->__pgrid_1_comm, &__state->__pgrid_1_group);
        MPI_Comm_rank(__state->__pgrid_1_comm, &__state->__pgrid_1_rank);
        MPI_Comm_size(__state->__pgrid_1_comm, &__state->__pgrid_1_size);
        MPI_Cart_coords(__state->__pgrid_1_comm, __state->__pgrid_1_rank, 4, __state->__pgrid_1_coords);

        __state->__pgrid_1_valid = true;
    }


    if (__result) {
        delete __state;
        return nullptr;
    }
    return __state;
}

DACE_EXPORTED void __dace_exit_mttkrp_order_5_mode_2(mttkrp_order_5_mode_2_t *__state)
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

