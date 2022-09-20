/* DaCe AUTO-GENERATED FILE. DO NOT MODIFY */
#include <dace/dace.h>
#include "../../include/hash.h"
#include "../include/dace_cutensor.h"

struct mttkrp_order_3_mode_1_compute_t {
    dace::cuda::Context *gpu_context;
    dace::linalg::CuTensorHandle cutensor_handle;
};

DACE_EXPORTED void __dace_runkernel_assign_203_4_map_0_1_7(mttkrp_order_3_mode_1_compute_t *__state, double * __restrict__ gpu_out, int R, int S1);
DACE_EXPORTED void __dace_runkernel_mttkrp_order_3_mode_1_compute_204_0_0_0(mttkrp_order_3_mode_1_compute_t *__state, const double * __restrict__ gpu_IM, double * __restrict__ gpu_out, const double * __restrict__ tmp, int R, int S0, int S1);
void __program_mttkrp_order_3_mode_1_compute_internal(mttkrp_order_3_mode_1_compute_t *__state, double * __restrict__ IM, double * __restrict__ KM, double * __restrict__ X, double * __restrict__ out, int R, int S0, int S1, int S2)
{
    double * tmp;
    cudaMalloc((void**)&tmp, ((R * S0) * S1) * sizeof(double));
    double * gpu_IM;
    cudaMalloc((void**)&gpu_IM, (R * S0) * sizeof(double));
    double * gpu_out;
    cudaMalloc((void**)&gpu_out, (R * S1) * sizeof(double));

    {
        double * gpu_KM;
        cudaMalloc((void**)&gpu_KM, (R * S2) * sizeof(double));
        double * gpu_X;
        cudaMalloc((void**)&gpu_X, ((S0 * S1) * S2) * sizeof(double));

        cudaMemcpyAsync(gpu_X, X, ((S0 * S1) * S2) * sizeof(double), cudaMemcpyHostToDevice, __state->gpu_context->streams[1]);
        cudaMemcpyAsync(gpu_KM, KM, (R * S2) * sizeof(double), cudaMemcpyHostToDevice, __state->gpu_context->streams[2]);

        cudaEventRecord(__state->gpu_context->events[0], __state->gpu_context->streams[2]);
        cudaStreamWaitEvent(__state->gpu_context->streams[1], __state->gpu_context->events[0], 0);

        {
            double * _left_tensor = &gpu_X[0];
            double * _right_tensor = &gpu_KM[0];
            double* _out_tensor = tmp;

            ///////////////////
            int __dace_current_stream_id = 1;
            cudaStream_t __dace_current_stream = __state->gpu_context->streams[__dace_current_stream_id];
            const int __dace_cuda_device = 0;
            cutensorHandle_t &__dace_cutensor_handle = __state->cutensor_handle.Get(__dace_cuda_device);
            // cutensorSetStream(__dace_cutensor_handle, __dace_current_stream);

            double alpha = (double)1.0;
            double beta = (double)0.0;

            std::vector<int> modeA{0,1,2};
            std::vector<int> modeB{2,4};
            std::vector<int> modeC{0,1,4};
            std::unordered_map<int, int64_t> extent;
            extent[0] = S0;
            extent[1] = S1;
            extent[2] = S2;
            extent[2] = S2;
            extent[4] = R;

            std::vector<int64_t> extentA;
            for (auto mode : modeA) extentA.push_back(extent[mode]);
            std::vector<int64_t> extentB;
            for (auto mode : modeB) extentB.push_back(extent[mode]);
            std::vector<int64_t> extentC;
            for (auto mode : modeC) extentC.push_back(extent[mode]);

            std::vector<int64_t> stridesA{S1*S2,S2,1};
            std::vector<int64_t> stridesB{R,1};
            std::vector<int64_t> stridesC{R*S1,R,1};

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
        cudaMemcpyAsync(gpu_IM, IM, (R * S0) * sizeof(double), cudaMemcpyHostToDevice, __state->gpu_context->streams[3]);
        __dace_runkernel_assign_203_4_map_0_1_7(__state, gpu_out, R, S1);
        cudaStreamSynchronize(__state->gpu_context->streams[0]);
        cudaStreamSynchronize(__state->gpu_context->streams[1]);
        cudaStreamSynchronize(__state->gpu_context->streams[3]);

        cudaFree(gpu_KM);
        cudaFree(gpu_X);

    }
    {

        __dace_runkernel_mttkrp_order_3_mode_1_compute_204_0_0_0(__state, gpu_IM, gpu_out, tmp, R, S0, S1);
        cudaMemcpyAsync(out, gpu_out, (R * S1) * sizeof(double), cudaMemcpyDeviceToHost, __state->gpu_context->streams[0]);
        cudaStreamSynchronize(__state->gpu_context->streams[0]);


    }
    cudaFree(tmp);
    cudaFree(gpu_IM);
    cudaFree(gpu_out);
}

DACE_EXPORTED void __program_mttkrp_order_3_mode_1_compute(mttkrp_order_3_mode_1_compute_t *__state, double * __restrict__ IM, double * __restrict__ KM, double * __restrict__ X, double * __restrict__ out, int R, int S0, int S1, int S2)
{
    __program_mttkrp_order_3_mode_1_compute_internal(__state, IM, KM, X, out, R, S0, S1, S2);
}
DACE_EXPORTED int __dace_init_cuda(mttkrp_order_3_mode_1_compute_t *__state, int R, int S0, int S1, int S2);
DACE_EXPORTED int __dace_exit_cuda(mttkrp_order_3_mode_1_compute_t *__state);

DACE_EXPORTED mttkrp_order_3_mode_1_compute_t *__dace_init_mttkrp_order_3_mode_1_compute(int R, int S0, int S1, int S2)
{
    int __result = 0;
    mttkrp_order_3_mode_1_compute_t *__state = new mttkrp_order_3_mode_1_compute_t;


    __result |= __dace_init_cuda(__state, R, S0, S1, S2);

    if (__result) {
        delete __state;
        return nullptr;
    }
    return __state;
}

DACE_EXPORTED void __dace_exit_mttkrp_order_3_mode_1_compute(mttkrp_order_3_mode_1_compute_t *__state)
{
    __dace_exit_cuda(__state);
    delete __state;
}

