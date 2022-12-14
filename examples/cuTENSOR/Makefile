CXX_FLAGS=-std=c++11 -O3 -I${CUTENSOR_ROOT}/include -I../../include \
						 -L${CUTENSOR_ROOT}/lib/11 -lcutensor -lcudart -lcublas \
						 --compiler-options "-Wall -Wextra" \
						 -gencode=arch=compute_86,code=sm_86 \
						 -gencode=arch=compute_80,code=sm_80 \
						 -gencode=arch=compute_70,code=sm_70 \
						 -use_fast_math -lineinfo

all: einsum contraction contraction_simple contraction_autotuning elementwise_binary elementwise_permute elementwise_trinary reduction gemm_cublas cublas_benchmark

einsum: einsum.cu
	nvcc einsum.cu -o  einsum ${CXX_FLAGS}

contraction: contraction.cu
	nvcc contraction.cu -o  contraction ${CXX_FLAGS}

cublas_benchmark: cublas_benchmark.cu
	nvcc cublas_benchmark.cu -o  cublas_benchmark ${CXX_FLAGS}

gemm_cublas: gemm_cublas.cu
	nvcc gemm_cublas.cu -o  gemm_cublas ${CXX_FLAGS}

contraction_simple: contraction_simple.cu
	nvcc contraction_simple.cu -o  contraction_simple ${CXX_FLAGS}

contraction_autotuning: contraction_autotuning.cu
	nvcc contraction_autotuning.cu -o  contraction_autotuning ${CXX_FLAGS}

elementwise_binary: elementwise_binary.cu
	nvcc elementwise_binary.cu -o  elementwise_binary ${CXX_FLAGS}

elementwise_permute: elementwise_permute.cu
	nvcc elementwise_permute.cu -o  elementwise_permute ${CXX_FLAGS}

elementwise_trinary: elementwise_trinary.cu
	nvcc elementwise_trinary.cu -o  elementwise_trinary ${CXX_FLAGS}

reduction: reduction.cu
	nvcc reduction.cu -o  reduction ${CXX_FLAGS}

clean:
	rm -f contraction contraction_simple contraction_autotuning elementwise_binary elementwise_permute elementwise_trinary reduction gemm_cublas cublas_benchmark

nsys_profile:
	nsys profile -t cuda,cublas,nvtx --stats=true --cuda-memory-usage=true --gpu-metrics-device=all ./contraction

