CXX_FLAGS=-std=c++11 -O3 -I${CUTENSOR_ROOT}/include -I../../include \
						 -L${CUTENSOR_ROOT}/lib/11 -lcutensor -lcudart -lcublas -lcublasLt \
						 --compiler-options "-Wall -Wextra" \
						 -gencode=arch=compute_86,code=sm_86 \
						 -gencode=arch=compute_80,code=sm_80 \
						 -gencode=arch=compute_70,code=sm_70 \
						 -use_fast_math -lineinfo

all: contraction gemm_cublas cublas_single_algo cublas_benchmark lt_benchmark two_index_matmul batched_matmul

contraction: contraction.cu
	nvcc contraction.cu -o  contraction ${CXX_FLAGS}

cublas_single_algo: cublas_single_algo.cu
	nvcc cublas_single_algo.cu -o  cublas_single_algo ${CXX_FLAGS}

cublas_benchmark: cublas_benchmark.cu
	nvcc cublas_benchmark.cu -o  cublas_benchmark ${CXX_FLAGS}

batched_matmul: batched_matmul.cu
	nvcc batched_matmul.cu -o  batched_matmul ${CXX_FLAGS}

lt_benchmark: lt_benchmark.cu
	nvcc lt_benchmark.cu -o  lt_benchmark ${CXX_FLAGS}

gemm_cublas: gemm_cublas.cu
	nvcc gemm_cublas.cu -o  gemm_cublas ${CXX_FLAGS}

two_index_matmul: two_index_matmul.cu
	nvcc two_index_matmul.cu -o  two_index_matmul ${CXX_FLAGS}

clean:
	rm -f contraction gemm_cublas cublas_single_algo cublas_benchmark lt_benchmark two_index_matmul batched_matmul

nsys_profile:
	nsys profile -t cuda,cublas,nvtx --stats=true --cuda-memory-usage=true --gpu-metrics-device=all ./contraction

