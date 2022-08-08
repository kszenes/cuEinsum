CXX_FLAGS=-std=c++11 -I${CUTENSOR_ROOT}/include -L${CUTENSOR_ROOT}/lib/11 -lcutensor -lcudart

all: einsum contraction contraction_simple contraction_autotuning elementwise_binary elementwise_permute elementwise_trinary reduction

einsum: einsum.cu
	nvcc einsum.cu -o  einsum ${CXX_FLAGS}

contraction: contraction.cu
	nvcc contraction.cu -o  contraction ${CXX_FLAGS}

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
	rm -f contraction contraction_simple contraction_autotuning elementwise_binary elementwise_permute elementwise_trinary reduction

