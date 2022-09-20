# DTYPE_FLAG=TENSOR
CXX_FLAGS=-std=c++11 -O3 -I${CUTENSOR_ROOT}/include -I../../../../include \
						 -L${CUTENSOR_ROOT}/lib/11 -lcutensor -lcudart -lcublas -lcublasLt \
						 --compiler-options "-Wall -Wextra" \
						 -gencode=arch=compute_86,code=sm_86 \
						 -gencode=arch=compute_80,code=sm_80 \
						 -gencode=arch=compute_70,code=sm_70 \
						 -use_fast_math -lineinfo




all: mttkrp

mttkrp: mttkrp.cu
	nvcc mttkrp.cu -o  mttkrp ${CXX_FLAGS}
clean:
	rm -f mttkrp

