OPENMP_INPUT = input.cpp
COMPOSITION_FREQ = 50
HOST_BINARY_FLAGS = $(DATA_SIZE) $(KERNEL_SIZE)
OPTIMIZATION_FLAGS= --cache 262144:16:4
include ~/compiler/common.mk
