OPENMP_INPUT = input.cpp
HOST_BINARY_FLAGS = $(NUM_ITEMS) $(SPACE) 
COMPOSITION_FREQ = 100
OPTIMIZATION_FLAGS= --cache 262144:16:4
include ~/compiler/common.mk