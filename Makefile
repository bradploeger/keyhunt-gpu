# keyhunt-gpu
#
#   make                 build with autodetected GPU architecture
#   make ARCH=sm_86      build for a specific architecture
#   make HSIZE=256       larger batch = fewer inversions, more local memory
#   make test            build and run the host-side math tests
#   make clean

NVCC   ?= nvcc
CXX    ?= g++
ARCH   ?= native
HSIZE  ?= 256
FILTER ?= 19

NVCCFLAGS = -O3 -std=c++14 -arch=$(ARCH) \
            -DHSIZE=$(HSIZE) -DFILTER_LOG2_BITS=$(FILTER) \
            -Xptxas -O3,-v --restrict

BIN = keyhunt-gpu

all: $(BIN)

$(BIN): src/search.cu src/secp256k1.h
	$(NVCC) $(NVCCFLAGS) -o $@ src/search.cu

# Host-only validation of the shared arithmetic header. Runs without a GPU.
test: test/test_math.cpp test/test_kernel_logic.cpp test/test_targets.cpp \
      test/test_progress.cpp test/test_ptx_sequence.cpp \
      src/secp256k1.h src/targets.h src/progress.h
	$(CXX) -O2 -std=c++14 -o /tmp/kh_test_math test/test_math.cpp
	$(CXX) -O2 -std=c++14 -o /tmp/kh_test_logic test/test_kernel_logic.cpp
	$(CXX) -O2 -std=c++14 -DFILTER_LOG2_BITS=$(FILTER) \
	    -o /tmp/kh_test_targets test/test_targets.cpp
	$(CXX) -O2 -std=c++14 -o /tmp/kh_test_progress test/test_progress.cpp
	$(CXX) -O2 -std=c++14 -o /tmp/kh_test_ptx test/test_ptx_sequence.cpp
	/tmp/kh_test_math
	@echo
	/tmp/kh_test_logic
	@echo
	/tmp/kh_test_targets
	@echo
	/tmp/kh_test_progress
	@echo
	/tmp/kh_test_ptx

vectors:
	python3 tools/gen_vectors.py > test/vectors.h

clean:
	rm -f $(BIN) /tmp/kh_test_math /tmp/kh_test_logic /tmp/kh_test_targets /tmp/kh_test_progress /tmp/kh_test_ptx

.PHONY: all test clean vectors
