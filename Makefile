CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -pthread -fopenmp
CPPFLAGS += -Isrc/third_party/hnswlib -Isrc/third_party/dgm/include
DGM = $(wildcard src/third_party/dgm/src/*.cpp)
CORE = src/starr.cpp
HEADERS = $(shell find src -name '*.h' -o -name '*.hpp' -o -name '*.inc')

.PHONY: all check clean
all: build/starr build/benchmark build/benchmark_matrix build/recall_qps build/ablation build/dgm_sweep build/matrix build/direct_reuse
build:
	mkdir -p build
build/starr: src/main.cpp src/benchmark.cpp $(CORE) $(DGM) $(HEADERS) | build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(CORE) $(DGM) -o $@
build/benchmark: src/benchmark.cpp $(CORE) $(DGM) $(HEADERS) | build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(CORE) $(DGM) -o $@
build/benchmark_matrix: src/benchmark.cpp src/matrix/starr.cpp $(DGM) $(HEADERS) | build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -DMATRIX_OURS $< src/matrix/starr.cpp $(DGM) -o $@
build/recall_qps: src/recall_qps.cpp src/benchmark.cpp $(CORE) $(DGM) $(HEADERS) | build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(CORE) $(DGM) -o $@
build/ablation: src/ablation.cpp src/benchmark.cpp $(CORE) $(DGM) $(HEADERS) | build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(CORE) $(DGM) -o $@
build/dgm_sweep: src/dgm_sweep.cpp src/benchmark.cpp $(CORE) $(DGM) $(HEADERS) | build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(CORE) $(DGM) -o $@
build/matrix: src/matrix/benchmark.cpp src/matrix/evaluation.h src/matrix/starr.cpp $(HEADERS) | build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< src/matrix/starr.cpp -o $@
build/direct_reuse: src/direct_reuse.cpp src/matrix/evaluation.h $(HEADERS) | build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $@
check: all
	python3 experiments/smoke_test.py
	python3 experiments/script_smoke_test.py
clean:
	rm -rf build
