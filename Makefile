CXX := g++
NVCC ?= nvcc

GPU_ARCH ?=

# CUDA toolkit location. Override on hosts where CUDA is not at /usr/local/cuda,
# e.g. the training box: make gpt_train GPU_ARCH=sm_120 CUDA_HOME=/usr/local/cuda-13.4
CUDA_HOME ?= /usr/local/cuda
CUDA_LIB := $(wildcard $(CUDA_HOME)/lib64)

CXXFLAGS := -O3 -std=c++17 -Wall -Wextra -I.
NVCCFLAGS := -O3 -std=c++17 -I. $(if $(GPU_ARCH),-arch=$(GPU_ARCH),)
LDFLAGS := -lcublas -lcurand
# If the toolkit's lib64 exists, link against it and bake an rpath so the built
# binaries find the right libcublas/libcurand without LD_LIBRARY_PATH set.
ifneq ($(CUDA_LIB),)
LDFLAGS += -L$(CUDA_LIB) -Xlinker -rpath -Xlinker $(CUDA_LIB)
endif

SRC_DIR := src
OBJ_DIR := build

CPP_SRCS := $(shell find $(SRC_DIR) -name "*.cpp")
CU_SRCS := $(shell find $(SRC_DIR) -name "*.cu")

CPP_OBJS := $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(CPP_SRCS))
CU_OBJS := $(patsubst $(SRC_DIR)/%.cu, $(OBJ_DIR)/%.o, $(CU_SRCS))

MAIN_SRC ?= main.cpp
TARGET ?= main

.PHONY: all clean bpe_train gpt_train gpt_train_ir gpt_chat gpt_gen bf16_adam_test lib

all: $(TARGET)

$(TARGET): $(CPP_OBJS) $(CU_OBJS) $(MAIN_SRC)
	$(NVCC) $(NVCCFLAGS) $(MAIN_SRC) $(CPP_OBJS) $(CU_OBJS) $(LDFLAGS) -o $(TARGET)

bpe_train: scripts/bpe_train.cpp $(OBJ_DIR)/core/tokenizer.o
	$(CXX) $(CXXFLAGS) $^ -o $@

gpt_train: $(CPP_OBJS) $(CU_OBJS) scripts/gpt_train.cpp
	$(NVCC) $(NVCCFLAGS) scripts/gpt_train.cpp $(CPP_OBJS) $(CU_OBJS) $(LDFLAGS) -o $@

gpt_train_ir: $(CPP_OBJS) $(CU_OBJS) scripts/gpt_train_ir.cpp
	$(NVCC) $(NVCCFLAGS) scripts/gpt_train_ir.cpp $(CPP_OBJS) $(CU_OBJS) $(LDFLAGS) -o $@

gpt_chat: $(CPP_OBJS) $(CU_OBJS) scripts/gpt_chat.cpp
	$(NVCC) $(NVCCFLAGS) scripts/gpt_chat.cpp $(CPP_OBJS) $(CU_OBJS) $(LDFLAGS) -o $@

gpt_gen: $(CPP_OBJS) $(CU_OBJS) scripts/gpt_gen.cpp
	$(NVCC) $(NVCCFLAGS) scripts/gpt_gen.cpp $(CPP_OBJS) $(CU_OBJS) $(LDFLAGS) -o $@

bf16_adam_test: $(CPP_OBJS) $(CU_OBJS) scripts/bf16_adam_test.cpp
	$(NVCC) $(NVCCFLAGS) scripts/bf16_adam_test.cpp $(CPP_OBJS) $(CU_OBJS) $(LDFLAGS) -o $@

lib: $(CPP_OBJS) $(CU_OBJS)
	ar rcs libscratch.a $(CPP_OBJS) $(CU_OBJS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJ_DIR) $(TARGET) libscratch.a
