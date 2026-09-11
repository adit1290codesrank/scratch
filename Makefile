CXX := g++
NVCC := nvcc

GPU_ARCH ?=

CXXFLAGS := -O3 -std=c++17 -Wall -Wextra -I.
NVCCFLAGS := -O3 -std=c++17 -I. $(if $(GPU_ARCH),-arch=$(GPU_ARCH),)
LDFLAGS := -lcublas -lcurand

SRC_DIR := src
OBJ_DIR := build

CPP_SRCS := $(shell find $(SRC_DIR) -name "*.cpp")
CU_SRCS := $(shell find $(SRC_DIR) -name "*.cu")

CPP_OBJS := $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(CPP_SRCS))
CU_OBJS := $(patsubst $(SRC_DIR)/%.cu, $(OBJ_DIR)/%.o, $(CU_SRCS))

MAIN_SRC ?= main.cpp
TARGET ?= main

.PHONY: all clean bpe_train gpt_train gpt_chat

all: $(TARGET)

$(TARGET): $(CPP_OBJS) $(CU_OBJS) $(MAIN_SRC)
	$(NVCC) $(NVCCFLAGS) $(MAIN_SRC) $(CPP_OBJS) $(CU_OBJS) $(LDFLAGS) -o $(TARGET)

bpe_train: scripts/bpe_train.cpp $(OBJ_DIR)/core/tokenizer.o
	$(CXX) $(CXXFLAGS) $^ -o $@

gpt_train: $(CPP_OBJS) $(CU_OBJS) scripts/gpt_train.cpp
	$(NVCC) $(NVCCFLAGS) scripts/gpt_train.cpp $(CPP_OBJS) $(CU_OBJS) $(LDFLAGS) -o $@

gpt_chat: $(CPP_OBJS) $(CU_OBJS) scripts/gpt_chat.cpp
	$(NVCC) $(NVCCFLAGS) scripts/gpt_chat.cpp $(CPP_OBJS) $(CU_OBJS) $(LDFLAGS) -o $@

gpt_gen: $(CPP_OBJS) $(CU_OBJS) scripts/gpt_gen.cpp
      $(NVCC) $(NVCCFLAGS) scripts/gpt_gen.cpp $(CPP_OBJS) $(CU_OBJS) $(LDFLAGS) -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJ_DIR) $(TARGET)
