CXX := g++
NVCC := nvcc

CXXFLAGS := -O3 -std=c++17 -Wall -Wextra -I.
NVCCFLAGS := -O3 -std=c++17 -I.
LDFLAGS := -lcublas -lcurand

SRC_DIR := src
OBJ_DIR := build

CPP_SRCS := $(shell find $(SRC_DIR) -name "*.cpp")
CU_SRCS := $(shell find $(SRC_DIR) -name "*.cu")

CPP_OBJS := $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(CPP_SRCS))
CU_OBJS := $(patsubst $(SRC_DIR)/%.cu, $(OBJ_DIR)/%.o, $(CU_SRCS))

MAIN_SRC ?= main.cpp
TARGET ?= main

all: $(TARGET)

$(TARGET): $(CPP_OBJS) $(CU_OBJS) $(MAIN_SRC)
	$(NVCC) $(NVCCFLAGS) $(MAIN_SRC) $(CPP_OBJS) $(CU_OBJS) $(LDFLAGS) -o $(TARGET)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJ_DIR) $(TARGET)
