# --- Compilers ---
# CXX is the standard C++ compiler. NVCC is NVIDIA's CUDA compiler.
CXX = g++
NVCC = nvcc

# --- Compiler Flags ---
# -O3 turns on maximum performance optimization. -Wall shows all warnings.
CXXFLAGS = -O3 -Wall -std=c++14
NVCCFLAGS = -O3 -std=c++14

# --- Directories ---
SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin

# --- Final Executable Name ---
TARGET = $(BIN_DIR)/sw

# --- File Hunting ---
CPP_SRCS = $(wildcard $(SRC_DIR)/*.cpp)
CU_SRCS = $(wildcard $(SRC_DIR)/*.cu)

CPP_OBJS = $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(CPP_SRCS))
CU_OBJS = $(patsubst $(SRC_DIR)/%.cu, $(OBJ_DIR)/%.o, $(CU_SRCS))
OBJS = $(CPP_OBJS) $(CU_OBJS)

# --- Libraries ---
LIBS = -L/usr/local/cuda/lib64 -lcudart

# --- Rules ---

all: directories $(TARGET)

directories:
	@mkdir -p $(OBJ_DIR)
	@mkdir -p $(BIN_DIR)

# Link all .o files together to create the final executable
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

# Compile C++ files into .o files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile CUDA files into .o files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Clean rule to wipe out compiled files (useful for a fresh start)
clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

# Convenience rule to build and run in one command
run: all
	./$(TARGET)