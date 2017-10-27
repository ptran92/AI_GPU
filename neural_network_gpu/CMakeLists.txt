cmake_minimum_required(VERSION 2.8.0)
project(gpu_mnist)
find_package(CUDA REQUIRED)

# For cuda build flags
set(
CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++11 -arch=compute_60 -code=compute_60,sm_62
)

# For executable folder
set(
CMAKE_RUNTIME_OUTPUT_DIRECTORY
${gpu_mnist_SOURCE_DIR}/bin
)

# For include directories
include_directories(
src/components
src/layers
)

# For library path
link_directories(
/usr/local/cuda/lib64
)

# For compilation
file( GLOB components 
src/components/*.cu
)
file( GLOB layers
src/layers/*.cu
)
cuda_add_executable(
build main.cu ${components} ${layers}
)

# For linking
target_link_libraries(
build
-lcublas
)

