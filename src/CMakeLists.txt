# Add CUDA source files
add_library(cuda_lib STATIC
    kernels/image_ops.cu
    utils/cuda_utils.cu
)

# Set properties
set_target_properties(cuda_lib PROPERTIES
    CUDA_SEPARABLE_COMPLAINTION ON
    POSITION_INDEPENDENT_CODE ON
)

# Include directories
target_include_directories(cuda_lib PUBLIC 
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/kernels
)