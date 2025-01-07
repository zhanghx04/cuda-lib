cmake_minimum_required(VERSION 3.18)
project(cuda_lib CUDA CXX)

# Set CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES 75) # Adjust based on your GPU

# Find CUDA
find_package(CUDA REQUIRED)

# Set compiler flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++17")

# Include directories
include_directories(
	${PROJECT_SOURCE_DIR}/include
	${CUDA_INCLUDE_DIRS}
)

# Add subdirectories
add_subdirectory(src)
if(BUILD_TESTING)
	add_subdirectory(tests)
endif()

if(BUILD_EXAMLPLES)
	add_subdirectory(examples)
endif()
