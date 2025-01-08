cmake_minimum_required(VERSION 3.18)
project(cuda_lib CUDA CXX)

# Build options
option(BUILD_TESTING "Build tests" ON)
option(BUILD_EXAMPLES "Build examples" OFF)

# Set CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES 52) # GTX 980M (Maxwell architecture)

# Set CUDA policy
cmake_policy(SET CMP0146 NEW)

# Set compiler flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++17")

# Include directories
include_directories(
	${PROJECT_SOURCE_DIR}/include
	${GTEST_INCLUDE_DIRS}
)

# Add subdirectories
add_subdirectory(src)

if(BUILD_TESTING)
	# Enable testing right after project()
	include(CTest)
	enable_testing()
	add_subdirectory(tests)
endif()

if(BUILD_EXAMLPLES)
	add_subdirectory(examples)
endif()
