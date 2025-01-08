find_package(GTest REQUIRED)
include(GoogleTest)

# Add test executable
add_executable(test_image_ops
    test_image_ops.cu
)

# Set CUDA as the language for the source file
set_source_files_properties(test_image_ops.cu PROPERTIES LANGUAGE CUDA)

# Link libraries
target_link_libraries(test_image_ops
    PRIVATE
    GTest::GTest
    GTest::Main
    cuda_lib
)

# Enable testing
enable_testing()
gtest_discover_tests(test_image_ops) 