#include <gtest/gtest.h>
#include "image_ops.cuh"
#include "common.cuh"

class ImageOpsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Test image dimensions
        width = 32;
        height = 32;
        size = width * height;

        // Allocate host memory
        h_input = new pixel[size];
        h_output = new pixel[size];
        h_expected = new pixel[size];

        // Allocate device memory
        CUDA_CHECK(cudaMalloc(&d_input, size * sizeof(pixel)));
        CUDA_CHECK(cudaMalloc(&d_output, size * sizeof(pixel)));
    }

    void TearDown() override {
        // Free host memeory
        delete[] h_input;
        delete[] h_output;
        delete[] h_expected;

        // Free device memory
        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_output));
    }

    int width, height, size;
    pixel *h_input, *h_output, *h_expected;
    pixel *d_input, *d_output;
};

TEST_F(ImageOpsTest, GrayscaleConversion) {
    // Initialize input image
    for (int i = 0; i < size; ++i) {
        h_input[i].r = 100;
        h_input[i].g = 150;
        h_input[i].b = 200;

        // Calculate expected output
        unsigned char expected = static_cast<unsigned char>(
            0.299f * h_input[i].r + 
            0.587f * h_input[i].g + 
            0.114f * h_input[i].b
        );

        h_expected[i].r = expected;
        h_expected[i].g = expected;
        h_expected[i].b = expected;
    }

    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, size * sizeof(pixel), cudaMemcpyHostToDevice));

    // Launch kernel
    dim3 block(16, 16);
    dim3 grid(
        (width + block.x - 1) / block.x,
        (height + block.y - 1) / block.y
    );
    cudalib::image::grayscale<<<grid, block>>>(d_input, d_output, width, height);

    // Check for errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy output to host
    CUDA_CHECK(cudaMemcpy(h_output, d_output, size * sizeof(pixel), cudaMemcpyDeviceToHost));

    // Verify results
    for (int i = 0; i < size; ++i) {
        EXPECT_EQ(h_output[i].r, h_expected[i].r);
        EXPECT_EQ(h_output[i].g, h_expected[i].g);
        EXPECT_EQ(h_output[i].b, h_expected[i].b);
    }
}

TEST_F(ImageOpsTest, GrayscaleEdgeCases) {
    // All pixels are black
    for (int i = 0; i < size; ++i) {
        h_input[i].r = 0;
        h_input[i].g = 0;
        h_input[i].b = 0;
    }

    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, size * sizeof(pixel), cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid(
        (width + block.x - 1) / block.x,
        (height + block.y - 1) / block.y
    );

    cudalib::image::grayscale<<<grid, block>>>(d_input, d_output, width, height);

    // Check for errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy output to host
    CUDA_CHECK(cudaMemcpy(h_output, d_output, size * sizeof(pixel), cudaMemcpyDeviceToHost));

    // Verify results
    for (int i = 0; i < size; ++i) {
        EXPECT_EQ(h_output[i].r, 0);
        EXPECT_EQ(h_output[i].g, 0);
        EXPECT_EQ(h_output[i].b, 0);
    }
}

TEST_F(ImageOpsTest, GrayscaleRandomValueCases) {
    // Init with random values
    for (int i = 0; i < size; ++i) {
        h_input[i].r = rand() % 256;
        h_input[i].g = rand() % 256;
        h_input[i].b = rand() % 256;

        unsigned char expected = static_cast<unsigned char>(
            0.299f * h_input[i].r + 
            0.587f * h_input[i].g + 
            0.114f * h_input[i].b
        );

        h_expected[i].r = expected;
        h_expected[i].g = expected;
        h_expected[i].b = expected;
    }

    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, size * sizeof(pixel), cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid(
        (width + block.x - 1) / block.x,
        (height + block.y - 1) / block.y
    );

    cudalib::image::grayscale<<<grid, block>>>(d_input, d_output, width, height);

    // Check for errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());    

    // Copy output to host
    CUDA_CHECK(cudaMemcpy(h_output, d_output, size * sizeof(pixel), cudaMemcpyDeviceToHost));

    // Verify results
    for (int i = 0; i < size; ++i) {
        EXPECT_EQ(h_output[i].r, h_expected[i].r);
        EXPECT_EQ(h_output[i].g, h_expected[i].g);
        EXPECT_EQ(h_output[i].b, h_expected[i].b);
    }
}

TEST_F(ImageOpsTest, GrayscaleLargeImage) {
    int large_width = 1980;
    int large_height = 1080;
    int large_size = large_width * large_height;

    // Init input image
    pixel *h_large_input = new pixel[large_size];
    pixel *h_large_output = new pixel[large_size];
    pixel *h_large_expected = new pixel[large_size];

    // Init device memory
    pixel *d_large_input;
    pixel *d_large_output;
    CUDA_CHECK(cudaMalloc(&d_large_input, large_size * sizeof(pixel)));
    CUDA_CHECK(cudaMalloc(&d_large_output, large_size * sizeof(pixel)));

    // Init input image
    for (int i = 0; i < large_size; ++i) {
        h_large_input[i].r = rand() % 256;
        h_large_input[i].g = rand() % 256;
        h_large_input[i].b = rand() % 256;

        unsigned char expected = static_cast<unsigned char>(
            0.299f * h_large_input[i].r + 
            0.587f * h_large_input[i].g + 
            0.114f * h_large_input[i].b
        );

        h_large_expected[i].r = expected;
        h_large_expected[i].g = expected;
        h_large_expected[i].b = expected;
    }

    CUDA_CHECK(cudaMemcpy(d_large_input, h_large_input, large_size * sizeof(pixel), cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid(
        (large_width + block.x - 1) / block.x,
        (large_height + block.y - 1) / block.y
    );

    cudalib::image::grayscale<<<grid, block>>>(d_large_input, d_large_output, large_width, large_height);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_large_output, d_large_output, large_size * sizeof(pixel), cudaMemcpyDeviceToHost));

    for (int i = 0; i < large_size; ++i) {
        EXPECT_NEAR(h_large_output[i].r, h_large_expected[i].r, 1);
        EXPECT_NEAR(h_large_output[i].g, h_large_expected[i].g, 1);
        EXPECT_NEAR(h_large_output[i].b, h_large_expected[i].b, 1);
    }
}