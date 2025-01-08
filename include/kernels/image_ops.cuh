#pragma once
#include "common.cuh"

namespace cudalib {
namespace image {
    
// Basic image processing operations
__global__ void grayscale(const pixel *input, pixel *output, int width, int height);
__global__ void medianFilter(const pixel *input, pixel *output, int width, int height);

// Color space conversions
__global__ void rgb2gray(const pixel *input, pixel *output, int width, int height);

}   // namespace image
}   // namespace cudalib
