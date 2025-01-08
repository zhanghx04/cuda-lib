#include "image_ops.cuh"

namespace cudalib {
namespace image {

// Implementations of each kernel in separate sections
// ------------------------------------------
// Basic image processing operations
// ------------------------------------------
__global__ void grayscale(const pixel *input, pixel *output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return ;
    }

    // Calculate the grayscale value
    int index = y * width + x;
    unsigned char gray = static_cast<unsigned char>(
        /* grayscale: BT.601 standard */
        grayscale_weights[0] * input[index].r + 
        grayscale_weights[1] * input[index].g + 
        grayscale_weights[2] * input[index].b
    );

    // Assign the grayscale value to all channels
    output[index].r = gray;
    output[index].g = gray;
    output[index].b = gray;
}

// ------------------------------------------
// Color space conversions
// ------------------------------------------


}   // namespace image
}   // namespace cudalib