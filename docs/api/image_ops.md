# Image Processing Operations

This documentation covers the image processing operations available in the CUDA library.

## Table of Contents
- [Basic Operations](#basic-operations)
  - [Grayscale Conversion](#grayscale-conversion)

## Basic Operations

### Grayscale Conversion

Converts an RGB image to grayscale using the BT.601 standard weights.

#### Function Signature
```CPP
__global__ void grayscale(const pixel *input, pixel *output, int width, int height)
```

#### Description
Converts each RGB pixel to grayscale using the following weights:
- Red: 0.299
- Green: 0.587
- Blue: 0.114

#### Parameters
| Parameter | Type | Description |
|-----------|------|-------------|
| `input` | `const pixel*` | Input image buffer containing RGB pixels |
| `output` | `pixel*` | Output image buffer for grayscale result |
| `width` | `int` | Width of the image in pixels |
| `height` | `int` | Height of the image in pixels |

#### Example Usage
```CPP
// Define grid and block dimensions
dim3 block(16, 16);
dim3 grid(
    (width + block.x - 1) / block.x,
    (height + block.y - 1) / block.y
);
// Launch kernel
cudalib::image::grayscale<<<grid, block>>>(d_input, d_output, width, height);
```

#### Performance Guidelines
- **Block Size**: Recommended 16x16 threads per block
- **Memory Access**: Uses coalesced memory pattern
- **Thread Divergence**: Minimal (only at image boundaries)
- **Memory Requirements**:
  - Input buffer: `width * height * sizeof(pixel)` bytes
  - Output buffer: `width * height * sizeof(pixel)` bytes
  - Constant memory: 12 bytes (grayscale weights)

#### Implementation Details
The kernel:
1. Maps each thread to a pixel position using block and thread indices
2. Performs bounds checking to ensure valid pixel access
3. Computes grayscale value using BT.601 standard weights
4. Writes the same grayscale value to all RGB channels of the output pixel

#### Error Handling
- Bounds checking prevents out-of-bounds memory access
- No dynamic memory allocation within the kernel
- Returns early if thread coordinates exceed image dimensions

#### Tested Scenarios
- ✅ Basic grayscale conversion
- ✅ Black images (all pixels 0)
- ✅ Random pixel values
- ✅ Large image dimensions (1980x1080)
- ✅ Various image sizes

#### Dependencies
- Requires `common.cuh` for:
  - `pixel` structure definition
  - `grayscale_weights` constant array
- CUDA Runtime API

#### Notes
- Input and output buffers must be pre-allocated on the device
- The output image will have identical values across all RGB channels
- Thread block size can be adjusted based on the target GPU architecture