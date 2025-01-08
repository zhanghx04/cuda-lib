/*
    This header file contains common utilities, macros, and type definitions for CUDA kernels.
*/
#pragma once
#include <cuda_runtime.h>

#include <stdio.h>      // for fprintf

// Define grayscale weights
__constant__ float grayscale_weights[] = {0.299f, 0.587f, 0.114f};

// Define pixel structure
struct pixel {
    unsigned char r, g, b;
};


// ERROR CHECKING MACRO
#define CUDA_CHECK(val) check((val), #val, __FILE__, __LINE__)
inline void check(cudaError_t err, const char * const func, const char * const file, const int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error at %s : %d\n", file, line);
        fprintf(stderr, "%s %s\n", cudaGetErrorString(err), func);
        exit(EXIT_FAILURE);
    }
}
    