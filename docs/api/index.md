# CUDA Library API Documentation

Welcome to the Self Use CUDA Library API documentation. This library provides high-performance CUDA implementations for various operations including image processing, mathematical computations, and memory operations.

## Overview

The library is organized into the following main components:

- **Image Operations** - Image processing kernels and utilities
- **Math Operations** - Mathematical computation kernels
- **Memory Operations** - Memory management and manipulation utilities

## Modules

### [Image Operations](image_ops.md)
- Image processing kernels
- Color space conversions
- Filters and transformations

### [Math Operations](math_ops.md)
- Basic arithmetic operations
- Matrix operations
- Vector computations

### [Memory Operations](memory_ops.md)
- Memory allocation utilities
- Data transfer operations
- Memory pattern operations

## Getting Started

To use this library in your CUDA project:

1. Include the relevant headers:

2. Link against the library in your CMake project:
```cmake
target_link_libraries(your_project PRIVATE cuda_lib)
```
## Common Utilities

All modules use common utilities defined in `common.cuh`, including:
- Error checking macros
- Common data structures
- Shared constants

## Performance Considerations

- All kernels are optimized for the Maxwell architecture (Compute Capability 5.2)
- Default block sizes are optimized for common use cases
- See individual module documentation for specific performance guidelines

## Thread Safety

- All CUDA kernels are thread-safe by design
- Host-side functions are documented individually for thread safety guarantees

## Error Handling

The library uses the `CUDA_CHECK` macro for error handling. All CUDA operations are checked for errors automatically when this macro is used.

## Further Reading

- [Examples](../examples/index.md)
- [Performance Benchmarks](../performance/index.md)
- [API Reference](./index.md)