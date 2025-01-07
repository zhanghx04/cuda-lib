# Self Use CUDA Library 

## Structure
```
cuda-lib/
├── CMakeLists.txt
├── README.md
├── .gitignore
├── include/
│   └── kernels/
│       ├── common.cuh        # Common utilities and helper functions
│       ├── math_ops.cuh     # Math operation kernels
│       ├── image_ops.cuh    # Image processing kernels
│       └── memory_ops.cuh   # Memory operation kernels
├── src/
│   ├── kernels/
│   │   ├── math_ops.cu
│   │   ├── image_ops.cu
│   │   └── memory_ops.cu
│   └── utils/
│       └── cuda_utils.cu     # CUDA utility functions
├── tests/
│   ├── CMakeLists.txt
│   ├── test_math_ops.cu
│   ├── test_image_ops.cu
│   └── test_memory_ops.cu
├── examples/
│   ├── CMakeLists.txt
│   ├── basic_operations/
│   └── image_processing/
└── docs/
    ├── api/
    ├── examples/
    └── performance/
```