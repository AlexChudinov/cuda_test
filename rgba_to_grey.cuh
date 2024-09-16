#ifndef _RGBA_TO_GREY_CUH_
#define _RGBA_TO_GREY_CUH_

#include <cuda_runtime.h>

void rgba_to_grayscale_simple_wrapper(
    const uchar4* const d_imageRGBA,
    unsigned char* const d_imageGray,
    size_t numRows,
    size_t numCols,
    const dim3& gridSize,
    const dim3& blockSize
);

void rgba_to_grayscale_optimized_wrapper(
    const uchar4* const d_imageRGBA,
    unsigned char* const d_imageGray,
    int numRows,
    int numCols,
    int elementsPerThread,
    dim3 gridSize,
    dim3 blockSize
);

#endif
