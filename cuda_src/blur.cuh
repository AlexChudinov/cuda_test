#pragma once

void gaussian_blur_wrapper(const uchar4* const d_image,
                           uchar4* const d_blurredImage,
                           const int numRows,
                           const int numCols,
                           const float * const d_filter, 
                           const int filterWidth,
                           dim3 gridSize,
                           dim3 blockSize,
                           size_t filterSize);
