#include "blur.cuh"


__global__
void gaussian_blur(const uchar4* const d_image,
                   uchar4* const d_blurredImage,
                   const int nRows,
                   const int nCols,
                   const float * const d_filter,
                   const int filterWidth)
{
    const int filterSize = filterWidth * filterWidth;
    extern __shared__ float shared_filter[];
    if (threadIdx.y == 0 && threadIdx.x == 0)
    {
        for (int i = 0; i < filterSize; ++i) {
            shared_filter[i] = d_filter[i];
        }
    }
    __syncthreads();

    const int row = blockIdx.y*blockDim.y+threadIdx.y;
    const int col = blockIdx.x*blockDim.x+threadIdx.x;

    if (col >= nCols || row >= nRows)
        return;

    const int halfWidth = filterWidth / 2;

    float R = .0f;
    float G = .0f;
    float B = .0f;
    for (int j = 0; j < filterSize; ++j) {
        int x = j / filterWidth;
        int y = j % filterWidth;
        size_t nColk = max(min(col + x - halfWidth, static_cast<int>(nCols - 1)), 0);
        size_t nRowk = max(min(row + y - halfWidth, static_cast<int>(nRows - 1)), 0);
        size_t k = nRowk * nCols + nColk;
        R += shared_filter[j] * float(d_image[k].x);
        G += shared_filter[j] * float(d_image[k].y);
        B += shared_filter[j] * float(d_image[k].z);
    }

    d_blurredImage[row * nCols + col] = make_uchar4(B, G, R, 255);
}

void gaussian_blur_wrapper(const uchar4* const d_image,
                           uchar4* const d_blurredImage,
                           const int numRows,
                           const int numCols,
                           const float * const d_filter,
                           const int filterWidth,
                           dim3 gridSize,
                           dim3 blockSize,
                           size_t filterSize)
{
    gaussian_blur<<<gridSize, blockSize, filterSize>>>(d_image, d_blurredImage, numRows, numCols, d_filter, filterWidth);
}
