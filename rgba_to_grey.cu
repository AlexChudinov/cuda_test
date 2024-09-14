#include "rgba_to_grey.cuh"

__global__
void rgba_to_grayscale_simple(const uchar4* const d_imageRGBA,
                              unsigned char* const d_imageGray,
                              int numRows, int numCols)
{
    int y = blockDim.y*blockIdx.y + threadIdx.y;
    int x = blockDim.x*blockIdx.x + threadIdx.x;
    if (x>=numCols || y>=numRows)
      return;
    const int offset = y*numCols+x;
    const uchar4 pixel = d_imageRGBA[offset];
    d_imageGray[offset] = 0.299f*pixel.x + 0.587f*pixel.y+0.114f*pixel.z;
}

void rgba_to_grayscale_simple_wrapper(
    const uchar4* const d_imageRGBA,
    unsigned char* const d_imageGray,
    int numRows, 
    int numCols,
    dim3 gridSize,
    dim3 blockSize
) {
    rgba_to_grayscale_simple<<<gridSize, blockSize>>>(d_imageRGBA, d_imageGray, numRows, numCols);
}

#define WARP_SIZE 32

__global__
void rgba_to_grayscale_optimized(const uchar4* const d_imageRGBA,
                                 unsigned char* const d_imageGray,
                                 int numRows, int numCols,
                                 int elemsPerThread)
{
    int y = blockDim.y*blockIdx.y + threadIdx.y;
    int x = blockDim.x*blockIdx.x + threadIdx.x;
    const int loop_start =  (x/WARP_SIZE * WARP_SIZE)*(elemsPerThread-1)+x;
    for (int i=loop_start, j=0; j<elemsPerThread && i<numCols; i+=WARP_SIZE, ++j)
    {
      const int offset = y*numCols+i;
      const uchar4 pixel = d_imageRGBA[offset];
      d_imageGray[offset] = 0.299f*pixel.x + 0.587f*pixel.y+0.114f*pixel.z;
    }
}

void rgba_to_grayscale_optimized_wrapper(
    const uchar4* const d_imageRGBA,
    unsigned char* const d_imageGray,
    int numRows, 
    int numCols,
    int elementsPerThread,
    dim3 gridSize,
    dim3 blockSize
) {
    return rgba_to_grayscale_optimized<<<gridSize, blockSize>>>(d_imageRGBA, d_imageGray, numRows, numCols, elementsPerThread);
}
