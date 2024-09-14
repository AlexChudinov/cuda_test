#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <omp.h>
#include <iostream>
#include <filesystem>
#include <chrono>

#include "utils.h"
#include "rgba_to_grey.cuh"

#if !defined(DATA_DIR) && !defined(OUTPUT_DIR)
#error "Data directories were not defined"
#endif

const std::filesystem::path data_dir = DATA_DIR;
const std::filesystem::path output_dir=OUTPUT_DIR;

namespace ch = std::chrono;

template <class T1, class T2>
void prepareImagePointers(const std::string& inputImageFileName,
                          cv::Mat& inputImage, 
                          T1** inputImageArray, 
                          cv::Mat& outputImage,
                          T2** outputImageArray, 
                          const int outputImageType)
{
    inputImage = cv::imread(inputImageFileName, cv::IMREAD_COLOR);

    if (inputImage.empty()) 
    {
        throw std::runtime_error("couldn't open input file");
    }

    outputImage.create(inputImage.rows, inputImage.cols, outputImageType);

    cv::cvtColor(inputImage, inputImage, cv::COLOR_BGR2BGRA);

    *inputImageArray = (T1*)inputImage.ptr<char>(0);
    *outputImageArray  = (T2*)outputImage.ptr<char>(0);
}

void RGBtoGrayscaleOpenMP(uchar4 *imageArray, unsigned char *imageGrayArray, int numRows, int numCols)
{
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < numRows; ++i)
    {
        for (int j = 0; j < numCols; ++j)
        {
            const uchar4& pixel = imageArray[i*numCols+j];
            imageGrayArray[i*numCols+j] = 0.299f*pixel.x+ 0.587f*pixel.y+0.114f*pixel.z;
        }
    }
}

void RGBtoGrayscaleCUDA(const uchar4 * const h_imageRGBA, unsigned char* const h_imageGray, size_t numRows, size_t numCols)
{
    uchar4 *d_imageRGBA;
    unsigned char *d_imageGray;
    const size_t numPixels = numRows * numCols;
    cudaSetDevice(0);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaMalloc(&d_imageRGBA, sizeof(uchar4) * numPixels));
    checkCudaErrors(cudaMalloc(&d_imageGray, sizeof(unsigned char) * numPixels));
    checkCudaErrors(cudaMemcpy(d_imageRGBA, h_imageRGBA, sizeof(uchar4) * numPixels, cudaMemcpyHostToDevice));

    dim3 blockSize;
    dim3 gridSize;
    int threadNum;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    threadNum = 1024;
    blockSize = dim3(threadNum, 1, 1);
    gridSize = dim3(numCols/threadNum+1, numRows, 1);
    cudaEventRecord(start);
    rgba_to_grayscale_simple_wrapper(d_imageRGBA, d_imageGray, numRows, numCols, gridSize, blockSize);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "CUDA time simple (ms): " << milliseconds << std::endl;
    checkCudaErrors(cudaMemcpy(h_imageGray, d_imageGray, sizeof(unsigned char) * numPixels, cudaMemcpyDeviceToHost));
    cudaFree(d_imageGray);
    cudaFree(d_imageRGBA);
}

void RGBtoGrayscaleCUDAOpt(const uchar4 * const h_imageRGBA, unsigned char* const h_imageGray, size_t numRows, size_t numCols) {
    uchar4 *d_imageRGBA;
    unsigned char *d_imageGray;
    const size_t numPixels = numRows * numCols;
    cudaSetDevice(0);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaMalloc(&d_imageRGBA, sizeof(uchar4) * numPixels));
    checkCudaErrors(cudaMalloc(&d_imageGray, sizeof(unsigned char) * numPixels));
    checkCudaErrors(cudaMemcpy(d_imageRGBA, h_imageRGBA, sizeof(uchar4) * numPixels, cudaMemcpyHostToDevice));

    int threadNum=128;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    const int elemsPerThread = 16;
    dim3 blockSize(threadNum, 1, 1);
    dim3 gridSize(numCols / (threadNum*elemsPerThread) + 1, numRows, 1);
    cudaEventRecord(start);
    rgba_to_grayscale_optimized_wrapper(d_imageRGBA, d_imageGray, numRows, numCols, elemsPerThread, gridSize, blockSize);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    float milliseconds = 0.;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "CUDA time optimized (ms): " << milliseconds << std::endl;
    checkCudaErrors(cudaMemcpy(h_imageGray, d_imageGray, sizeof(unsigned char) * numPixels, cudaMemcpyDeviceToHost));
    cudaFree(d_imageGray);
    cudaFree(d_imageRGBA);
}

TEST(openmp_vs_gpu, rgb_to_grey_cpu) {
    cv::Mat image;
    cv::Mat imageGray;
    uchar4 *imageArray;
    unsigned char *imageGrayArray;
    prepareImagePointers(data_dir / "images/zimorodoc.jpg", image, &imageArray, imageGray, &imageGrayArray, CV_8UC1);

    auto start = ch::system_clock::now();
    RGBtoGrayscaleOpenMP(imageArray, imageGrayArray, image.rows, image.cols);
    auto duration = ch::duration_cast<ch::milliseconds>(ch::system_clock::now() - start);
    std::cout << "OpenMP: " << duration.count() << " мс" << std::endl;

    cv::imwrite(output_dir / "zimorodoc.jpg", imageGray);
}

TEST(openmp_vs_gpu, rgb_to_grey_gpu) {
    cv::Mat image;
    cv::Mat imageGray;
    uchar4 *imageArray;
    unsigned char *imageGrayArray;
    prepareImagePointers(data_dir / "images/zimorodoc.jpg", image, &imageArray, imageGray, &imageGrayArray, CV_8UC1);

    auto start = ch::system_clock::now();
    RGBtoGrayscaleCUDA(imageArray, imageGrayArray, image.rows, image.cols);
    auto duration = ch::duration_cast<ch::milliseconds>(ch::system_clock::now() - start);
    std::cout << "CUDA: " << duration.count() << " мс" << std::endl;

    cv::imwrite(output_dir / "zimorodoc-cuda.jpg", imageGray);
}

TEST(openmp_vs_gpu, rgb_to_grey_gpu_opt) {
    cv::Mat image;
    cv::Mat imageGray;
    uchar4 *imageArray;
    unsigned char *imageGrayArray;
    prepareImagePointers(data_dir / "images/zimorodoc.jpg", image, &imageArray, imageGray, &imageGrayArray, CV_8UC1);

    auto start = ch::system_clock::now();
    RGBtoGrayscaleCUDAOpt(imageArray, imageGrayArray, image.rows, image.cols);
    auto duration = ch::duration_cast<ch::milliseconds>(ch::system_clock::now() - start);
    std::cout << "CUDA: " << duration.count() << " мс" << std::endl;

    cv::imwrite(output_dir / "zimorodoc-cuda-opt.jpg", imageGray);
}
