#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <omp.h>
#include <iostream>
#include <filesystem>
#include <chrono>

#include "test_config.h"
#include "utils/utils.h"
#include "rgba_to_grey.cuh"

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

void RGBtoGrayscaleCUDA(
    const std::vector<uchar4>& h_imageRGBA,
    std::vector<unsigned char>& h_imageGray,
    size_t numRows,
    size_t numCols
)
{
    cudaSetDevice(0);
    checkCudaErrors(cudaGetLastError());

    int threadNum = 1024;
    dim3 blockSize(threadNum, 1, 1);
    dim3 gridSize(numCols/threadNum + 1, numRows, 1);

    auto runner = cuda_benchmark_matrix_map<uchar4, unsigned char>(
        rgba_to_grayscale_simple_wrapper,
        h_imageRGBA,
        h_imageGray,
        numRows,
        numCols
    );

    runner(
        gridSize,
        blockSize
    );
}

void RGBtoGrayscaleCUDAOpt(
    const std::vector<uchar4>& h_imageRGBA,
    std::vector<unsigned char>& h_imageGray,
    size_t numRows,
    size_t numCols
)
{
    const size_t numPixels = numRows * numCols;
    cudaSetDevice(0);
    checkCudaErrors(cudaGetLastError());
    auto d_imageRGBA = make_cuda_ptr<uchar4>(h_imageRGBA);
    auto d_imageGray = make_cuda_ptr<unsigned char>(h_imageRGBA.size());

    int threadNum=128;
    const int elemsPerThread = 16;
    dim3 blockSize(threadNum, 1, 1);
    dim3 gridSize(numCols / (threadNum*elemsPerThread) + 1, numRows, 1);

    auto wrapper = [&](
        const uchar4* d_imageRGBA,
        unsigned char* d_imageGray,
        size_t nRows,
        size_t nCols,
        const dim3& gridSize,
        const dim3& blockSize
    ) {
        rgba_to_grayscale_optimized_wrapper(d_imageRGBA, d_imageGray, nRows, nCols, elemsPerThread, gridSize, blockSize);
    };

    auto runner = cuda_benchmark_matrix_map<uchar4, unsigned char>(
        wrapper,
        h_imageRGBA,
        h_imageGray,
        numRows,
        numCols
    );

    runner(gridSize, blockSize);
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
    cv::Mat image = cv::imread(data_dir / "images/zimorodoc.jpg");
    cv::Mat imageBGRA;
    cv::cvtColor(image, imageBGRA, cv::COLOR_BGR2BGRA);
    std::vector<uchar4> imageArray(imageBGRA.ptr<uchar4>(), imageBGRA.ptr<uchar4>() + imageBGRA.total());
    std::vector<unsigned char> imageGrayArray(imageBGRA.total());

    auto start = ch::system_clock::now();
    RGBtoGrayscaleCUDA(imageArray, imageGrayArray, image.rows, image.cols);
    auto duration = ch::duration_cast<ch::milliseconds>(ch::system_clock::now() - start);
    std::cout << "CUDA: " << duration.count() << " мс" << std::endl;

    cv::Mat imageOut;
    imageOut.create(image.rows, image.cols, CV_8UC1);
    std::copy(imageGrayArray.begin(), imageGrayArray.end(), imageOut.ptr<unsigned char>());
    cv::imwrite(output_dir / "zimorodoc-cuda.jpg", imageOut);
}

TEST(openmp_vs_gpu, rgb_to_grey_gpu_opt) {
    cv::Mat image = cv::imread(data_dir / "images/zimorodoc.jpg");
    cv::Mat imageBGRA;
    cv::cvtColor(image, imageBGRA, cv::COLOR_BGR2BGRA);
    std::vector<uchar4> imageArray(imageBGRA.ptr<uchar4>(), imageBGRA.ptr<uchar4>() + imageBGRA.total());
    std::vector<unsigned char> imageGrayArray(imageBGRA.total());


    auto start = ch::system_clock::now();
    RGBtoGrayscaleCUDAOpt(imageArray, imageGrayArray, image.rows, image.cols);
    auto duration = ch::duration_cast<ch::milliseconds>(ch::system_clock::now() - start);
    std::cout << "CUDA: " << duration.count() << " мс" << std::endl;


    cv::Mat imageOut;
    imageOut.create(image.rows, image.cols, CV_8UC1);
    std::copy(imageGrayArray.begin(), imageGrayArray.end(), imageOut.ptr<unsigned char>());
    cv::imwrite(output_dir / "zimorodoc-cuda-opt.jpg", imageOut);
}
