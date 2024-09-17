#include <chrono>
#include <filesystem>

#include <gtest/gtest.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "utils/utils.h"
#include "cuda_src/blur.cuh"
#include "test_config.h"

using namespace std::chrono;

const std::filesystem::path test_data_path = DATA_DIR;
const std::filesystem::path output_path=OUTPUT_DIR;
const std::filesystem::path image_path = test_data_path / "images/zimorodoc.jpg";


constexpr double FILTER_SIGMA = 2.;
constexpr size_t FILTER_WIDTH = 9.;


template <size_t FilterWidth>
class Filter {
public:
    Filter(double sigma) {
        constexpr int halfWidth = (FilterWidth >> 1) + FilterWidth % 2;
        float sum = .0f;
        for (size_t i = 0; i < weights_.size(); ++i) {
            int dx = static_cast<int>(i / FilterWidth) - halfWidth;
            int dy = static_cast<int>(i % FilterWidth) - halfWidth;
            int r_2 = dx * dx + dy * dy;
            weights_[i] = expf( -r_2 / 2. / sigma / sigma );
            sum += weights_[i];
        }

        for (auto& w : weights_) {
            w /= sum;
        }
    }

    const std::array<float, FilterWidth * FilterWidth>& weights() const {
        return weights_;
    }

    void Blur(const std::vector<uchar4>& image, std::vector<uchar4>& blureImage, size_t nRows, size_t nCols) const {
        constexpr int halfWidth = (FilterWidth >> 1) + FilterWidth % 2;

        #pragma omp parallel for
        for (size_t i = 0; i < image.size(); ++i) {
            int nCol = i / nRows;
            int nRow = i % nRows;

            float B = 0.;
            float G = 0.;
            float R = 0.;
            for (size_t j = 0; j < weights_.size(); ++j) {
                int x = j / FilterWidth;
                int y = j % FilterWidth;
                size_t nColk = std::max(std::min(nCol + x - halfWidth, static_cast<int>(nCols - 1)), 0);
                size_t nRowk = std::max(std::min(nRow + y - halfWidth, static_cast<int>(nRows - 1)), 0);
                size_t k = nRows * nColk + nRowk;
                R += weights_[j] * float(image[k].x);
                G += weights_[j] * float(image[k].y);
                B += weights_[j] * float(image[k].z);
            }

            blureImage[i] = uchar4{
                .x = static_cast<uchar>(B),
                .y = static_cast<uchar>(G),
                .z = static_cast<uchar>(R),
                .w = image[i].w,
            };
        }
    }

private:
    std::array<float, FilterWidth * FilterWidth> weights_;
};

TEST(gauss_filter, openmp) {
    ASSERT_TRUE(std::filesystem::exists(image_path));
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    cv::Mat imageBGRA;
    cv::cvtColor(image, imageBGRA, cv::COLOR_BGR2RGBA);

    cv::Mat reference;
    auto start = system_clock::now();
    cv::GaussianBlur(image,
                     reference,
                     cv::Size{
                        FILTER_WIDTH,
                        FILTER_WIDTH,
                    },
                     FILTER_SIGMA,
                     FILTER_SIGMA,
                     cv::BORDER_REPLICATE);
    auto stop = system_clock::now();
    std::cout<<"OpenCV time (us): " << duration_cast<microseconds>(stop - start).count() << std::endl;
    cv::imwrite(output_path / "zimorodoc_blur_ref.jpg", reference);

    Filter<FILTER_WIDTH> filter(FILTER_SIGMA);
    std::vector<uchar4> imageArrayBGRA(imageBGRA.ptr<uchar4>(),
                                       imageBGRA.ptr<uchar4>() + imageBGRA.total());
    std::vector<uchar4> imageArrayBlured(imageArrayBGRA.size());
    start = system_clock::now();
    filter.Blur(imageArrayBGRA, imageArrayBlured, image.rows, image.cols);
    stop = system_clock::now();
    std::cout<<"OpenMP time (us): " << duration_cast<microseconds>(stop - start).count() << std::endl;
    cv::imwrite(output_path / "zimorodoc_blur.jpg", cv::Mat(image.rows, image.cols, CV_8UC4, imageArrayBlured.data()));
    for (size_t i = 0; i < imageArrayBlured.size(); ++i) {
        ASSERT_NEAR(imageArrayBlured[i].x, reference.at<uchar3>(i).x, 10);
    }
}

TEST(gauss_filter, cuda) {
    ASSERT_TRUE(std::filesystem::exists(image_path));
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    cv::Mat imageBGRA;
    cv::cvtColor(image, imageBGRA, cv::COLOR_BGR2RGBA);

    cv::Mat reference;
    auto start = system_clock::now();
    cv::GaussianBlur(image,
                    reference,
                    cv::Size{
                        FILTER_WIDTH,
                        FILTER_WIDTH,
                    },
                    FILTER_SIGMA,
                    FILTER_SIGMA,
                    cv::BORDER_REPLICATE);
    std::vector<uchar4> imageArrayBGRA(imageBGRA.ptr<uchar4>(), imageBGRA.ptr<uchar4>() + imageBGRA.total());
    std::vector<uchar4> blurredArray(imageArrayBGRA.size());

    Filter<FILTER_WIDTH> filter(FILTER_SIGMA);

    auto wrapper = [&](const uchar4* d_image,
                       uchar4* d_blurredImage,
                       size_t nRows,
                       size_t nCols,
                       const dim3& gridSize,
                       const dim3& blockSize)
    {
        auto d_filterPtr = make_cuda_ptr<float, decltype(filter.weights())>(filter.weights());
        gaussian_blur_wrapper(
            d_image,
            d_blurredImage,
            nRows,
            nCols,
            *d_filterPtr,
            FILTER_WIDTH,
            gridSize,
            blockSize,
            filter.weights().size() * sizeof(float)
        );
    };

    auto runner = cuda_benchmark_matrix_map<uchar4, uchar4>(wrapper,
                                                            imageArrayBGRA,
                                                            blurredArray,
                                                            image.rows,
                                                            image.cols);
    int threadNum = 32;
    dim3 gridSize(image.rows / threadNum + 1, image.cols / threadNum + 1, 1);
    dim3 blockSize(threadNum, threadNum, 1);
    runner(
        gridSize,
        blockSize
    );

    cv::imwrite(output_path / "zimorodoc_blur_cuda.jpg", cv::Mat(image.rows, image.cols, CV_8UC4, blurredArray.data()));
    for (size_t i = 0; i < blurredArray.size(); ++i) {
        ASSERT_NEAR(blurredArray[i].x, reference.at<uchar3>(i).x, 30);
    }
}
