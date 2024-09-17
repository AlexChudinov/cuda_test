#pragma once

#include <cuda_runtime.h>
#include <functional>
#include <memory>
#include <sstream>
#include <vector>

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

template <typename T>
void check(T err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        std::stringstream out;
        out << "CUDA error at: " << file << ":" << line << ":" 
        << cudaGetErrorString(err) << " " << func << std::endl;
        throw std::runtime_error(out.str());
    }
}

template <typename T>
struct CudaPtrDeleter {
    void operator()(T** ptr) const {
        cudaFree(*ptr);
        delete ptr;
    }
};

template <typename T>
using cuda_ptr = std::unique_ptr<T*, CudaPtrDeleter<T>>;

template <typename T>
cuda_ptr<T> make_cuda_ptr(size_t num_elements) {
    auto h_ptr = std::make_unique<T*>();
    checkCudaErrors(cudaMalloc(h_ptr.get(), num_elements * sizeof(T)));
    cuda_ptr<T> d_ptr;
    d_ptr.reset(h_ptr.release());
    return d_ptr;
}

template <typename T>
void cuda_fill_ptr(const T* h_ptr, cuda_ptr<T>& d_ptr, size_t num_elements) {
    checkCudaErrors(
        cudaMemcpy(*d_ptr, h_ptr, num_elements * sizeof(T), cudaMemcpyHostToDevice));
}

template <typename T>
void cuda_get_data(const cuda_ptr<T>& d_ptr, T* h_ptr, size_t num_elements) {
    checkCudaErrors(
        cudaMemcpy(h_ptr, *d_ptr, num_elements * sizeof(T), cudaMemcpyDeviceToHost));
}

template <typename T, class Container>
void cuda_fill_ptr(const Container& h_data, cuda_ptr<T>& d_ptr) {
    cuda_fill_ptr(h_data.data(), d_ptr, h_data.size());
}

template <typename T>
void cuda_get_data(const cuda_ptr<T>& d_ptr, std::vector<T>& h_data) {
    cuda_get_data(d_ptr, h_data.data(), h_data.size());
}

template <typename T, class Container>
cuda_ptr<T> make_cuda_ptr(const Container& h_data) {
    cuda_ptr<T> d_ptr = make_cuda_ptr<T>(h_data.size());
    cuda_fill_ptr(h_data, d_ptr);
    return d_ptr;
}

template <typename Tin, typename Tout>
using MatrixMapFun = std::function<void(const Tin*, Tout*, size_t, size_t, const dim3&, const dim3&)>;

using CudaWrapFun = std::function<void(const dim3&, const dim3&)>;

template <typename Tin, typename Tout>
CudaWrapFun cuda_benchmark_matrix_map(const MatrixMapFun<Tin, Tout>& fun,
                                      const std::vector<Tin>& h_input,
                                      std::vector<Tout>& h_output,
                                      size_t numRows,
                                      size_t numCols)
{
     return [&](const dim3& gridSize, const dim3& blockSize) {
        auto d_input = make_cuda_ptr<Tin>(h_input);
        auto d_output = make_cuda_ptr<Tout>(h_output.size());

        cudaEvent_t start;
        cudaEvent_t stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        fun(*d_input, *d_output, numRows, numCols, gridSize, blockSize);
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        std::cout << "CUDA time elapsed (ms): " << milliseconds << std::endl;
        cuda_get_data(d_output, h_output);
    };
}
