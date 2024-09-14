#pragma once

#include <cuda_runtime.h>
#include <sstream>

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        std::stringstream out;
        out << "CUDA error at: " << file << ":" << line << ":" 
        << cudaGetErrorString(err) << " " << func << std::endl;
        throw std::runtime_error(out.str());
    }
}