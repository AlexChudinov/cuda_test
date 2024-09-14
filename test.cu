#include "test.cuh"
#include <stdio.h>

__global__ void test_print() {
    printf("Hello CUDA\n");
}

void wrap_test_print() {
    test_print<<<1, 1>>>();
}
