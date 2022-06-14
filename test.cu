#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>

// recursive function to calculate the factorial of a number on device
__device__ int factorial(int n) {
    if (n == 0) {
        return 1;
    } else {
        return n * factorial(n - 1);
    }
}

__global__ void kernel(int n, int *result) {
    *result = factorial(n);
}

int main()
{
    // create int variable on device
    int n = 15;
    int *result;
    cudaMalloc(&result, sizeof(int));
    kernel <<<1, 1>>> (n, result);

    int *result_host;
    cudaMemcpy(&result_host, result, sizeof(int), cudaMemcpyDeviceToHost);
    printf("%d\n", result_host);
}