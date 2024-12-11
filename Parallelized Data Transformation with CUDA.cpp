#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

// CUDA kernel for squaring elements
__global__ void squareElements(int* d_data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_data[idx] *= d_data[idx];
    }
}

void parallelTransformCUDA(std::vector<int>& data) {
    int n = data.size();
    int size = n * sizeof(int);

    // Allocate memory on the device
    int* d_data;
    cudaMalloc(&d_data, size);

    // Copy data to device
    cudaMemcpy(d_data, data.data(), size, cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    squareElements<<<numBlocks, blockSize>>>(d_data, n);

    // Copy back results
    cudaMemcpy(data.data(), d_data, size, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_data);
}

int main() {
    const int n = 1 << 20; // 1 million elements
    std::vector<int> data(n, 2); // Initialize all elements to 2

    auto start = std::chrono::high_resolution_clock::now();
    parallelTransformCUDA(data);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end - start;
    std::cout << "Time taken with CUDA transformation: " << duration.count() << " seconds" << std::endl;

    return 0;
}
