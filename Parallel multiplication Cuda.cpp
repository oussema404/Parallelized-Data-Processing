#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

// CUDA kernel for matrix multiplication
__global__ void matrixMultiplyCUDA(int* A, int* B, int* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        int sum = 0;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

void matrixMultiplyCUDAWrapper(const std::vector<std::vector<int>>& A,
                               const std::vector<std::vector<int>>& B,
                               std::vector<std::vector<int>>& C) {
    int n = A.size();
    int size = n * n * sizeof(int);

    // Flatten 2D vectors to 1D for CUDA
    std::vector<int> A_flat(n * n);
    std::vector<int> B_flat(n * n);
    std::vector<int> C_flat(n * n, 0);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A_flat[i * n + j] = A[i][j];
            B_flat[i * n + j] = B[i][j];
        }
    }

    // Allocate memory on the device
    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Error checking for memory allocation
    if (d_A == nullptr || d_B == nullptr || d_C == nullptr) {
        std::cerr << "CUDA memory allocation failed" << std::endl;
        return;
    }

    // Copy data to the device with error checking
    if (cudaMemcpy(d_A, A_flat.data(), size, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_B, B_flat.data(), size, cudaMemcpyHostToDevice) != cudaSuccess) {
        std::cerr << "CUDA memory copy to device failed" << std::endl;
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return;
    }

    // Define block and grid sizes
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (n + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch the kernel
    matrixMultiplyCUDA<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);

    // Check for kernel launch errors
    cudaError_t kernelError = cudaGetLastError();
    if (kernelError != cudaSuccess) {
        std::cerr << "CUDA kernel launch failed: " << cudaGetErrorString(kernelError) << std::endl;
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return;
    }

    // Copy result back to host with error checking
    if (cudaMemcpy(C_flat.data(), d_C, size, cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::cerr << "CUDA memory copy from device failed" << std::endl;
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return;
    }

    // Copy back to 2D vector
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            C[i][j] = C_flat[i * n + j];
        }
    }

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    int n = 1000; // Size of the matrix
    std::vector<std::vector<int>> A(n, std::vector<int>(n, 1));
    std::vector<std::vector<int>> B(n, std::vector<int>(n, 1));
    std::vector<std::vector<int>> C(n, std::vector<int>(n, 0));

    // Measure time for CUDA-based matrix multiplication
    auto start = std::chrono::high_resolution_clock::now();
    matrixMultiplyCUDAWrapper(A, B, C);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Time with CUDA: " << duration.count() << " seconds" << std::endl;

    return 0;
}