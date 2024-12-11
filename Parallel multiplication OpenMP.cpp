#include <iostream>
#include <vector>
#include <omp.h>
#include <chrono>

// Function to perform parallelized matrix multiplication using OpenMP
void matrixMultiplyOpenMP(const std::vector<std::vector<int>>& A,
                          const std::vector<std::vector<int>>& B,
                          std::vector<std::vector<int>>& C) {
    int n = A.size(); // Get the number of rows/columns
    #pragma omp parallel for collapse(2) // Parallelize nested loops
    for (int i = 0; i < n; i++) { // Loop over rows of A
        for (int j = 0; j < n; j++) { // Loop over columns of B
            C[i][j] = 0; // Initialize result matrix
            for (int k = 0; k < n; k++) { // Dot product of row and column
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// Function to perform matrix multiplication without OpenMP
void matrixMultiply(const std::vector<std::vector<int>>& A,
                    const std::vector<std::vector<int>>& B,
                    std::vector<std::vector<int>>& C) {
    int n = A.size(); // Get the number of rows/columns
    for (int i = 0; i < n; i++) { // Loop over rows of A
        for (int j = 0; j < n; j++) { // Loop over columns of B
            C[i][j] = 0; // Initialize result matrix
            for (int k = 0; k < n; k++) { // Dot product of row and column
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

int main() {
    int n = 120; // Dimension of square matrices (n x n)

    // Initialize matrices A, B, and C
    std::vector<std::vector<int>> A(n, std::vector<int>(n, 1)); // Matrix A with all elements = 1
    std::vector<std::vector<int>> B(n, std::vector<int>(n, 2)); // Matrix B with all elements = 2
    std::vector<std::vector<int>> C(n, std::vector<int>(n, 0)); // Result matrix initialized to 0

    // Measure time for matrix multiplication with OpenMP
    auto start = std::chrono::high_resolution_clock::now();
    matrixMultiplyOpenMP(A, B, C);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationOpenMP = end - start;
    std::cout << "Time taken with OpenMP: " << durationOpenMP.count() << " seconds\n";

    // Measure time for matrix multiplication without OpenMP
    start = std::chrono::high_resolution_clock::now();
    matrixMultiply(A, B, C);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Time taken without OpenMP: " << duration.count() << " seconds\n";

    return 0;
}
