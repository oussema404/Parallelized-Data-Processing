#include <iostream>
#include <vector>
#include <algorithm>
#include <omp.h>
#include <chrono>

// Parallelized merge sort using OpenMP
void merge(std::vector<int>& arr, int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;
    std::vector<int> L(n1), R(n2);

    for (int i = 0; i < n1; ++i)
        L[i] = arr[left + i];
    for (int i = 0; i < n2; ++i)
        R[i] = arr[mid + 1 + i];

    int i = 0, j = 0, k = left;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k++] = L[i++];
        } else {
            arr[k++] = R[j++];
        }
    }
    while (i < n1)
        arr[k++] = L[i++];
    while (j < n2)
        arr[k++] = R[j++];
}

void mergeSort(std::vector<int>& arr, int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;

        #pragma omp parallel sections
        {
            #pragma omp section
            mergeSort(arr, left, mid);

            #pragma omp section
            mergeSort(arr, mid + 1, right);
        }

        merge(arr, left, mid, right);
    }
}

int main() {
    const int n = 1 << 20; // 1 million elements
    std::vector<int> data(n);

    // Initialize data with random values
    for (int i = 0; i < n; ++i)
        data[i] = rand();

    auto start = std::chrono::high_resolution_clock::now();
    mergeSort(data, 0, n - 1);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end - start;
    std::cout << "Time taken with OpenMP merge sort: " << duration.count() << " seconds" << std::endl;

    return 0;
}
