#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cblas.h>
#include "../include/knn.h"

// Helper function to swap elements
void swap(double* val1, double* val2, int* idx1, int* idx2) {
    double temp_val = *val1;
    *val1 = *val2;
    *val2 = temp_val;
    int temp_idx = *idx1;
    *idx1 = *idx2;
    *idx2 = temp_idx;
}

// Partition for QuickSelect
int partition(double* arr, int* idx, int low, int high) {
    double pivot = arr[high];
    int i = (low - 1);
    for (int j = low; j <= high - 1; j++) {
        if (arr[j] <= pivot) {
            i++;
            swap(&arr[i], &arr[j], &idx[i], &idx[j]);
        }
    }
    swap(&arr[i + 1], &arr[high], &idx[i + 1], &idx[high]);
    return (i + 1);
}

// QuickSelect to find the k-th smallest element
void quickSelect(double* arr, int* idx, int low, int high, int k) {
    if (low < high) {
        int pi = partition(arr, idx, low, high);
        if (pi == k) return;
        else if (pi > k) quickSelect(arr, idx, low, pi - 1, k);
        else quickSelect(arr, idx, pi + 1, high, k);
    }
}

void knn_v0(double* C, double* Q, int n_C, int n_Q, int d, int k, int* idx, double* dst) {
    // 1. Calculate C^2 (n_C x 1)
    double* C2 = (double*)malloc(n_C * sizeof(double));
    for (int i = 0; i < n_C; i++) {
        C2[i] = 0;
        for (int j = 0; j < d; j++) C2[i] += C[i * d + j] * C[i * d + j];
    }

    // 2. Iterate through each query in Q to avoid building a huge D matrix memory
    double* Q_row = (double*)malloc(d * sizeof(double));
    double* D_row = (double*)malloc(n_C * sizeof(double));
    int* row_idx  = (int*)malloc(n_C * sizeof(int));

    for (int i = 0; i < n_Q; i++) {
        double Q2 = 0;
        for (int j = 0; j < d; j++) {
            Q_row[j] = Q[i * d + j];
            Q2 += Q_row[j] * Q_row[j];
        }

        // D_row = -2 * C * Q^T
        // cblas_dgemv (Matrix-Vector multiplication)
        cblas_dgemv(CblasRowMajor, CblasNoTrans, n_C, d, -2.0, C, d, Q_row, 1, 0.0, D_row, 1);

        // Add C^2 and Q^2 to form distance squared, calculate sqrt, and init indices
        for (int j = 0; j < n_C; j++) {
            D_row[j] = sqrt(fabs(D_row[j] + C2[j] + Q2));
            row_idx[j] = j;
        }

        // 3. QuickSelect to find the top k
        quickSelect(D_row, row_idx, 0, n_C - 1, k);

        // Sort the top k elements to maintain exact ordering (optional but good for testing)
        // (A simple bubble sort since k is usually very small)
        for (int x = 0; x < k - 1; x++) {
            for (int y = 0; y < k - x - 1; y++) {
                if (D_row[y] > D_row[y + 1]) {
                    swap(&D_row[y], &D_row[y + 1], &row_idx[y], &row_idx[y + 1]);
                }
            }
        }

        // Store results
        for (int j = 0; j < k; j++) {
            dst[i * k + j] = D_row[j];
            idx[i * k + j] = row_idx[j];
        }
    }

    free(C2);
    free(Q_row);
    free(D_row);
    free(row_idx);
}