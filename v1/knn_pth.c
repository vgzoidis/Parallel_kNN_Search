#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <cblas.h>
#include "../include/knn.h"

// Helper function to swap elements
static void swap(double* val1, double* val2, int* idx1, int* idx2) {
    double temp_val = *val1;
    *val1 = *val2;
    *val2 = temp_val;
    int temp_idx = *idx1;
    *idx1 = *idx2;
    *idx2 = temp_idx;
}

// Partition for QuickSelect
static int partition(double* arr, int* idx, int low, int high) {
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
static void quickSelect(double* arr, int* idx, int low, int high, int k) {
    if (low < high) {
        int pi = partition(arr, idx, low, high);
        if (pi == k) return;
        else if (pi > k) quickSelect(arr, idx, low, pi - 1, k);
        else quickSelect(arr, idx, pi + 1, high, k);
    }
}

// Struct for Pthreads arguments
typedef struct {
    double* C;
    double* Q;
    double* C2;
    int n_C;
    int n_Q;
    int d;
    int k;
    int start_Q_idx;
    int end_Q_idx;
    int* out_idx;
    double* out_dst;
} thread_args_t;

void* knn_worker(void* args) {
    thread_args_t* targs = (thread_args_t*)args;
    
    // Disable nested OpenBLAS multithreading to eliminate overhead context-thrashing
    openblas_set_num_threads(1);

    double* Q_row = (double*)malloc(targs->d * sizeof(double));
    double* D_row = (double*)malloc(targs->n_C * sizeof(double));
    int* row_idx  = (int*)malloc(targs->n_C * sizeof(int));

    for (int i = targs->start_Q_idx; i < targs->end_Q_idx; i++) {
        double Q2 = 0;
        for (int j = 0; j < targs->d; j++) {
            Q_row[j] = targs->Q[i * targs->d + j];
            Q2 += Q_row[j] * Q_row[j];
        }

        // D_row = -2 * C * Q^T
        cblas_dgemv(CblasRowMajor, CblasNoTrans, targs->n_C, targs->d, -2.0, targs->C, targs->d, Q_row, 1, 0.0, D_row, 1);

        // Add C^2 and Q^2 to form distance squared, calculate sqrt, and init indices
        for (int j = 0; j < targs->n_C; j++) {
            D_row[j] = sqrt(fabs(D_row[j] + targs->C2[j] + Q2));
            row_idx[j] = j;
        }

        // 3. QuickSelect to find the top k
        quickSelect(D_row, row_idx, 0, targs->n_C - 1, targs->k);

        // Sort the top k elements to maintain exact ordering
        for (int x = 0; x < targs->k - 1; x++) {
            for (int y = 0; y < targs->k - x - 1; y++) {
                if (D_row[y] > D_row[y + 1]) {
                    swap(&D_row[y], &D_row[y + 1], &row_idx[y], &row_idx[y + 1]);
                }
            }
        }

        // Store results
        for (int j = 0; j < targs->k; j++) {
            targs->out_dst[i * targs->k + j] = D_row[j];
            targs->out_idx[i * targs->k + j] = row_idx[j];
        }
    }

    free(Q_row);
    free(D_row);
    free(row_idx);
    pthread_exit(NULL);
}

void knn_v1_pth(double* C, double* Q, int n_C, int n_Q, int d, int k, int* idx, double* dst) {
    // 1. Calculate C^2 (n_C x 1)
    double* C2 = (double*)malloc(n_C * sizeof(double));
    for (int i = 0; i < n_C; i++) {
        C2[i] = 0;
        for (int j = 0; j < d; j++) C2[i] += C[i * d + j] * C[i * d + j];
    }

    int n_threads = 4; // Default
    char* env_threads = getenv("PTH_NUM_THREADS");
    if (env_threads) {
        n_threads = atoi(env_threads);
        if (n_threads <= 0) n_threads = 4;
    }
    
    pthread_t* threads = (pthread_t*)malloc(n_threads * sizeof(pthread_t));
    thread_args_t* targs = (thread_args_t*)malloc(n_threads * sizeof(thread_args_t));

    int chunk = n_Q / n_threads;
    int remainder = n_Q % n_threads;

    int current_idx = 0;
    for (int i = 0; i < n_threads; i++) {
        targs[i].C = C;
        targs[i].Q = Q;
        targs[i].C2 = C2;
        targs[i].n_C = n_C;
        targs[i].n_Q = n_Q;
        targs[i].d = d;
        targs[i].k = k;
        targs[i].out_idx = idx;
        targs[i].out_dst = dst;

        targs[i].start_Q_idx = current_idx;
        int pts_for_thread = chunk + (i < remainder ? 1 : 0);
        targs[i].end_Q_idx = current_idx + pts_for_thread;
        current_idx = targs[i].end_Q_idx;

        pthread_create(&threads[i], NULL, knn_worker, (void*)&targs[i]);
    }

    for (int i = 0; i < n_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    free(C2);
    free(threads);
    free(targs);
}