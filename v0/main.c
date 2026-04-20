#include <stdio.h>
#include <stdlib.h>
#include "../include/knn.h"

int main() {
    int n_C = 10;
    int n_Q = 3;
    int d = 2;
    int k = 2;

    double C[] = {0,0,  1,1,  2,2,  3,3,  4,4,  5,5,  6,6,  7,7,  8,8,  9,9};
    double Q[] = {0.1,0.1,  4.1,4.1,  8.1,8.1};

    int* idx = (int*)malloc(n_Q * k * sizeof(int));
    double* dst = (double*)malloc(n_Q * k * sizeof(double));

    knn_v0(C, Q, n_C, n_Q, d, k, idx, dst);

    for (int i = 0; i < n_Q; i++) {
        printf("Query %d nearest neighbors:\n", i);
        for (int j = 0; j < k; j++) {
            printf("  Node %d (dist: %f)\n", idx[i * k + j], dst[i * k + j]);
        }
    }

    free(idx);
    free(dst);
    return 0;
}