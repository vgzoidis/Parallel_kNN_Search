#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "../include/knn.h"

int main(int argc, char** argv) {
    if (argc != 6) {
        printf("Usage: %s <C.bin> <Q.bin> <k> <idx.bin> <dst.bin>\n", argv[0]);
        return 1;
    }

    FILE* f_C = fopen(argv[1], "rb");
    if (!f_C) { perror("Error opening Corpus file"); return 1; }
    int n_C, d_C;
    if (fread(&n_C, sizeof(int), 1, f_C) != 1) { perror("Error reading n_C"); fclose(f_C); return 1; }
    if (fread(&d_C, sizeof(int), 1, f_C) != 1) { perror("Error reading d_C"); fclose(f_C); return 1; }
    double* C = (double*)malloc(n_C * d_C * sizeof(double));
    if (!C) { perror("Failed to allocate Corpus memory"); fclose(f_C); return 1; }
    if (fread(C, sizeof(double), n_C * d_C, f_C) != n_C * d_C) { perror("Error reading Corpus array"); free(C); fclose(f_C); return 1; }
    fclose(f_C);

    FILE* f_Q = fopen(argv[2], "rb");
    if (!f_Q) { perror("Error opening Query file"); free(C); return 1; }
    int n_Q, d_Q;
    if (fread(&n_Q, sizeof(int), 1, f_Q) != 1) { perror("Error reading n_Q"); free(C); fclose(f_Q); return 1; }
    if (fread(&d_Q, sizeof(int), 1, f_Q) != 1) { perror("Error reading d_Q"); free(C); fclose(f_Q); return 1; }
    double* Q = (double*)malloc(n_Q * d_Q * sizeof(double));
    if (!Q) { perror("Failed to allocate Query memory"); free(C); fclose(f_Q); return 1; }
    if (fread(Q, sizeof(double), n_Q * d_Q, f_Q) != n_Q * d_Q) { perror("Error reading Query array"); free(C); free(Q); fclose(f_Q); return 1; }
    fclose(f_Q);

    int k = atoi(argv[3]);
    int* idx = (int*)malloc(n_Q * k * sizeof(int));
    double* dst = (double*)malloc(n_Q * k * sizeof(double));

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    knn_v0(C, Q, n_C, n_Q, d_Q, k, idx, dst);

    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_taken = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("%.6f\n", time_taken);

    FILE* f_idx = fopen(argv[4], "wb");
    fwrite(idx, sizeof(int), n_Q * k, f_idx);
    fclose(f_idx);

    FILE* f_dst = fopen(argv[5], "wb");
    fwrite(dst, sizeof(double), n_Q * k, f_dst);
    fclose(f_dst);

    free(C); free(Q); free(idx); free(dst);
    return 0;
}
