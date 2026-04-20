#ifndef KNN_H
#define KNN_H

// Computes the k nearest neighbors of Q in C
// C   : Corpus set, size (n_C x d)
// Q   : Query set, size (n_Q x d)
// n_C : Number of points in C
// n_Q : Number of points in Q
// d   : Dimensions
// k   : Number of neighbors
// idx : Output indices, size (n_Q x k)
// dst : Output distances, size (n_Q x k)
void knn_v0(double* C, double* Q, int n_C, int n_Q, int d, int k, int* idx, double* dst);

#endif // KNN_H