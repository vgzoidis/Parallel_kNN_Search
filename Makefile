CC = gcc
CFLAGS = -Wall -O3 -I./include
LDFLAGS = -lopenblas -lm

all: v0 v1_omp v1_pth

v0: v0/main.c v0/knn.c | bin
	$(CC) $(CFLAGS) -o bin/knn_v0 v0/main.c v0/knn.c $(LDFLAGS)

v1_omp: v1/main_omp.c v1/knn_omp.c | bin
	$(CC) $(CFLAGS) -fopenmp -o bin/knn_v1_omp v1/main_omp.c v1/knn_omp.c $(LDFLAGS)

v1_pth: v1/main_pth.c v1/knn_pth.c | bin
	$(CC) $(CFLAGS) -pthread -o bin/knn_v1_pth v1/main_pth.c v1/knn_pth.c $(LDFLAGS)

bin:
	mkdir -p bin

clean:
	rm -rf bin
