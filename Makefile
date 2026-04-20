CC = gcc
CFLAGS = -Wall -O3 -I./include
LDFLAGS = -lopenblas -lm

all: v0

v0: v0/main.c v0/knn.c | bin
	$(CC) $(CFLAGS) -o bin/knn_v0 v0/main.c v0/knn.c $(LDFLAGS)

bin:
	mkdir -p bin

clean:
	rm -rf bin
