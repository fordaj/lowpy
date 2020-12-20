#include <stdio.h>
// Compile with:
// nvcc -o example example.cu
#define N 1000

__global__ void add(int *a, int *b) {
    int i = blockIdx.x;
    if (i<N) {
        b[i] = 2*a[i];
    }
}

int main() {
    // Create int arrays on the CPU.
    // ('h' stands for "host".)
    int ha[N], hb[N];
    // Create corresponding int arrays on the GPU.
    // ('d' stands for "device".)
    int *da, *db;
    cudaMalloc((void **)&da, N*sizeof(int));
    cudaMalloc((void **)&db, N*sizeof(int));
    // Initialise the input data on the CPU.
    for (int i = 0; i<N; ++i) {
        ha[i] = i;
    }
    // Copy input data to array on GPU.
    cudaMemcpy(da, ha, N*sizeof(int), cudaMemcpyHostToDevice);
    // Launch GPU code with N threads, one per
    // array element.
    add<<<N, 1>>>(da, db);
    // Copy output array from GPU back to CPU.
    cudaMemcpy(hb, db, N*sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i<N; ++i) {
        printf("%d\n", hb[i]);
    }

    //
    // Free up the arrays on the GPU.
    //
    cudaFree(da);
    cudaFree(db);

    return 0;
}