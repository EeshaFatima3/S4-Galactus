#include <stdio.h>
#include <stdlib.h>
#include "task_316_tlts.h"

// Helper function to read binary files
float* read_binary(const char* filename, int size) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error: Could not open %s\n", filename);
        exit(1);
    }
    float* data = (float*)malloc(size * sizeof(float));
    fread(data, sizeof(float), size, file);
    fclose(file);
    return data;
}

int main() {
    int L = 4096;
    int D = 64;

    printf("--- Testing Task 3.1.6: TakeLastTimestep ---\n");
    
    float* layer2_in = read_binary("sync_layer2_out.bin", L * D);
    float* golden_pool = read_binary("sync_post_pool.bin", D);
    float my_pool_out[64] = {0}; 

    take_last_timestep(layer2_in, my_pool_out, L, D);

    double mse = 0.0;
    for (int i = 0; i < D; i++) {
        double diff = my_pool_out[i] - golden_pool[i];
        mse += (diff * diff);
    }
    mse /= D;

    printf("TakeLastTimestep MSE: %e\n", mse);
    if (mse < 1e-12) {
        printf("SUCCESS: Pooling math is perfect!\n");
    } else {
        printf("FAILED: MSE too high.\n");
    }

    free(layer2_in);
    free(golden_pool);

    return 0;
}