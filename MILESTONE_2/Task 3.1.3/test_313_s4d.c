#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "task_313_s4d.h"

float* read_binary(const char* filename, int size) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("error opening %s\n", filename);
        exit(1);
    }
    float* data = (float*)malloc(size * sizeof(float));
    fread(data, sizeof(float), size, file);
    fclose(file);
    return data;
}

int main() {
    int l = 4096;
    int d = 64;

    printf("\n--- task 3.1.3: s4d layer validation ---\n");
    
    float* weights = read_binary("weights_s4d_l1.bin", 8320);
    float* u       = read_binary("input_s4d_l1.bin", l * d);
    float* y_ref   = read_binary("output_s4d_layer1_ref.bin", l * d);
    float* y_out   = (float*)calloc(l * d, sizeof(float));

    printf("computing...\n");
    s4d_forward(weights, u, y_out, l, d);

    double mse = 0.0;
    double mae = 0.0;
    for (int i = 0; i < l * d; i++) {
        double diff = y_out[i] - y_ref[i];
        mse += (diff * diff);
        mae += fabs(diff);
    }
    mse /= (l * d);
    mae /= (l * d);

    printf("mse: %e\n", mse);
    printf("mae: %e\n", mae);

    if (mse < 1e-7 && mae < 1e-4) {
        printf("success\n");
    } else {
        printf("failed\n");
    }

    free(weights); free(u); free(y_ref); free(y_out);
    return 0;
}