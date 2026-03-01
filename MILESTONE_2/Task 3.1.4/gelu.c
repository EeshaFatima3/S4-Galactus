#include "gelu.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void gelu(const float* input, float* output, size_t size) {
    // Exact approximations
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coeff = 0.044715f;
    
    for (size_t i = 0; i < size; i++) {
        float x = input[i];
        float x3 = x * x * x;
        float inner = sqrt_2_over_pi * (x + coeff * x3);
        
        output[i] = 0.5f * x * (1.0f + tanhf(inner));  
    }
}

float compute_mse(const float* ref, const float* pred, size_t size) {
    double sum = 0.0; // double to prevent accumulation loss
    for (size_t i = 0; i < size; i++) {
        double diff = (double)ref[i] - (double)pred[i];
        sum += diff * diff;
    }
    return (float)(sum / size);
}

float compute_mae(const float* ref, const float* pred, size_t size) {
    double sum = 0.0; 
    for (size_t i = 0; i < size; i++) {
        double diff = (double)ref[i] - (double)pred[i];
        sum += fabs(diff);
    }
    return (float)(sum / size);
}

int load_bin(const char* filename, float* array, size_t size) {
    FILE* f = fopen(filename, "rb");
    if (!f) {
        printf("Failed to open %s\n", filename);
        return 0;
    }
    size_t read = fread(array, sizeof(float), size, f);
    fclose(f);
    if (read != size) {
        printf("Expected %zu elements, got %zu from %s\n", size, read, filename);
        return 0;
    }
    return 1;
}