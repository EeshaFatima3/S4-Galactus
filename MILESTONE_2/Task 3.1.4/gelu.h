#ifndef GELU_H
#define GELU_H

#include <stddef.h>

// GELU activation
void gelu(const float* input, float* output, size_t size);

// MSE and MAE computations
float compute_mse(const float* ref, const float* pred, size_t size);
float compute_mae(const float* ref, const float* pred, size_t size);

// Load binary file
int load_bin(const char* filename, float* array, size_t size);

#endif // GELU_H