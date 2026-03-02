#include "linearlayer.h"

// Linear layer forward pass: Y = X W^T + b
void linear_forward(const float *X, const float *W, const float *b,
                    float *Y, size_t L, size_t Din, size_t Dout)
{
    for (size_t i = 0; i < L; i++) {
        for (size_t j = 0; j < Dout; j++) {
            // Initialize with bias
            Y[i * Dout + j] = b[j];
        }
    }

    for (size_t i = 0; i < L; i++) {
        for (size_t j = 0; j < Dout; j++) {
            for (size_t k = 0; k < Din; k++) {
                Y[i * Dout + j] += X[i * Din + k] * W[j * Din + k]; // row-major
            }
        }
    }
}