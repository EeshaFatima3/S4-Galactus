#ifndef LINEAR_LAYER_H
#define LINEAR_LAYER_H

#include <stddef.h>

void linear_forward(const float *X, const float *W, const float *b,
                    float *Y, size_t L, size_t Din, size_t Dout);

#endif 