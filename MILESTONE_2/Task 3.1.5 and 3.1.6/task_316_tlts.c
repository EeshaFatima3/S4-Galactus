#include "task_316_tlts.h"

void take_last_timestep(const float* input, float* output, int L, int D) {
    // Row-major offset: skip (L-1) rows of size D
    int offset = (L - 1) * D;
    for (int d = 0; d < D; d++) {
        output[d] = input[offset + d];
    }
}