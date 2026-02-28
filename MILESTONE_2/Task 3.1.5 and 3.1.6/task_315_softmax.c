#include "task_315_softmax.h"
#include <math.h>

void softmax(float* logits, int num_classes) {
    // 1. Find max value for numerical stability
    float max_val = logits[0];
    for (int i = 1; i < num_classes; i++) {
        if (logits[i] > max_val) {
            max_val = logits[i];
        }
    }

    // 2. Compute exponentials and sum
    float sum = 0.0f;
    for (int i = 0; i < num_classes; i++) {
        logits[i] = expf(logits[i] - max_val);
        sum += logits[i];
    }

    // 3. Normalize to get probabilities
    for (int i = 0; i < num_classes; i++) {
        logits[i] /= sum;
    }
}