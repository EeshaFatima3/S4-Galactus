#include <stdio.h>
#include <stdlib.h>
#include <math.h> 
#include "task_315_softmax.h"

// helper function to read binary files
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
    int num_classes = 4;

    printf("\n--- Testing Task 3.1.5: Softmax ---\n");
    
    float* my_logits = read_binary("sync_logits.bin", num_classes);
    float* golden_probs = read_binary("sync_probs.bin", num_classes);

   
    softmax(my_logits, num_classes);

    double mse = 0.0;
    double mae = 0.0; 
    float prob_sum = 0.0f; 
    
    for (int i = 0; i < num_classes; i++) {
        double diff = my_logits[i] - golden_probs[i];
        mse += (diff * diff);
        mae += fabs(diff); 
        prob_sum += my_logits[i]; // add up probs
    }
    mse /= num_classes;
    mae /= num_classes;

    printf("Softmax MSE: %e (Threshold: < 1e-8)\n", mse);
    printf("Softmax MAE: %e (Threshold: < 1e-4)\n", mae);
    
    // check if probs sum to 1.0 (within float tolerance)
    printf("Probability Sum: %f\n", prob_sum);

    // validate all conditions
    if (mse < 1e-8 && mae < 1e-4 && fabs(prob_sum - 1.0f) < 1e-5) {
        printf("SUCCESS: Softmax math is perfect, passes all validations, and sums to 1.0!\n");
    } else {
        printf("FAILED: Error metrics are too high, or sum is not 1.0.\n");
    }

    free(my_logits);
    free(golden_probs);
    return 0;
}