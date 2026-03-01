#include <stdio.h>
#include <stdlib.h>
#include "gelu.h"

int main() {
    size_t size = 4096 * 64; 

    float* input = (float*)malloc(sizeof(float) * size);
    float* output = (float*)malloc(sizeof(float) * size);
    float* ref = (float*)malloc(sizeof(float) * size);

    if (!load_bin("gelu_input_ref.bin", input, size)) return 1;
    if (!load_bin("gelu_output_ref.bin", ref, size)) return 1;

    gelu(input, output, size);

    float mse = compute_mse(ref, output, size);
    float mae = compute_mae(ref, output, size);

    printf("GELU Validation:\nMSE: %.10f\nMAE: %.10f\n", mse, mae);

    free(input); free(output); free(ref);
    return 0;
}