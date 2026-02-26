#include <stdio.h>
#include <stdlib.h>
#include "hilbert.h"

int* load_hilbert_indices(const char* filename)
{
    FILE* f = fopen(filename, "rb");
    if (!f) {
        printf("Error opening file\n");
        return NULL;
    }

    int* indices = malloc(SEQ_LEN * sizeof(int));
    fread(indices, sizeof(int), SEQ_LEN, f);
    fclose(f);

    return indices;
}

float* load_float_file(const char* filename, int count)
{
    FILE* f = fopen(filename, "rb");
    if (!f) {
        printf("Error opening %s\n", filename);
        return NULL;
    }

    float* data = malloc(count * sizeof(float));
    fread(data, sizeof(float), count, f);
    fclose(f);

    return data;
}

int main()
{
    float image[C][H][W];
    float sequence[SEQ_LEN][C];

    // Load original image from file
    float* input_data = load_float_file("input_before_hilbert.bin", C*H*W);
    if (!input_data) return 1;

    for (int c = 0; c < C; c++)
        for (int i = 0; i < H; i++)
            for (int j = 0; j < W; j++)
                image[c][i][j] = input_data[c*H*W + i*W + j];

    free(input_data);

    int* hilbert_indices = load_hilbert_indices("model_weights.bin");
    if (!hilbert_indices)
        return 1;

    hilbert_scan(image, hilbert_indices, sequence);

    printf("First 5 outputs:\n");
    for (int i = 0; i < 5; i++) {
        printf("d=%d: ", i);
        for (int c = 0; c < C; c++)
            printf("%f ", sequence[i][c]);
        printf("\n");
    }

    // Load Python reference output
    float* python_output = load_float_file("input_after_hilbert.bin", SEQ_LEN*C);
    if (!python_output) return 1;

    // Compute MSE
    double mse = 0.0;
    for (int d = 0; d < SEQ_LEN; d++) {
        for (int c = 0; c < C; c++) {
            double diff = sequence[d][c] - python_output[d*C + c];
            mse += diff * diff;
        }
    }
    mse /= (SEQ_LEN * C);
    printf("MSE = %.15f\n", mse);

    free(hilbert_indices);
    free(python_output);

    return 0;
}