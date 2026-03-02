#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "linearlayer.h"

// Utility function to count floats in a text file
size_t count_floats_in_file(const char *filename) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        perror("Cannot open file");
        exit(1);
    }
    size_t count = 0;
    float tmp;
    while (fscanf(fp, "%f", &tmp) == 1) {
        count++;
    }
    fclose(fp);
    return count;
}

// Load floats from a text file into an array
void load_floats(const char *filename, float *array, size_t N) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        perror(filename);
        exit(1);
    }
    for (size_t i = 0; i < N; i++) {
        if (fscanf(fp, "%f", &array[i]) != 1) {
            fprintf(stderr, "Error reading element %zu from %s\n", i, filename);
            exit(1);
        }
    }
    fclose(fp);
}

int main() {
    const char *input_file = "sample_input.txt";
    const char *weight_file = "input_projection_weight.txt";
    const char *bias_file   = "input_projection_bias.txt";
    const char *expected_file = "expected_output_linear.txt";

    // Count number of floats in input and weight
    size_t num_input_floats = count_floats_in_file(input_file);
    size_t num_weight_floats = count_floats_in_file(weight_file);
    size_t num_bias_floats = count_floats_in_file(bias_file);

    size_t Din = num_weight_floats / num_bias_floats; // weight shape: Dout x Din
    size_t Dout = num_bias_floats;
    size_t L = num_input_floats / Din;

    printf("Detected shapes: L=%zu, Din=%zu, Dout=%zu\n", L, Din, Dout);

    // Allocate arrays
    float *X = malloc(sizeof(float) * L * Din);
    float *W = malloc(sizeof(float) * Dout * Din);
    float *b = malloc(sizeof(float) * Dout);
    float *Y = malloc(sizeof(float) * L * Dout);
    float *Y_expected = malloc(sizeof(float) * L * Dout);

    if (!X || !W || !b || !Y || !Y_expected) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }

    // Load files
    load_floats(input_file, X, L * Din);
    load_floats(weight_file, W, Dout * Din);
    load_floats(bias_file, b, Dout);
    load_floats(expected_file, Y_expected, L * Dout);

    // Compute linear forward
    linear_forward(X, W, b, Y, L, Din, Dout);

    // Compute MSE and MAE
    double mse = 0.0, mae = 0.0;
    for (size_t i = 0; i < L * Dout; i++) {
        double diff = Y[i] - Y_expected[i];
        mse += diff * diff;
        mae += fabs(diff);
    }
    mse /= (L * Dout);
    mae /= (L * Dout);

    printf("Linear Layer Validation:\n");
    printf("MSE = %.10f\n", mse);
    printf("MAE = %.10f\n", mae);

    free(X); free(W); free(b); free(Y); free(Y_expected);
    return 0;
}