/*
 * benchmark.c  —  S4D Galaxy Classifier: Performance Benchmarking
 * Task 3.3 (Performance Benchmarking)  |  Milestone 2
 *
 * Measures per-layer and total inference time using high-resolution timers.
 * Runs NUM_RUNS iterations (default 100) to compute:
 *   - Average execution time per layer
 *   - Standard deviation per layer
 *   - Percentage of total inference time per layer
 *   - Total throughput (images/sec)
 *
 * Outputs results to both stdout and a CSV file for downstream analysis.
 *
 * Compile:  gcc -O2 -o benchmark benchmark.c nn.c -lm
 * Usage:    ./benchmark [model_weights.bin] [sample_input.bin] [output.csv]
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "nn.h"

#define NUM_RUNS   100
#define NUM_STAGES 7

static const char *STAGE_NAMES[NUM_STAGES] = {
    "Hilbert Scan",
    "Input Projection",
    "S4D Layer 1",
    "GELU 1",
    "S4D Layer 2",
    "GELU 2",
    "TakeLast+FC+Softmax"
};

int main(int argc, char *argv[]) {
    const char *w_path   = (argc > 1) ? argv[1] : "model_weights.bin";
    const char *img_path = (argc > 2) ? argv[2] : "sample_bins/sample_0.bin";
    const char *out_csv  = (argc > 3) ? argv[3] : "benchmark_results.csv";

    /* --- 1. Load model weights --- */
    if (!load_weights(w_path)) {
        fprintf(stderr, "Error: Cannot load weights from %s\n", w_path);
        return 1;
    }

    /* --- 2. Load input image from binary file --- */
    float image[SEQ_LEN * C_IN];
    FILE *fp = fopen(img_path, "rb");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open image file %s\n", img_path);
        return 1;
    }
    size_t n = fread(image, sizeof(float), SEQ_LEN * C_IN, fp);
    fclose(fp);
    if (n != SEQ_LEN * C_IN) {
        fprintf(stderr, "Error: Expected %d floats, got %zu\n", SEQ_LEN * C_IN, n);
        return 1;
    }

    /* --- 3. Benchmarking loop --- */
    /* layer_data[run][stage] stores the time in seconds for that stage */
    double layer_data[NUM_RUNS][NUM_STAGES];
    float  probs[NUM_CLASSES], logits[NUM_CLASSES];

    printf("==========================================================\n");
    printf("S4D Galaxy Classifier — Performance Benchmark\n");
    printf("==========================================================\n");
    printf("Weights:     %s\n", w_path);
    printf("Image:       %s\n", img_path);
    printf("Iterations:  %d\n", NUM_RUNS);
    printf("----------------------------------------------------------\n");
    printf("Running benchmark...\n");

    for (int r = 0; r < NUM_RUNS; r++) {
        model_forward_profiled(image, probs, logits, layer_data[r]);
    }

    /* --- 4. Compute statistics (Welford's online algorithm) --- */
    double avg[NUM_STAGES], stddev[NUM_STAGES];
    double total_avg = 0.0;

    for (int s = 0; s < NUM_STAGES; s++) {
        double mean = 0.0, M2 = 0.0;
        for (int r = 0; r < NUM_RUNS; r++) {
            double delta = layer_data[r][s] - mean;
            mean += delta / (r + 1);
            M2   += delta * (layer_data[r][s] - mean);
        }
        avg[s]    = mean;
        stddev[s] = sqrt(M2 / NUM_RUNS);
        total_avg += mean;
    }

    /* --- 5. Print results table --- */
    printf("\n%-25s | %12s | %12s | %10s\n",
           "Layer", "Avg (ms)", "StdDev (ms)", "% Total");
    printf("--------------------------------------------------------------\n");

    for (int s = 0; s < NUM_STAGES; s++) {
        printf("%-25s | %12.4f | %12.6f | %9.2f%%\n",
               STAGE_NAMES[s],
               avg[s] * 1000.0,
               stddev[s] * 1000.0,
               (avg[s] / total_avg) * 100.0);
    }

    printf("--------------------------------------------------------------\n");
    printf("%-25s | %12.4f | %12s | %9s\n",
           "TOTAL INFERENCE", total_avg * 1000.0, "-", "100.00%");
    printf("\nThroughput: %.4f images/sec\n", 1.0 / total_avg);
    printf("Inference per image: %.4f ms\n", total_avg * 1000.0);

    /* --- 6. Identify bottleneck --- */
    int bottleneck = 0;
    for (int s = 1; s < NUM_STAGES; s++)
        if (avg[s] > avg[bottleneck]) bottleneck = s;
    printf("\nBottleneck layer: %s (%.2f%% of total time)\n",
           STAGE_NAMES[bottleneck], (avg[bottleneck] / total_avg) * 100.0);

    /* --- 7. Write CSV for analyze_performance.py --- */
    FILE *f = fopen(out_csv, "w");
    if (f) {
        fprintf(f, "Layer,Avg_ms,StdDev_ms,Percent\n");
        for (int s = 0; s < NUM_STAGES; s++) {
            fprintf(f, "%s,%.6f,%.6f,%.4f\n",
                    STAGE_NAMES[s],
                    avg[s] * 1000.0,
                    stddev[s] * 1000.0,
                    (avg[s] / total_avg) * 100.0);
        }
        fprintf(f, "Total,%.6f,0,100.0\n", total_avg * 1000.0);
        fclose(f);
        printf("\nResults written to: %s\n", out_csv);
    }

    printf("==========================================================\n");
    return 0;
}
