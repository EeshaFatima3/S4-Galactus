/*
 * validate.c  —  Validate S4D Galaxy Classifier using CSV samples
 * * Compile: gcc -O2 -o validate nn.c validate.c -lm
 * Usage:   ./validate model_weights.bin galaxy_samples.csv
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "nn.h"

#define MAX_SAMPLES 100
// FIX 1: 4096 floats * 20 chars per float = ~80,000 chars. Increased to 100,000 to prevent lines from cutting off!
#define MAX_LINE_LENGTH 100000 

/* Parse a CSV line containing label + 4096 pixels */
int parse_csv_line(char* line, float* image, int* label) {
    char* token;
    int col = 0;
    
    // FIX 2: The FIRST token is the label in your CSV file!
    token = strtok(line, ",\n");
    if (token == NULL) return -1;
    *label = atoi(token);
    
    // The next 4096 tokens are the pixels
    token = strtok(NULL, ",\n");
    while (token != NULL && col < SEQ_LEN) {
        image[col] = atof(token);
        token = strtok(NULL, ",\n");
        col++;
    }
    
    return (col == SEQ_LEN) ? 0 : -1;
}

/* Load all samples from CSV file */
int load_samples_from_csv(const char* csv_path, 
                          float samples[][SEQ_LEN], 
                          int* labels, 
                          int max_samples) {
    FILE* fp = fopen(csv_path, "r");
    if (!fp) {
        printf("Error: Cannot open CSV file %s\n", csv_path);
        return -1;
    }
    
    // Using malloc to avoid blowing up the stack with a massive line string
    char* line = (char*)malloc(MAX_LINE_LENGTH);
    int sample_count = 0;
    
    // Read header line if present
    if (fgets(line, MAX_LINE_LENGTH, fp)) {
        if (line[0] < '0' || line[0] > '9') {
            printf("Skipping header line.\n");
        } else {
            if (parse_csv_line(line, samples[sample_count], &labels[sample_count]) == 0) {
                sample_count++;
            }
        }
    }
    
    // Read remaining lines
    while (fgets(line, MAX_LINE_LENGTH, fp) && sample_count < max_samples) {
        if (parse_csv_line(line, samples[sample_count], &labels[sample_count]) == 0) {
            sample_count++;
        }
    }
    
    free(line);
    fclose(fp);
    return sample_count;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: %s <weights_file> <csv_file>\n", argv[0]);
        return 1;
    }
    
    printf("=== S4D Galaxy Classifier Validation ===\n\n");
    
    /* Load weights */
    printf("Loading weights from %s... ", argv[1]);
    fflush(stdout);
    if (load_weights(argv[1]) == 0) { ///////////////////////////////////////////////////////////////////////
        printf("FAILED\n");
        return 1;
    }
    
    /* Load samples from CSV */
    printf("Loading samples from %s... ", argv[2]);
    fflush(stdout);
    
    float (*samples)[SEQ_LEN] = malloc(MAX_SAMPLES * sizeof(*samples));
    int* labels = malloc(MAX_SAMPLES * sizeof(int));
    
    int num_samples = load_samples_from_csv(argv[2], samples, labels, MAX_SAMPLES);
    if (num_samples <= 0) {
        printf("FAILED (no valid samples)\n");
        return 1;
    }
    printf("OK (%d samples found)\n\n", num_samples);
    
    /* Run inference on all samples */
    int correct = 0;
    printf("%-4s %-10s %-10s %-40s\n", 
           "Idx", "True", "Pred", "Probabilities [Smooth Disk Edge Irreg]");
    printf("%s\n", "------------------------------------------------------------");
    
    for (int i = 0; i < num_samples; i++) {
        float probs[NUM_CLASSES];
        float logits[NUM_CLASSES];
        
        int predicted = model_forward(samples[i], probs, logits);
        if (predicted == labels[i]) { correct++; }
        
        printf("%-4d %-10d %-10d [", i, labels[i], predicted);
        for (int j = 0; j < NUM_CLASSES; j++) {
            printf("%6.3f%s", probs[j], (j < NUM_CLASSES-1) ? " " : "");
        }
        printf("] %s\n", (predicted == labels[i]) ? "PASS" : "FAIL");
    }
    
    /* Summary */
    float accuracy = 100.0f * correct / num_samples;
    printf("\n=========================================\n");
    printf("VALIDATION SUMMARY\n");
    printf("=========================================\n");
    printf("Total samples: %d\n", num_samples);
    printf("Correct predictions: %d\n", correct);
    printf("Accuracy: %.2f%%\n", accuracy);
    
    free(samples);
    free(labels);
    
    if (accuracy == 100.0f) {
        printf("\nSUCCESS: TASK 3.1.7 PASSED (100%% accuracy)\n");
        return 0;
    } else {
        printf("\nFAILED: Expected 100%% accuracy\n");
        return 1;
    }
}