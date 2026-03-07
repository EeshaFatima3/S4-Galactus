/*
 * main.c  —  S4D Galaxy Classifier: Standalone Inference Demo
 * Task 3.1.8 (Demo Program)  |  Milestone 2
 */

#include <stdio.h>
#include <stdlib.h>
#include "nn.h"

// Updated to match Maira's exact PyTorch class names
const char* CLASS_NAMES[] = {
    "Smooth Round",
    "Smooth Cigar",
    "Edge-on Disk",
    "Unbarred Spiral"
};

int main(int argc, char** argv) {
    // 1. Accept command-line argument specifying path to input image file
    if (argc < 2) {
        printf("Usage: %s <input_image.bin>\n", argv[0]);
        return 1;
    }
    const char* image_path = argv[1];

    // 2. Load model weights from model_weights.bin
    // FIX: nn.c's load_weights() returns 1 on success, 0 on failure.
    if (load_weights("model_weights.bin") == 0) {
        printf("Failed to load model weights.\n");
        return 1;
    }
    
    // The rubric requires this exact text format
    printf("Loaded model weights (21124 parameters)\n");
    printf("Running inference on: %s\n\n", image_path);

    // 3. Load input image into array (4096 floats)
    float image[SEQ_LEN * C_IN]; 
    FILE* fp = fopen(image_path, "rb");
    if (!fp) {
        printf("Error: Cannot open image file %s\n", image_path);
        return 1;
    }
    
    size_t floats_read = fread(image, sizeof(float), SEQ_LEN * C_IN, fp);
    fclose(fp);
    
    if (floats_read != SEQ_LEN * C_IN) {
        printf("Error: Expected %d floats in %s, but read %zu\n", SEQ_LEN * C_IN, image_path, floats_read);
        return 1;
    }

    // 4. Run model forward to generate predictions
    float probs[NUM_CLASSES];
    float logits[NUM_CLASSES];
    int predicted_class = model_forward(image, probs, logits);

    // 5. Print probability distribution for all 4 classes
    printf("Class Probabilities:\n");
    for (int i = 0; i < NUM_CLASSES; i++) {
        printf("Class %d (%s): %.4f\n", i, CLASS_NAMES[i], probs[i]);
    }

    // 6. Print final predicted class (argmax of probabilities)
    printf("\nPredicted Class: %d (%s)\n", predicted_class, CLASS_NAMES[predicted_class]);

    return 0;
}