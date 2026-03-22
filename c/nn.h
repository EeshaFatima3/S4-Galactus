/*
 * nn.h  —  S4D Galaxy Classifier: Core Declarations
 * Task 3.1.7 (End-to-End Forward Pass)  |  Milestone 2
 */
#ifndef NN_H
#define NN_H

#include <stddef.h>  // for size_t

/* ============ Model Constants ============ */
#define SEQ_LEN     4096
#define D_MODEL     64
#define N_HALF      32
#define NUM_CLASSES 4
#define IMG_DIM     64
#define C_IN        1   /* grayscale input */

/* ============ S4D Parameter Structure ============ */
typedef struct {
    float log_dt[D_MODEL];
    float log_A_real[D_MODEL][N_HALF];
    float A_imag[D_MODEL][N_HALF];
    float C_re[D_MODEL][N_HALF];
    float C_im[D_MODEL][N_HALF];
    float D_skip[D_MODEL];
} S4DParams;

/* ============ Weight / Data I/O ============ */
int load_weights(const char *path);
int load_image(const char *path, float *pixels, int *label);

/* ============ Individual Layer Functions ============ */

/* Hilbert scan: converts 2D image into 1D sequence via Hilbert indices */
void hilbert_scan(const float image[C_IN][IMG_DIM][IMG_DIM],
                  const int* hilbert_indices,
                  float sequence[SEQ_LEN][C_IN]);

/* Linear layer forward: Y = X W^T + b */
void linear_forward(const float *X,
                    const float *W,
                    const float *b,
                    float *Y,
                    size_t L, size_t Din, size_t Dout);

/* S4D recurrent / convolutional layer forward */
void s4d_forward(const S4DParams* params,
                 const float* input,
                 float* output,
                 int L, int D);

/* GELU activation (in-place, tanh approximation) */
void gelu_inplace(float (*buf)[D_MODEL]);

/* Take last timestep from sequence: (L,D) -> (D) */
void take_last_timestep(const float* input, float* output, int L, int D);

/* Fully-connected linear layer: (D) -> (NUM_CLASSES) */
void fc_forward(const float *in, float *out);

/* Softmax: convert logits -> probability distribution */
void softmax(float* logits, int num_classes);

/* ============ Master Forward Pass ============ */
int model_forward(const float *image, float *probs, float *logits);

#endif /* NN_H */
