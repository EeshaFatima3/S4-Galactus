/*
 * nn.c  —  S4D Galaxy Classifier: Core Inference Pipeline
 * Task 3.1.7 (End-to-End Forward Pass)  |  Milestone 2
 */
#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "nn.h"

/* ================== Static Weight Storage ================== */
static int hilbert_idx[SEQ_LEN];
static float up_w[D_MODEL * C_IN];
static float up_b[D_MODEL];
static S4DParams s4_layer1;
static S4DParams s4_layer2;
static float fc_w[NUM_CLASSES][D_MODEL];
static float fc_b[NUM_CLASSES];

/* ================== Static Buffers ================== */
static float post_hilbert[SEQ_LEN][C_IN];
static float buf_a[SEQ_LEN][D_MODEL];
static float buf_b[SEQ_LEN][D_MODEL];
static float post_pool[D_MODEL];

/* ================== Weight Loading ================== */
int load_weights(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        printf("Error: Cannot open %s\n", path);
        return -1;
    }
    
    // Corrected: Read Hilbert indices directly as raw integers (like your original code!)
    if (fread(hilbert_idx, sizeof(int), SEQ_LEN, fp) != SEQ_LEN) {
        printf("Error reading Hilbert indices\n"); return -1;
    }
    
    // Read uproject weight and bias
    if (fread(up_w, sizeof(float), D_MODEL * C_IN, fp) != D_MODEL * C_IN) {
        printf("Error reading uproject weight\n"); return -1;
    }
    if (fread(up_b, sizeof(float), D_MODEL, fp) != D_MODEL) {
        printf("Error reading uproject bias\n"); return -1;
    }
    
    // Corrected: Read S4D Layer 1 as contiguous sequential blocks (like your original code!)
    if (fread(s4_layer1.log_dt, sizeof(float), D_MODEL, fp) != D_MODEL ||
        fread(s4_layer1.log_A_real, sizeof(float), D_MODEL * N_HALF, fp) != D_MODEL * N_HALF ||
        fread(s4_layer1.A_imag, sizeof(float), D_MODEL * N_HALF, fp) != D_MODEL * N_HALF ||
        fread(s4_layer1.C_re, sizeof(float), D_MODEL * N_HALF, fp) != D_MODEL * N_HALF ||
        fread(s4_layer1.C_im, sizeof(float), D_MODEL * N_HALF, fp) != D_MODEL * N_HALF ||
        fread(s4_layer1.D_skip, sizeof(float), D_MODEL, fp) != D_MODEL) {
        printf("Error reading S4D Layer 1\n"); return -1;
    }
    
    // Corrected: Read S4D Layer 2 as contiguous sequential blocks
    if (fread(s4_layer2.log_dt, sizeof(float), D_MODEL, fp) != D_MODEL ||
        fread(s4_layer2.log_A_real, sizeof(float), D_MODEL * N_HALF, fp) != D_MODEL * N_HALF ||
        fread(s4_layer2.A_imag, sizeof(float), D_MODEL * N_HALF, fp) != D_MODEL * N_HALF ||
        fread(s4_layer2.C_re, sizeof(float), D_MODEL * N_HALF, fp) != D_MODEL * N_HALF ||
        fread(s4_layer2.C_im, sizeof(float), D_MODEL * N_HALF, fp) != D_MODEL * N_HALF ||
        fread(s4_layer2.D_skip, sizeof(float), D_MODEL, fp) != D_MODEL) {
        printf("Error reading S4D Layer 2\n"); return -1;
    }
    
    // Read FC weight and bias
    if (fread(fc_w, sizeof(float), NUM_CLASSES * D_MODEL, fp) != NUM_CLASSES * D_MODEL ||
        fread(fc_b, sizeof(float), NUM_CLASSES, fp) != NUM_CLASSES) {
        printf("Error reading FC layer\n"); return -1;
    }
    
    fclose(fp);
    return 0;
}

/* ================== Complex Math ================== */
typedef struct { float r, i; } Complex;
static Complex c_add(Complex a, Complex b) { return (Complex){a.r + b.r, a.i + b.i}; }
static Complex c_sub(Complex a, Complex b) { return (Complex){a.r - b.r, a.i - b.i}; }
static Complex c_mul(Complex a, Complex b) { return (Complex){a.r*b.r - a.i*b.i, a.r*b.i + a.i*b.r}; }
static Complex c_div(Complex a, Complex b) {
    float den = b.r*b.r + b.i*b.i;
    if (den == 0.0f) return (Complex){0, 0};
    return (Complex){(a.r*b.r + a.i*b.i)/den, (a.i*b.r - a.r*b.i)/den};
}
static Complex c_exp(Complex c) {
    float e = expf(c.r);
    return (Complex){e*cosf(c.i), e*sinf(c.i)};
}

/* ================== Hilbert Scan ================== */
void hilbert_scan(const float image[C_IN][IMG_DIM][IMG_DIM],
                  const int* hilbert_indices,
                  float sequence[SEQ_LEN][C_IN])
{
    for (int d = 0; d < SEQ_LEN; d++) {
        int idx = hilbert_indices[d]; 
        int row = idx / IMG_DIM;
        int col = idx % IMG_DIM;
        for (int c = 0; c < C_IN; c++) {
            sequence[d][c] = image[c][row][col];
        }
    }
}

/* ================== Linear Layer ================== */
void linear_forward(const float *X, const float *W, const float *b,
                    float *Y, size_t L, size_t Din, size_t Dout)
{
    for (size_t i = 0; i < L; i++) {
        for (size_t j = 0; j < Dout; j++) {
            Y[i * Dout + j] = b[j];
            for (size_t k = 0; k < Din; k++) {
                Y[i * Dout + j] += X[i * Din + k] * W[j * Din + k];
            }
        }
    }
}

/* ================== S4D Forward ================== */
void s4d_forward(const S4DParams* params, const float* input, 
                 float* output, int L, int D)
{
    int n_complex = N_HALF;
    for (int h = 0; h < D; h++) {
        float kernel[4096] = {0};
        float dt = expf(params->log_dt[h]);
        
        for (int n = 0; n < n_complex; n++) {
            float lambda_real = -expf(params->log_A_real[h][n]);
            Complex a = {lambda_real, params->A_imag[h][n]};
            Complex c = {params->C_re[h][n], params->C_im[h][n]};
            Complex dta = c_mul(a, (Complex){dt, 0.0f});
            Complex exp_dta = c_exp(dta);
            Complex one = {1.0f, 0.0f};
            Complex c_tilde = c_div(c_mul(c, c_sub(exp_dta, one)), a);
            
            Complex a_pow = {1.0f, 0.0f};
            for (int t = 0; t < L && t < 4096; t++) {
                Complex term = c_mul(c_tilde, a_pow);
                kernel[t] += 2.0f * term.r;
                a_pow = c_mul(a_pow, exp_dta);
            }
        }
        
        float d_val = params->D_skip[h];
        for (int k = 0; k < L; k++) {
            float sum = 0.0f;
            for (int j = 0; j <= k; j++) {
                sum += kernel[j] * input[(k-j) * D + h];
            }
            output[k * D + h] = sum + (d_val * input[k * D + h]);
        }
    }
}

/* ================== GELU ================== */
void gelu_inplace(float (*buf)[D_MODEL])
{
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coeff = 0.044715f;
    for (int t = 0; t < SEQ_LEN; t++) {
        for (int h = 0; h < D_MODEL; h++) {
            float x = buf[t][h];
            float x3 = x * x * x;
            buf[t][h] = 0.5f * x * (1.0f + tanhf(sqrt_2_over_pi * (x + coeff * x3)));
        }
    }
}

/* ================== Take Last Timestep ================== */
void take_last_timestep(const float* input, float* output, int L, int D)
{
    int offset = (L - 1) * D;
    for (int d = 0; d < D; d++) {
        output[d] = input[offset + d];
    }
}

/* ================== FC Layer ================== */
void fc_forward(const float* in, float* out)
{
    for (int c = 0; c < NUM_CLASSES; c++) {
        out[c] = fc_b[c];
        for (int h = 0; h < D_MODEL; h++) {
            out[c] += fc_w[c][h] * in[h];
        }
    }
}

/* ================== Softmax ================== */
void softmax(float* logits, int num_classes)
{
    float max_val = logits[0];
    for (int i = 1; i < num_classes; i++) {
        if (logits[i] > max_val) max_val = logits[i];
    }
    
    float sum = 0.0f;
    for (int i = 0; i < num_classes; i++) {
        logits[i] = expf(logits[i] - max_val);
        sum += logits[i];
    }
    
    if (sum > 0) {
        for (int i = 0; i < num_classes; i++) {
            logits[i] /= sum;
        }
    }
}

/* ================== Model Forward ================== */
int model_forward(const float* image, float* probs, float* logits)
{
    hilbert_scan((const float(*)[IMG_DIM][IMG_DIM])image, hilbert_idx, post_hilbert);
    linear_forward(&post_hilbert[0][0], up_w, up_b, &buf_a[0][0], SEQ_LEN, C_IN, D_MODEL);
    s4d_forward(&s4_layer1, &buf_a[0][0], &buf_b[0][0], SEQ_LEN, D_MODEL);
    gelu_inplace(buf_b);
    s4d_forward(&s4_layer2, &buf_b[0][0], &buf_a[0][0], SEQ_LEN, D_MODEL);
    gelu_inplace(buf_a);
    take_last_timestep(&buf_a[0][0], post_pool, SEQ_LEN, D_MODEL);
    fc_forward(post_pool, logits);
    memcpy(probs, logits, NUM_CLASSES * sizeof(float));
    softmax(probs, NUM_CLASSES);
    
    int predicted = 0;
    for (int i = 1; i < NUM_CLASSES; i++) {
        if (probs[i] > probs[predicted]) predicted = i;
    }
    return predicted;
}