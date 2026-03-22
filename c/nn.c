/*
 * nn.c  —  S4D Galaxy Classifier: Core Inference Pipeline
 * Task 3.1.7 (End-to-End Forward Pass)  |  Milestone 2
 *
 * All layer implementations + model_forward() master function.
 * Uses static memory only (no malloc). Tanh-based GELU per rubric.
 *
 * Compile together with main.c:
 *   gcc -O2 -o s4d_classifier nn.c main.c -lm
 */
#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "nn.h"

/* ============ Static Weight Storage (no malloc) ============ */
static int   hilbert_idx[SEQ_LEN];
static float up_w[D_MODEL];         /* uproject weight (64,1) flattened */
static float up_b[D_MODEL];         /* uproject bias   (64)            */
static S4DParams s4[2];             /* two S4D layers                  */
static float fc_w[NUM_CLASSES][D_MODEL];
static float fc_b[NUM_CLASSES];

/* ============ Static Intermediate Buffers (reused) ============ */
static float post_hilbert[SEQ_LEN];
static float buf_a[SEQ_LEN][D_MODEL];   /* reusable buffer A */
static float buf_b[SEQ_LEN][D_MODEL];   /* reusable buffer B */
static float s4_kernel[D_MODEL][SEQ_LEN];
static float post_pool[D_MODEL];

/* ================================================================
 * GELU — Tanh Approximation (Milestone 2 Rubric §3.1.4)
 *   GELU(x) ≈ 0.5x (1 + tanh(sqrt(2/π)(x + 0.044715 x³)))
 * ================================================================ */
static double gelu_tanh(double x) {
    const double SQRT_2_OVER_PI = 0.7978845608028654;  /* sqrt(2/pi) */
    const double COEFF = 0.044715;
    double inner = SQRT_2_OVER_PI * (x + COEFF * x * x * x);
    return 0.5 * x * (1.0 + tanh(inner));
}

/* ================================================================
 * Weight Loading — reads model_weights.bin in state_dict() order
 * ================================================================ */
int load_weights(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "ERROR: cannot open %s\n", path); return 0; }

    /* 1. Hilbert indices [4096], stored as float -> cast to int */
    float tmp[SEQ_LEN];
    if (fread(tmp, 4, SEQ_LEN, f) != SEQ_LEN) goto fail;
    for (int i = 0; i < SEQ_LEN; i++)
        hilbert_idx[i] = (int)(tmp[i] + 0.5f);

    /* 2-3. UProject weight [64,1] and bias [64] */
    if (fread(up_w, 4, D_MODEL, f) != D_MODEL) goto fail;
    if (fread(up_b, 4, D_MODEL, f) != D_MODEL) goto fail;

    /* 4-13. Two S4D layers */
    for (int layer = 0; layer < 2; layer++) {
        S4DParams *p = &s4[layer];
        if (fread(p->log_dt,     4, D_MODEL,          f) != (size_t)D_MODEL)          goto fail;
        if (fread(p->log_A_real, 4, D_MODEL * N_HALF,  f) != (size_t)(D_MODEL*N_HALF)) goto fail;
        if (fread(p->A_imag,     4, D_MODEL * N_HALF,  f) != (size_t)(D_MODEL*N_HALF)) goto fail;

        /* C is interleaved [64][32][2] -> split real / imag */
        float c_raw[D_MODEL][N_HALF][2];
        if (fread(c_raw, 4, D_MODEL * N_HALF * 2, f) != (size_t)(D_MODEL*N_HALF*2)) goto fail;
        for (int h = 0; h < D_MODEL; h++)
            for (int n = 0; n < N_HALF; n++) {
                p->C_re[h][n] = c_raw[h][n][0];
                p->C_im[h][n] = c_raw[h][n][1];
            }

        if (fread(p->D_skip, 4, D_MODEL, f) != (size_t)D_MODEL) goto fail;
    }

    /* 14-15. FC weight [4,64] and bias [4] */
    if (fread(fc_w, 4, NUM_CLASSES * D_MODEL, f) != (size_t)(NUM_CLASSES*D_MODEL)) goto fail;
    if (fread(fc_b, 4, NUM_CLASSES,           f) != (size_t)NUM_CLASSES)           goto fail;

    fclose(f);
    return 1;
fail:
    fprintf(stderr, "ERROR: unexpected EOF reading %s\n", path);
    fclose(f);
    return 0;
}

/* ================================================================
 * CSV Loading — format: label,pixel_0,pixel_1,...,pixel_4095
 * ================================================================ */
int load_image(const char *path, float *pixels, int *label) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "ERROR: cannot open %s\n", path); return 0; }
    if (fscanf(f, "%d", label) != 1) { fclose(f); return 0; }
    for (int i = 0; i < SEQ_LEN; i++) {
        char sep;
        if (fscanf(f, "%c%f", &sep, &pixels[i]) != 2) { fclose(f); return 0; }
    }
    fclose(f);
    return 1;
}

/* ================================================================
 * Stage 1 — Hilbert Scan: (64,64) -> (4096)
 * ================================================================ */
void hilbert_scan_layer(const float *img, float *out) {
    for (int t = 0; t < SEQ_LEN; t++)
        out[t] = img[hilbert_idx[t]];
}

/* ================================================================
 * Stage 2 — Input Projection (Linear): (4096,1) -> (4096,64)
 *   y[t][h] = x[t] * weight[h] + bias[h]
 * ================================================================ */
void input_projection(const float *in, float (*out)[D_MODEL]) {
    for (int t = 0; t < SEQ_LEN; t++)
        for (int h = 0; h < D_MODEL; h++)
            out[t][h] = in[t] * up_w[h] + up_b[h];
}

/* ================================================================
 * Stages 3,5 — S4D Layer (convolutional mode)
 *   1) Build kernel k[h][t] via complex SSM formula
 *   2) Causal convolution: y = conv(k, u) + u * D
 * ================================================================ */
void s4d_layer(const S4DParams *p,
               float (*in)[D_MODEL],
               float (*out)[D_MODEL])
{
    /* --- Kernel generation (double precision for accuracy) --- */
    for (int h = 0; h < D_MODEL; h++) {
        double dt = exp((double)p->log_dt[h]);
        double Ct_re[N_HALF], Ct_im[N_HALF];
        double dtA_re[N_HALF], dtA_im[N_HALF];

        for (int n = 0; n < N_HALF; n++) {
            double A_re = -exp((double)p->log_A_real[h][n]);
            double A_im = (double)p->A_imag[h][n];
            dtA_re[n] = A_re * dt;
            dtA_im[n] = A_im * dt;

            double e_mag = exp(dtA_re[n]);
            double edtA_re = e_mag * cos(dtA_im[n]);
            double edtA_im = e_mag * sin(dtA_im[n]);

            double num_re = edtA_re - 1.0;
            double num_im = edtA_im;

            double Cr = (double)p->C_re[h][n];
            double Ci = (double)p->C_im[h][n];
            double cn_re = Cr * num_re - Ci * num_im;
            double cn_im = Cr * num_im + Ci * num_re;

            double denom = A_re * A_re + A_im * A_im;
            Ct_re[n] = (cn_re * A_re + cn_im * A_im) / denom;
            Ct_im[n] = (cn_im * A_re - cn_re * A_im) / denom;
        }

        /* k[h][t] = 2 * Re( sum_n C_tilde[n] * exp(dtA[n] * t) ) */
        for (int t = 0; t < SEQ_LEN; t++) {
            double k_re = 0.0;
            for (int n = 0; n < N_HALF; n++) {
                double e_mag = exp(dtA_re[n] * t);
                double angle = dtA_im[n] * t;
                double e_re  = e_mag * cos(angle);
                double e_im  = e_mag * sin(angle);
                k_re += Ct_re[n] * e_re - Ct_im[n] * e_im;
            }
            s4_kernel[h][t] = (float)(2.0 * k_re);
        }
    }

    /* --- Causal convolution + skip connection --- */
    for (int h = 0; h < D_MODEL; h++) {
        for (int t = 0; t < SEQ_LEN; t++) {
            float sum = 0.0f;
            for (int s = 0; s <= t; s++)
                sum += s4_kernel[h][s] * in[t - s][h];
            out[t][h] = sum + in[t][h] * p->D_skip[h];
        }
    }
}

/* ================================================================
 * Stages 4,6 — GELU Activation (in-place, Tanh approximation)
 * ================================================================ */
void gelu_inplace(float (*buf)[D_MODEL]) {
    for (int t = 0; t < SEQ_LEN; t++)
        for (int h = 0; h < D_MODEL; h++)
            buf[t][h] = (float)gelu_tanh((double)buf[t][h]);
}

/* ================================================================
 * Stage 7 — TakeLastTimestep: (4096,64) -> (64)
 * ================================================================ */
void take_last(float (*buf)[D_MODEL], float *out) {
    for (int h = 0; h < D_MODEL; h++)
        out[h] = buf[SEQ_LEN - 1][h];
}

/* ================================================================
 * Stage 8 — FC Layer (Linear): (64) -> (4)
 * ================================================================ */
void fc_forward(const float *in, float *out) {
    for (int c = 0; c < NUM_CLASSES; c++) {
        out[c] = fc_b[c];
        for (int h = 0; h < D_MODEL; h++)
            out[c] += fc_w[c][h] * in[h];
    }
}

/* ================================================================
 * Stage 9 — Softmax: (4) -> (4)   (numerically stable)
 * ================================================================ */
void softmax_forward(const float *in, float *out, int n) {
    float mx = in[0];
    for (int i = 1; i < n; i++) if (in[i] > mx) mx = in[i];
    float s = 0.0f;
    for (int i = 0; i < n; i++) { out[i] = (float)exp((double)(in[i] - mx)); s += out[i]; }
    for (int i = 0; i < n; i++) out[i] /= s;
}

/* ================================================================
 * model_forward() — Master Pipeline (Task 3.1.7)
 *
 *   Chains all 9 stages end-to-end using static buffers.
 *   Input:  image  — raw 64×64 pixel array (4096 floats)
 *   Output: probs  — 4-class probability distribution
 *           logits — raw logits (before softmax)
 *   Returns: predicted class index (argmax of probs)
 *
 * Buffer reuse:  buf_a and buf_b alternate to save memory.
 *   Stage 2 writes to buf_a
 *   Stage 3 reads buf_a, writes buf_b
 *   Stage 4 GELU buf_b in-place
 *   Stage 5 reads buf_b, writes buf_a
 *   Stage 6 GELU buf_a in-place
 * ================================================================ */
int model_forward(const float *image, float *probs, float *logits) {
    /* 1. Hilbert Scan */
    hilbert_scan_layer(image, post_hilbert);

    /* 2. Input Projection */
    input_projection(post_hilbert, buf_a);

    /* 3. S4D Layer 1 */
    s4d_layer(&s4[0], buf_a, buf_b);

    /* 4. GELU 1 (in-place on buf_b) */
    gelu_inplace(buf_b);

    /* 5. S4D Layer 2 */
    s4d_layer(&s4[1], buf_b, buf_a);

    /* 6. GELU 2 (in-place on buf_a) */
    gelu_inplace(buf_a);

    /* 7. TakeLastTimestep */
    take_last(buf_a, post_pool);

    /* 8. FC Layer */
    fc_forward(post_pool, logits);

    /* 9. Softmax */
    softmax_forward(logits, probs, NUM_CLASSES);

    /* Argmax */
    int predicted = 0;
    for (int i = 1; i < NUM_CLASSES; i++)
        if (probs[i] > probs[predicted]) predicted = i;

    return predicted;
}
/* ================================================================
 * Profiling Helper (Windows / Portable)
 * ================================================================ */
#ifdef _WIN32
#include <windows.h>
double get_time() {
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart / freq.QuadPart;
}
#else
#include <time.h>
double get_time() {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}
#endif

/* ================================================================
 * model_forward_profiled() — Benchmark version (Task 3.3)
 *   Records duration of each stage in seconds into layer_times[7].
 * ================================================================ */
int model_forward_profiled(const float *image, float *probs, float *logits, double *layer_times) {
    double start, end;

    /* 1. Hilbert Scan */
    start = get_time();
    hilbert_scan_layer(image, post_hilbert);
    layer_times[0] = get_time() - start;

    /* 2. Input Projection */
    start = get_time();
    input_projection(post_hilbert, buf_a);
    layer_times[1] = get_time() - start;

    /* 3. S4D Layer 1 */
    start = get_time();
    s4d_layer(&s4[0], buf_a, buf_b);
    layer_times[2] = get_time() - start;

    /* 4. GELU 1 */
    start = get_time();
    gelu_inplace(buf_b);
    layer_times[3] = get_time() - start;

    /* 5. S4D Layer 2 */
    start = get_time();
    s4d_layer(&s4[1], buf_b, buf_a);
    layer_times[4] = get_time() - start;

    /* 6. GELU 2 */
    start = get_time();
    gelu_inplace(buf_a);
    layer_times[5] = get_time() - start;

    /* 7. TakeLast + FC + Softmax */
    start = get_time();
    take_last(buf_a, post_pool);
    fc_forward(post_pool, logits);
    softmax_forward(logits, probs, NUM_CLASSES);
    layer_times[6] = get_time() - start;

    /* Argmax */
    int predicted = 0;
    for (int i = 1; i < NUM_CLASSES; i++)
        if (probs[i] > probs[predicted]) predicted = i;

    return predicted;
}
