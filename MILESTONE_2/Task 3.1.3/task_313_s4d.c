#include "task_313_s4d.h"
#include <math.h>

typedef struct {
    float r;
    float i;
} Complex;

static Complex c_add(Complex a, Complex b) { return (Complex){a.r + b.r, a.i + b.i}; }
static Complex c_sub(Complex a, Complex b) { return (Complex){a.r - b.r, a.i - b.i}; }
static Complex c_mul(Complex a, Complex b) { 
    return (Complex){a.r * b.r - a.i * b.i, a.r * b.i + a.i * b.r}; 
}
static Complex c_div(Complex a, Complex b) {
    float den = b.r * b.r + b.i * b.i;
    return (Complex){(a.r * b.r + a.i * b.i) / den, (a.i * b.r - a.r * b.i) / den};
}
static Complex c_exp(Complex c) {
    float e = expf(c.r);
    return (Complex){e * cosf(c.i), e * sinf(c.i)};
}

void s4d_forward(const float* weights, const float* input, float* output, int l, int d) {
    int n_complex = 32;


    // layout: log_dt -> log_A_real -> A_imag -> C_real -> C_imag -> D
    const float* p_log_dt   = weights;
    const float* p_log_a_re = p_log_dt + 64;
    const float* p_a_im     = p_log_a_re + 2048;
    const float* p_c_re     = p_a_im + 2048;
    const float* p_c_im     = p_c_re + 2048;
    const float* p_d        = p_c_im + 2048;

    for (int h = 0; h < d; h++) {
        float kernel[4096] = {0}; 
        float dt = expf(p_log_dt[h]); 

        for (int n = 0; n < n_complex; n++) {
            int idx = h * n_complex + n;
            
            float lambda_real = -expf(p_log_a_re[idx]);
            Complex a = {lambda_real, p_a_im[idx]}; 
            Complex c = {p_c_re[idx], p_c_im[idx]};

            Complex dta = c_mul(a, (Complex){dt, 0.0f});
            Complex exp_dta = c_exp(dta);
            
            Complex one = {1.0f, 0.0f};
            Complex c_tilde = c_div(c_mul(c, c_sub(exp_dta, one)), a);

            Complex a_pow = {1.0f, 0.0f};
            for (int t = 0; t < l; t++) {
                Complex term = c_mul(c_tilde, a_pow);
                kernel[t] += 2.0f * term.r;
                a_pow = c_mul(a_pow, exp_dta);
            }
        }

        float d_val = p_d[h];
        for (int k = 0; k < l; k++) {
            float sum = 0.0f;
            for (int j = 0; j <= k; j++) {
                sum += kernel[j] * input[(k - j) * d + h];
            }
            output[k * d + h] = sum + (d_val * input[k * d + h]);
        }
    }
}