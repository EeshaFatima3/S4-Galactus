#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <iomanip>

// --- PART 1: COMPLEX MATH ENGINE ---
struct Complex {
    float real, imag;

    Complex operator+(const Complex& other) const {
        return {real + other.real, imag + other.imag};
    }
    Complex operator*(const Complex& other) const {
        return {real * other.real - imag * other.imag, 
                real * other.imag + imag * other.real};
    }
    static Complex exp(Complex c) {
        float e_a = std::exp(c.real);
        return {e_a * std::cos(c.imag), e_a * std::sin(c.imag)};
    }
};

// --- PART 2: TASK 3.1.6 (TAKE LAST TIMESTEP) ---
void take_last_timestep(const std::vector<float>& input, std::vector<float>& output, int L, int D) {
    for (int d = 0; d < D; d++) {
        // Correct (L, D) shape indexing: skip (L-1) rows of length D
        output[d] = input[(L - 1) * D + d];
    }
}

// --- PART 3: TASK 3.1.5 (SOFTMAX) ---
void softmax(std::vector<float>& logits) {
    float max_val = logits[0];
    for (float val : logits) {
        if (val > max_val) max_val = val;
    }

    float sum = 0.0f;
    for (size_t i = 0; i < logits.size(); i++) {
        logits[i] = std::exp(logits[i] - max_val);
        sum += logits[i];
    }

    for (size_t i = 0; i < logits.size(); i++) {
        logits[i] /= sum;
    }
}

// --- PART 4: TASK 3.1.3 (S4D LAYER KERNEL) ---
void compute_s4d_channel(const std::vector<float>& u, std::vector<float>& y, 
                         float log_delta, float log_A_real, float A_imag, 
                         Complex C, float D, int L) {
    std::vector<float> kernel(L, 0.0f);
    float delta = std::exp(log_delta);
    
    Complex lambda = {-std::exp(log_A_real), A_imag};
    Complex A_bar = Complex::exp({lambda.real * delta, lambda.imag * delta});

    Complex A_pow = {1.0f, 0.0f}; 
    for (int t = 0; t < L; t++) {
        Complex val = C * A_pow;
        kernel[t] = 2.0f * val.real;
        A_pow = A_pow * A_bar; 
    }

    for (int k = 0; k < L; k++) {
        float conv_sum = 0.0f;
        for (int j = 0; j <= k; j++) {
            conv_sum += kernel[j] * u[k - j];
        }
        y[k] = conv_sum + (D * u[k]);
    }
}

// --- PART 5: FILE READING & TESTING ---
std::vector<float> read_binary(const std::string& filename, size_t expected_size) {
    std::vector<float> data(expected_size);
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Could not open " << filename << "\n";
        exit(1);
    }
    file.read(reinterpret_cast<char*>(data.data()), expected_size * sizeof(float));
    return data;
}

int main() {
    int L = 4096;
    int D = 64;
    int num_classes = 4;
    
    // ==========================================
    // TEST 1: Task 3.1.6 (TakeLastTimestep)
    // ==========================================
    std::cout << "--- Testing Task 3.1.6: TakeLastTimestep ---\n";
    std::cout << "Loading sync_layer2_out.bin...\n";
    std::vector<float> layer2_out = read_binary("sync_layer2_out.bin", L * D);
    
    std::cout << "Loading sync_post_pool.bin (Golden Tensor)...\n";
    std::vector<float> golden_pool = read_binary("sync_post_pool.bin", D);
    
    std::vector<float> my_pool_out(D, 0.0f);
    take_last_timestep(layer2_out, my_pool_out, L, D);
    
    double mse_pool = 0.0;
    for(int d = 0; d < D; d++) {
        double diff = my_pool_out[d] - golden_pool[d];
        mse_pool += (diff * diff);
    }
    mse_pool /= D; 
    
    std::cout << std::scientific << std::setprecision(10);
    std::cout << "Mean Squared Error (MSE): " << mse_pool << "\n";
    if (mse_pool < 1e-12) {
        std::cout << "SUCCESS: TakeLastTimestep matches Python perfectly!\n";
    } else {
        std::cout << "FAILED: MSE is too high. Check your indexing logic.\n";
    }

    // ==========================================
    // TEST 2: Task 3.1.5 (Softmax)
    // ==========================================
    std::cout << "\n--- Testing Task 3.1.5: Softmax ---\n";
    std::cout << "Loading sync_logits.bin...\n";
    std::vector<float> my_logits = read_binary("sync_logits.bin", num_classes);
    
    std::cout << "Loading sync_probs.bin (Golden Tensor)...\n";
    std::vector<float> golden_probs = read_binary("sync_probs.bin", num_classes);

    softmax(my_logits);

    double mse_softmax = 0.0;
    for(int i = 0; i < num_classes; i++) {
        double diff = my_logits[i] - golden_probs[i];
        mse_softmax += (diff * diff);
    }
    mse_softmax /= num_classes;

    std::cout << "Mean Squared Error (MSE): " << mse_softmax << "\n";
    if (mse_softmax < 1e-12) {
        std::cout << "SUCCESS: Softmax matches Python perfectly!\n";
    } else {
        std::cout << "FAILED: MSE is too high.\n";
    }

    return 0;
}