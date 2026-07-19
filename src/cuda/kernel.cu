#include <cuda_runtime.h>
#include <stdio.h>

#define checkCudaError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        printf("GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    }
}

// The Core RNN Memory Engine
extern "C" __global__ void rnn_forward_kernel(
    const float* x_embed, // The current letter (as a math vector)
    const float* h_prev,  // The AI's previous memory
    const float* w_xh,    // Weights for the letter
    const float* w_hh,    // Weights for the memory
    const float* b_h,     // Bias
    float* h_next,        // The NEW memory we are calculating
    int batch_size,
    int embed_dim,
    int hidden_dim
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Which text sequence are we on?
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Which memory cell are we calculating?
    
    if (row < batch_size && col < hidden_dim) {
        float sum = b_h[col];
        
        // 1. Multiply the Current Letter by its weights
        for (int k = 0; k < embed_dim; ++k) {
            sum += x_embed[row * embed_dim + k] * w_xh[k * hidden_dim + col];
        }
        
        // 2. Multiply the Past Memory by its weights (THIS IS THE MAGIC!)
        for (int k = 0; k < hidden_dim; ++k) {
            sum += h_prev[row * hidden_dim + k] * w_hh[k * hidden_dim + col];
        }
        
        // 3. Apply Tanh Activation (Squishes the new memory safely between -1.0 and 1.0)
        h_next[row * hidden_dim + col] = tanhf(sum);
    }
}

// 1. CUDA Kernel for MatMul Forward Pass: out = x * w + b
// Grid dimensions will map threads to: row (batch), col (out_dim)
__global__ void matmul_forward_kernel(
    float* out,        // [Output] Destination for layer outputs, shape: (batch, out_dim)
    const float* x,    // [Input] Layer inputs, shape: (batch, in_dim)
    const float* w,    // [Input] Weights, shape: (in_dim, out_dim)
    const float* b,    // [Input] Biases, shape: (out_dim)
    int batch,         // Batch size (number of training examples processed at once)
    int in_dim,        // Dimension of inputs (number of features coming in)
    int out_dim        // Dimension of outputs (number of neurons in this layer)
) {
    // blockIdx: Block Index, blockDim: Threads per Block, threadIdx: Thread Index
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Maps this thread to a specific batch row
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Maps this thread to a specific output column

    // Make sure the thread is within matrix boundaries
    if (row < batch && col < out_dim) {
        float sum = (b != nullptr) ? b[col] : 0.0f;
        for (int i = 0; i < in_dim; i++) {
            // Dot product: Row of x * Column of w
            sum += x[row * in_dim + i] * w[i * out_dim + col];
        }
        out[row * out_dim + col] = sum; // Write final result to memory
    }
}

// 2. CUDA Kernel to compute dW = X^T * dZ
// Grid dimensions map threads to: row (in_dim), col (out_dim)
__global__ void matmul_backward_weight_kernel(
    float* dw,         // [Output] Destination for weight gradients, shape: (in_dim, out_dim)
    const float* x,    // [Input] Original layer inputs, shape: (batch, in_dim)
    const float* dz,   // [Input] Incoming output gradients, shape: (batch, out_dim)
    int batch,         // Batch size
    int in_dim,        // Input dimension
    int out_dim        // Output dimension
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Maps to weight row (input dimension index)
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Maps to weight column (output dimension index)

    if (row < in_dim && col < out_dim) {
        float sum = 0.0f;
        for (int b = 0; b < batch; b++) {
            // Transpose multiplication: sum over batch of (X[b, row] * dZ[b, col])
            sum += x[b * in_dim + row] * dz[b * out_dim + col];
        }
        dw[row * out_dim + col] = sum; // Save computed gradient for this weight
    }
}

// 3. CUDA Kernel to compute dB = sum(dZ, axis=0)
// Grid dimensions map threads to: col (out_dim)
__global__ void matmul_backward_bias_kernel(
    float* db,         // [Output] Destination for bias gradients, shape: (out_dim)
    const float* dz,   // [Input] Incoming output gradients, shape: (batch, out_dim)
    int batch,         // Batch size
    int out_dim        // Output dimension
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Maps to bias vector index

    if (col < out_dim) {
        float sum = 0.0f;
        for (int b = 0; b < batch; b++) {
            // Sum the gradients for this output neuron across the entire batch
            sum += dz[b * out_dim + col];
        }
        db[col] = sum; // Save computed gradient for this bias
    }
}

// 4. CUDA Kernel to compute dX = dZ * W^T
// Grid dimensions map threads to: row (batch), col (in_dim)
__global__ void matmul_backward_input_kernel(
    float* dx,         // [Output] Destination for input gradients, shape: (batch, in_dim)
    const float* dz,   // [Input] Incoming output gradients, shape: (batch, out_dim)
    const float* w,    // [Input] Layer weights, shape: (in_dim, out_dim)
    int batch,         // Batch size
    int in_dim,        // Input dimension
    int out_dim        // Output dimension
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Maps to batch row
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Maps to input column

    if (row < batch && col < in_dim) {
        float sum = 0.0f;
        for (int o = 0; o < out_dim; o++) {
            // Transpose multiplication: sum over outputs of (dZ[row, o] * W[col, o])
            sum += dz[row * out_dim + o] * w[col * out_dim + o];
        }
        dx[row * in_dim + col] = sum; // Save computed gradient to pass to lower layer
    }
}

// 5. CUDA Kernel for ReLU Forward: out = max(0, inp)
__global__ void relu_forward_kernel(
    float* out,        // [Output] Destination for activated values
    const float* inp,  // [Input] Input pre-activation values
    int size           // Total number of elements in the tensor (batch * dimension)
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Flat global index for this thread
    if (idx < size) {
        float val = inp[idx];
        out[idx] = val > 0.0f ? val : 0.0f; // Zero out negative values
    }
}

// 6. CUDA Kernel for ReLU Backward: dinp = dout * [inp > 0]
__global__ void relu_backward_kernel(
    float* dinp,       // [Output] Destination for gradients w.r.t input pre-activation
    const float* dout, // [Input] Gradients w.r.t activated outputs
    const float* inp,  // [Input] Original pre-activation values (to check if they were > 0)
    int size           // Total number of elements in the tensor
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Flat global index
    if (idx < size) {
        // If the original input was positive, pass the gradient through. Otherwise, it is 0.
        dinp[idx] = inp[idx] > 0.0f ? dout[idx] : 0.0f;
    }
}

// 7. CUDA Kernel for SGD Update Step: param = param - lr * grad
__global__ void sgd_step_kernel(
    float* param,      // [In/Out] Parameter to update (weights or biases)
    const float* grad, // [Input] Computed gradients for this parameter
    float lr,          // Learning rate (size of step to take)
    int size           // Total number of elements in this parameter
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Flat global index
    if (idx < size) {
        param[idx] -= lr * grad[idx]; // Update the parameter
    }
}

__global__ void layernorm_forward_kernel(
    float* out,
    float* cache_mean,
    float* cache_var,
    const float* inp,
    const float* gamma,
    const float* beta,
    int batch,
    int dim,
    float eps
){
    int b = blockIdx.x * blockDim.x + threadIdx.x;

    if(b < batch){
        const float* x = inp + b * dim;
        float* y = out +b * dim;

        // 1. Calculate the Average (Mean) of this row
        float sum = 0;
        for(int i = 0;i < dim;i++){
            sum += x[i];
        }
        float mean = sum/dim;

        // 2. Calculate the Variance (How spread out the numbers are)
        float var_sum = 0.0f;
        for(int i = 0; i < dim;i++){
            float diff = x[i] - mean;
            var_sum += diff*diff;
        }
        float var = var_sum / dim ; 

        // Save these so Backprop can use them later!
        cache_mean[b] = mean;
        cache_var[b] = var;

        // rsqrtf is a hardware command for "1.0 / sqrt(x)"
        float inv_std = rsqrtf(var + eps);

        // 3. Normalize the row, then multiply by Gamma and add Beta
        for (int i = 0; i < dim; i++) {
            float x_hat = (x[i] - mean) * inv_std;
            y[i] = gamma[i] * x_hat + beta[i];
        }
        
    }
}

// 9. CUDA Kernel for LayerNorm Backward
__global__ void layernorm_backward_kernel(
    float* dinp, float* dgamma, float* dbeta,
    const float* dout, const float* inp, const float* cache_mean, const float* cache_var,
    const float* gamma, int batch, int dim, float eps
){
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (b < batch) {
        const float* x = inp + b * dim;
        const float* dy = dout + b * dim;
        float* dx = dinp + b * dim;
        
        float mean = cache_mean[b];
        float var = cache_var[b];
        float inv_std = rsqrtf(var + eps);
        
        float sum_dy_gamma = 0.0f;
        float sum_dy_gamma_xhat = 0.0f;
        
        for (int i = 0; i < dim; i++) {
            float dy_g = dy[i] * gamma[i];
            float x_hat = (x[i] - mean) * inv_std;
            sum_dy_gamma += dy_g;
            sum_dy_gamma_xhat += dy_g * x_hat;
            
            // atomicAdd safely adds the blame scores together across all batch threads!
            atomicAdd(&dgamma[i], dy[i] * x_hat);
            atomicAdd(&dbeta[i], dy[i]);
        }
        
        // Calculate the blame score for the original inputs
        for (int i = 0; i < dim; i++) {
            float dy_g = dy[i] * gamma[i];
            float x_hat = (x[i] - mean) * inv_std;
            dx[i] = inv_std * (dy_g - (sum_dy_gamma / dim) - x_hat * (sum_dy_gamma_xhat / dim));
        }
    }
}
// 10. CUDA Kernel for AdamW Optimizer Step
__global__ void adamw_step_kernel(
    float* param, float* m, float* v, const float* grad,
    float lr, float beta1, float beta2, float eps, float weight_decay,
    int step, int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float g = grad[idx]; // The raw blame score
        
        // 1. Apply a tiny bit of friction (Weight Decay)
        param[idx] -= lr * weight_decay * param[idx];
        
        // 2. Update our history of Momentum (m) and Velocity (v)
        float mt = beta1 * m[idx] + (1.0f - beta1) * g;
        float vt = beta2 * v[idx] + (1.0f - beta2) * g * g;
        m[idx] = mt;
        v[idx] = vt;
        
        // 3. Math trick to fix the history during the very first few steps
        float m_hat = mt / (1.0f - powf(beta1, step));
        float v_hat = vt / (1.0f - powf(beta2, step));
        
        // 4. Finally, update the actual weight using the history!
        param[idx] -= lr * m_hat / (sqrtf(v_hat) + eps);
    }
}

// ==========================================
// 2. FFI LAUNCH WRAPPERS (C LINKAGE)
// ==========================================
// extern "C" prevents C++ name mangling, allowing Rust to link directly to these names.

extern "C"{
    void* gpu_alloc(size_t size){
        void* device_ptr = nullptr;
        cudaMalloc(&device_ptr,size);
        return device_ptr;
    }

   // GPU Memory Free
void gpu_free(void* device_ptr) {
    if (device_ptr != nullptr) {
        cudaFree(device_ptr);
    }
}
// Copy CPU host memory to GPU device memory
void gpu_copy_to_device(void* device_dest, const void* host_src, size_t size) {
    cudaMemcpy(device_dest, host_src, size, cudaMemcpyHostToDevice);
}
// Copy GPU device memory to CPU host memory
void gpu_copy_to_host(void* host_dest, const void* device_src, size_t size) {
    cudaMemcpy(host_dest, device_src, size, cudaMemcpyDeviceToHost);
}
// Launcher for MatMul Forward
void launch_matmul_forward(
    float* out, const float* x, const float* w, const float* b,
    int batch, int in_dim, int out_dim
) {
    // We choose a block size of 16x16 threads (256 threads total per block)
    dim3 block(16, 16);
    // Calculate the number of blocks needed in the grid to cover all output cells
    dim3 grid((out_dim + block.x - 1) / block.x, (batch + block.y - 1) / block.y);
    matmul_forward_kernel<<<grid, block>>>(out, x, w, b, batch, in_dim, out_dim);
    checkCudaError(cudaDeviceSynchronize()); // Ensure execution finishes before returning to Rust
}
// Launcher for MatMul Backward Weight
void launch_matmul_backward_weight(
    float* dw, const float* x, const float* dz,
    int batch, int in_dim, int out_dim
) {
    dim3 block(16, 16);
    dim3 grid((out_dim + block.x - 1) / block.x, (in_dim + block.y - 1) / block.y);
    matmul_backward_weight_kernel<<<grid, block>>>(dw, x, dz, batch, in_dim, out_dim);
    checkCudaError(cudaDeviceSynchronize());
}
// Launcher for MatMul Backward Bias
void launch_matmul_backward_bias(
    float* db, const float* dz, int batch, int out_dim
) {
    // 1D block because bias is a 1D vector
    int block = 256;
    int grid = (out_dim + block - 1) / block;
    matmul_backward_bias_kernel<<<grid, block>>>(db, dz, batch, out_dim);
    checkCudaError(cudaDeviceSynchronize());
}
// Launcher for MatMul Backward Input
void launch_matmul_backward_input(
    float* dx, const float* dz, const float* w,
    int batch, int in_dim, int out_dim
) {
    dim3 block(16, 16);
    dim3 grid((in_dim + block.x - 1) / block.x, (batch + block.y - 1) / block.y);
    matmul_backward_input_kernel<<<grid, block>>>(dx, dz, w, batch, in_dim, out_dim);
    checkCudaError(cudaDeviceSynchronize());
}
// Launcher for ReLU Forward
void launch_relu_forward(float* out, const float* inp, int size) {
    int block = 256;
    int grid = (size + block - 1) / block;
    relu_forward_kernel<<<grid, block>>>(out, inp, size);
    cudaDeviceSynchronize();
}
// Launcher for ReLU Backward
void launch_relu_backward(float* dinp, const float* dout, const float* inp, int size) {
    int block = 256;
    int grid = (size + block - 1) / block;
    relu_backward_kernel<<<grid, block>>>(dinp, dout, inp, size);
    cudaDeviceSynchronize();
}
// Launcher for SGD Step
void launch_sgd_step(float* param, const float* grad, float lr, int size) {
    int block = 256;
    int grid = (size + block - 1) / block;
    sgd_step_kernel<<<grid, block>>>(param, grad, lr, size);
    cudaDeviceSynchronize();
}

// Launcher for LayerNorm Forward
void launch_layernorm_forward(
    float* out, float* cache_mean, float* cache_var,
    const float* inp, const float* gamma, const float* beta,
    int batch, int dim, float eps
) {
    int block = 256;
    int grid = (batch + block - 1) / block;
    layernorm_forward_kernel<<<grid, block>>>(out, cache_mean, cache_var, inp, gamma, beta, batch, dim, eps);
    cudaDeviceSynchronize();
}
// Launcher for LayerNorm Backward
void launch_layernorm_backward(
    float* dinp, float* dgamma, float* dbeta,
    const float* dout, const float* inp, const float* cache_mean, const float* cache_var,
    const float* gamma, int batch, int dim, float eps
) {
    int block = 256;
    int grid = (batch + block - 1) / block;
    layernorm_backward_kernel<<<grid, block>>>(dinp, dgamma, dbeta, dout, inp, cache_mean, cache_var, gamma, batch, dim, eps);
    cudaDeviceSynchronize();
}
// Launcher for AdamW Step
void launch_adamw_step(
    float* param, float* m, float* v, const float* grad,
    float lr, float beta1, float beta2, float eps, float weight_decay,
    int step, int size
) {
    int block = 256;
    int grid = (size + block - 1) / block;
    adamw_step_kernel<<<grid, block>>>(param, m, v, grad, lr, beta1, beta2, eps, weight_decay, step, size);
    cudaDeviceSynchronize();
}
// Helper function to wipe memory to 0.0 before atomicAdd
void gpu_memset(void* device_ptr, int value, size_t size) {
    cudaMemset(device_ptr, value, size);
}
}// extern "C"

// BPTT: Calculates the raw blame score by taking the derivative of Tanh
extern "C" __global__ void rnn_tanh_derivative_kernel(
    const float* dh,        // The Blame Score passed backward from the future
    const float* h_t,       // The snapshot of the AI's memory at this exact point in time
    float* dh_raw,          // The final Blame Score after Calculus
    int batch_size,
    int hidden_dim
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < batch_size && col < hidden_dim) {
        int idx = row * hidden_dim + col;
        
        // Calculus! The derivative of Tanh is: 1.0 - (h * h)
        float h_val = h_t[idx];
        float derivative = 1.0f - (h_val * h_val);
        
        // Multiply the incoming blame by the derivative!
        dh_raw[idx] = dh[idx] * derivative;
    }
}
// Phase 4: Google's Self-Attention Equation
extern "C" __global__ void attention_forward_kernel(
    const float* Q,
    const float* K,
    const float* V,
    float* Output,
    int seq_len,
    int embed_dim
) {
    // Which letter in the sentence are we calculating? (e.g. Letter 0 to 24)
    int row = blockIdx.x * blockDim.x + threadIdx.x; 
    
    if (row < seq_len) {
        
        // 1. Calculate Q * K (How much do I care about the other letters?)
        float scores[256]; 
        float max_score = -1e9f;
        
        for (int i = 0; i < seq_len; ++i) {
            if (i > row) {
                scores[i] = -1e9f;
                continue;
            }
            float sum = 0.0f;
            for (int d = 0; d < embed_dim; ++d) {
                sum += Q[row * embed_dim + d] * K[i * embed_dim + d]; // Q * K
            }
            // Scale it down so it doesn't explode (sqrt of 64 is 8.0)
            sum = sum / 8.0f; 
            scores[i] = sum;
            if (sum > max_score) max_score = sum;
        }
        
        // 2. Softmax (Turn the raw scores into percentages that add up to 100%)
        float sum_exp = 0.0f;
        for (int i = 0; i < seq_len; ++i) {
            scores[i] = expf(scores[i] - max_score);
            sum_exp += scores[i];
        }
        for (int i = 0; i < seq_len; ++i) {
            scores[i] /= sum_exp;
        }
        
        // 3. Absorb the Value (V) based on the percentages!
        for (int d = 0; d < embed_dim; ++d) {
            float out_sum = 0.0f;
            for (int i = 0; i < seq_len; ++i) {
                out_sum += scores[i] * V[i * embed_dim + d]; // Percentages * V
            }
            Output[row * embed_dim + d] = out_sum;
        }
    }
}
// Phase 4.5: The Attention Backward Pass
// Cross Entropy Loss & Gradients for Language Model
extern "C" __global__ void cross_entropy_kernel(
    const float* logits,
    const int* targets,
    float* dlogits,
    float* loss_out,
    int seq_len,
    int vocab_size
) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < seq_len) {
        int target = targets[t];
        const float* logit_row = logits + t * vocab_size;
        float* dlogit_row = dlogits + t * vocab_size;
        
        float max_val = -1e9f;
        for (int v = 0; v < vocab_size; ++v) {
            if (logit_row[v] > max_val) max_val = logit_row[v];
        }
        
        float sum_exp = 0.0f;
        for (int v = 0; v < vocab_size; ++v) {
            sum_exp += expf(logit_row[v] - max_val);
        }
        
        if (target >= 0 && target < vocab_size) {
            float p_target = expf(logit_row[target] - max_val) / sum_exp;
            float loss = -logf(fmaxf(p_target, 1e-7f));
            atomicAdd(loss_out, loss / (float)seq_len);
        }
        
        for (int v = 0; v < vocab_size; ++v) {
            float p_i = expf(logit_row[v] - max_val) / sum_exp;
            float target_i = (v == target) ? 1.0f : 0.0f;
            dlogit_row[v] = (p_i - target_i) / (float)seq_len;
        }
    }
}
// Exact Self-Attention Backward Pass (dQ, dK, dV)
extern "C" __global__ void attention_backward_kernel(
    const float* dOutput, 
    const float* Q,
    const float* K,
    const float* V,
    float* dQ,
    float* dK,
    float* dV,
    int seq_len,
    int embed_dim
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x; 
    
    if (row < seq_len) {
        float scores[256];
        float max_score = -1e9f;
        for (int i = 0; i < seq_len; ++i) {
            if (i > row) {
                scores[i] = -1e9f;
                continue;
            }
            float sum = 0.0f;
            for (int d = 0; d < embed_dim; ++d) {
                sum += Q[row * embed_dim + d] * K[i * embed_dim + d];
            }
            sum = sum / 8.0f;
            scores[i] = sum;
            if (sum > max_score) max_score = sum;
        }
        
        float sum_exp = 0.0f;
        for (int i = 0; i < seq_len; ++i) {
            scores[i] = expf(scores[i] - max_score);
            sum_exp += scores[i];
        }
        for (int i = 0; i < seq_len; ++i) {
            scores[i] /= sum_exp;
        }
        // dV = P^T * dOutput
        for (int i = 0; i < seq_len; ++i) {
            for (int d = 0; d < embed_dim; ++d) {
                atomicAdd(&dV[i * embed_dim + d], scores[i] * dOutput[row * embed_dim + d]);
            }
        }
        
        // dP = dOutput * V^T
        float dP[256];
        float sum_dP_P = 0.0f;
        for (int i = 0; i < seq_len; ++i) {
            float dp_sum = 0.0f;
            for (int d = 0; d < embed_dim; ++d) {
                dp_sum += dOutput[row * embed_dim + d] * V[i * embed_dim + d];
            }
            dP[i] = dp_sum;
            sum_dP_P += dp_sum * scores[i];
        }
        // dS = P * (dP - sum(dP * P)) / sqrt(d)
        for (int i = 0; i < seq_len; ++i) {
            float dS = scores[i] * (dP[i] - sum_dP_P) / 8.0f;
            for (int d = 0; d < embed_dim; ++d) {
                atomicAdd(&dQ[row * embed_dim + d], dS * K[i * embed_dim + d]);
                atomicAdd(&dK[i * embed_dim + d], dS * Q[row * embed_dim + d]);
            }
        }
    }
}

extern "C" {
    void launch_attention_forward(const float* Q, const float* K, const float* V, float* Output, int seq_len, int embed_dim) {
        int block = 256;
        int grid = (seq_len + block - 1) / block;
        attention_forward_kernel<<<grid, block>>>(Q, K, V, Output, seq_len, embed_dim);
        checkCudaError(cudaDeviceSynchronize());
    }
    
    void launch_cross_entropy(const float* logits, const int* targets, float* dlogits, float* loss_out, int seq_len, int vocab_size) {
        int block = 256;
        int grid = (seq_len + block - 1) / block;
        cross_entropy_kernel<<<grid, block>>>(logits, targets, dlogits, loss_out, seq_len, vocab_size);
        checkCudaError(cudaDeviceSynchronize());
    }
    
    void launch_attention_backward(const float* dOutput, const float* Q, const float* K, const float* V, float* dQ, float* dK, float* dV, int seq_len, int embed_dim) {
        int block = 256;
        int grid = (seq_len + block - 1) / block;
        attention_backward_kernel<<<grid, block>>>(dOutput, Q, K, V, dQ, dK, dV, seq_len, embed_dim);
        checkCudaError(cudaDeviceSynchronize());
    }
}

// 9. CUDA Kernel for In-Place Addition (Residual Connections)
__global__ void add_inplace_kernel(
    float* dest,
    const float* src,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dest[idx] += src[idx];
    }
}

extern "C" {
    void launch_add_inplace(float* dest, const float* src, int size) {
        int block = 256;
        int grid = (size + block - 1) / block;
        add_inplace_kernel<<<grid, block>>>(dest, src, size);
        checkCudaError(cudaDeviceSynchronize());
    }
}