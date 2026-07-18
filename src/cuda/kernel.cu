#include <cuda_runtime.h>
#include <stdio.h>

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
        float sum = b[col];
        printf("in_dim:%d\n" ,in_dim); // Start with the bias value for this output neuron
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
    cudaDeviceSynchronize(); // Ensure execution finishes before returning to Rust
}
// Launcher for MatMul Backward Weight
void launch_matmul_backward_weight(
    float* dw, const float* x, const float* dz,
    int batch, int in_dim, int out_dim
) {
    dim3 block(16, 16);
    dim3 grid((out_dim + block.x - 1) / block.x, (in_dim + block.y - 1) / block.y);
    matmul_backward_weight_kernel<<<grid, block>>>(dw, x, dz, batch, in_dim, out_dim);
    cudaDeviceSynchronize();
}
// Launcher for MatMul Backward Bias
void launch_matmul_backward_bias(
    float* db, const float* dz, int batch, int out_dim
) {
    // 1D block because bias is a 1D vector
    int block = 256;
    int grid = (out_dim + block - 1) / block;
    matmul_backward_bias_kernel<<<grid, block>>>(db, dz, batch, out_dim);
    cudaDeviceSynchronize();
}
// Launcher for MatMul Backward Input
void launch_matmul_backward_input(
    float* dx, const float* dz, const float* w,
    int batch, int in_dim, int out_dim
) {
    dim3 block(16, 16);
    dim3 grid((in_dim + block.x - 1) / block.x, (batch + block.y - 1) / block.y);
    matmul_backward_input_kernel<<<grid, block>>>(dx, dz, w, batch, in_dim, out_dim);
    cudaDeviceSynchronize();
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
} // extern "C"