use crate:: math_train::*;
pub struct MLP{
     // Model Parameters (Weights and Biases on GPU)
    pub w1: GPUMemory,
    pub b1: GPUMemory,
    pub w2: GPUMemory,
    pub b2: GPUMemory,
    // Parameter Gradients (computed during backward pass on GPU)
    pub dw1: GPUMemory,
    pub db1: GPUMemory,
    pub dw2: GPUMemory,
    pub db2: GPUMemory,
    // Intermediate activations (stored on GPU for backprop)
    pub z1: GPUMemory,  // Output of Linear 1 (pre-activation)
    pub a1: GPUMemory,  // Output of ReLU 1 (post-activation)
    pub z2: GPUMemory,  // Output of Linear 2 (final logits)
    // Intermediate gradients (stored on GPU)
    pub dz2: GPUMemory, // Gradient w.r.t Z2 (starting point of backward pass)
    pub da1: GPUMemory, // Gradient w.r.t A1
    pub dz1: GPUMemory, // Gradient w.r.t Z1
    pub dx: GPUMemory,  // Gradient w.r.t input X (optional, but good practice)
    // Dimensions
    pub batch_size: usize,
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub output_dim: usize,
}
impl MLP{
      pub fn new(batch_size: usize, input_dim: usize, hidden_dim: usize, output_dim: usize) -> Self {
        // 1. Allocate model parameters in VRAM
        let w1 = GPUMemory::new(input_dim * hidden_dim);
        let b1 = GPUMemory::new(hidden_dim);
        let w2 = GPUMemory::new(hidden_dim * output_dim);
        let b2 = GPUMemory::new(output_dim);
        // 2. Allocate gradients in VRAM
        let dw1 = GPUMemory::new(input_dim * hidden_dim);
        let db1 = GPUMemory::new(hidden_dim);
        let dw2 = GPUMemory::new(hidden_dim * output_dim);
        let db2 = GPUMemory::new(output_dim);
        // 3. Allocate intermediate buffers in VRAM
        let z1 = GPUMemory::new(batch_size * hidden_dim);
        let a1 = GPUMemory::new(batch_size * hidden_dim);
        let z2 = GPUMemory::new(batch_size * output_dim);
        let dz2 = GPUMemory::new(batch_size * output_dim);
        let da1 = GPUMemory::new(batch_size * hidden_dim);
        let dz1 = GPUMemory::new(batch_size * hidden_dim);
        let dx = GPUMemory::new(batch_size * input_dim);
        Self {
            w1, b1, w2, b2,
            dw1, db1, dw2, db2,
            z1, a1, z2,
            dz2, da1, dz1, dx,
            batch_size, input_dim, hidden_dim, output_dim,
        }
    }

    // Initialize weights randomly, biases to zero, and upload to the GPU
    pub fn initialize_parameters(&self) {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        // Kaiming Normal initialization: standard deviation = sqrt(2 / input_dimension)
        let std_w1 = (2.0 / self.input_dim as f32).sqrt();
        let mut host_w1 = vec![0.0; self.input_dim * self.hidden_dim];
        for val in host_w1.iter_mut() {
            *val = rng.gen_range(-1.0..1.0) * std_w1;
        }
        let std_w2 = (2.0 / self.hidden_dim as f32).sqrt();
        let mut host_w2 = vec![0.0; self.hidden_dim * self.output_dim];
        for val in host_w2.iter_mut() {
            *val = rng.gen_range(-1.0..1.0) * std_w2;
        }
        let host_b1 = vec![0.0; self.hidden_dim];
        let host_b2 = vec![0.0; self.output_dim];
        // Upload our initialized values to GPU VRAM
        self.w1.copy_to_device(&host_w1);
        self.b1.copy_to_device(&host_b1);
        self.w2.copy_to_device(&host_w2);
        self.b2.copy_to_device(&host_b2);
    }
     // Forward Pass on GPU
    pub fn forward(&self, gpu_input: &GPUMemory) {
        unsafe {
            // Layer 1: Z1 = X * W1 + B1
            launch_matmul_forward(
                self.z1.ptr,
                gpu_input.ptr,
                self.w1.ptr,
                self.b1.ptr,
                self.batch_size as i32,
                self.input_dim as i32,
                self.hidden_dim as i32,
            );
            // Activation 1: A1 = max(0, Z1)
            launch_relu_forward(
                self.a1.ptr,
                self.z1.ptr,
                (self.batch_size * self.hidden_dim) as i32,
            );
            // Layer 2: Z2 = A1 * W2 + B2 (logits)
            launch_matmul_forward(
                self.z2.ptr,
                self.a1.ptr,
                self.w2.ptr,
                self.b2.ptr,
                self.batch_size as i32,
                self.hidden_dim as i32,
                self.output_dim as i32,
            );
        }
    }

    // Backward Pass on GPU
    pub fn backward(&self, gpu_input: &GPUMemory) {
        unsafe {
            // 1. Backprop through Layer 2:
            // dW2 = A1^T * dZ2
            launch_matmul_backward_weight(
                self.dw2.ptr,
                self.a1.ptr,
                self.dz2.ptr,
                self.batch_size as i32,
                self.hidden_dim as i32,
                self.output_dim as i32,
            );
            // dB2 = sum(dZ2, axis=0)
            launch_matmul_backward_bias(
                self.db2.ptr,
                self.dz2.ptr,
                self.batch_size as i32,
                self.output_dim as i32,
            );
            // dA1 = dZ2 * W2^T
            launch_matmul_backward_input(
                self.da1.ptr,
                self.dz2.ptr,
                self.w2.ptr,
                self.batch_size as i32,
                self.hidden_dim as i32,
                self.output_dim as i32,
            );
            // 2. Backprop through ReLU Activation:
            // dZ1 = dA1 * [Z1 > 0]
            launch_relu_backward(
                self.dz1.ptr,
                self.da1.ptr,
                self.z1.ptr,
                (self.batch_size * self.hidden_dim) as i32,
            );
            // 3. Backprop through Layer 1:
            // dW1 = X^T * dZ1
            launch_matmul_backward_weight(
                self.dw1.ptr,
                gpu_input.ptr,
                self.dz1.ptr,
                self.batch_size as i32,
                self.input_dim as i32,
                self.hidden_dim as i32,
            );
            // dB1 = sum(dZ1, axis=0)
            launch_matmul_backward_bias(
                self.db1.ptr,
                self.dz1.ptr,
                self.batch_size as i32,
                self.hidden_dim as i32,
            );
            // dX = dZ1 * W1^T (optional for updating parameters, but good)
            launch_matmul_backward_input(
                self.dx.ptr,
                self.dz1.ptr,
                self.w1.ptr,
                self.batch_size as i32,
                self.input_dim as i32,
                self.hidden_dim as i32,
            );
        }
    }
     // Apply Gradient Descent weights/biases updates on the GPU
    pub fn update_parameters(&self, learning_rate: f32) {
        unsafe {
            // W1 = W1 - learning_rate * dW1
            launch_sgd_step(
                self.w1.ptr,
                self.dw1.ptr,
                learning_rate,
                (self.input_dim * self.hidden_dim) as i32,
            );
            // B1 = B1 - learning_rate * dB1
            launch_sgd_step(
                self.b1.ptr,
                self.db1.ptr,
                learning_rate,
                self.hidden_dim as i32,
            );
            // W2 = W2 - learning_rate * dW2
            launch_sgd_step(
                self.w2.ptr,
                self.dw2.ptr,
                learning_rate,
                (self.hidden_dim * self.output_dim) as i32,
            );
            // B2 = B2 - learning_rate * dB2
            launch_sgd_step(
                self.b2.ptr,
                self.db2.ptr,
                learning_rate,
                self.output_dim as i32,
            );
        }
    }
}