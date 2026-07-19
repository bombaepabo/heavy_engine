use std::ffi::c_void;

// ==========================================
// 1. CPU REFERENCE IMPLEMENTATION
// ==========================================

pub fn cpu_matmul_forward(
    out: &mut [f32],
    x: &[f32],
    w: &[f32],
    b: &[f32],
    batch: usize,
    in_dim: usize,
    out_dim: usize,
) {
    for r in 0..batch {
        for c in 0..out_dim {
            let mut sum = b[c];
            for i in 0..in_dim {
                sum += x[r * in_dim + i] * w[i * out_dim + c];
            }
            out[r * out_dim + c] = sum;
        }
    }
}

pub fn cpu_matmul_backward(
    dx: &mut [f32],
    dw: &mut [f32],
    db: &mut [f32],
    dz: &[f32],
    x: &[f32],
    w: &[f32],
    batch: usize,
    in_dim: usize,
    out_dim: usize,
) {
    // dW = X^T * dZ
    for i in 0..in_dim {
        for o in 0..out_dim {
            let mut sum = 0.0;
            for b in 0..batch {
                sum += x[b * in_dim + i] * dz[b * out_dim + o];
            }
            dw[i * out_dim + o] = sum;
        }
    }

    // dB = sum(dZ, axis=0)
    for o in 0..out_dim {
        let mut sum = 0.0;
        for b in 0..batch {
            sum += dz[b * out_dim + o];
        }
        db[o] = sum;
    }

    // dX = dZ * W^T
    for b in 0..batch {
        for i in 0..in_dim {
            let mut sum = 0.0;
            for o in 0..out_dim {
                sum += dz[b * out_dim + o] * w[i * out_dim + o];
            }
            dx[b * in_dim + i] = sum;
        }
    }
}

// ==========================================
// 2. LOW-LEVEL CUDA FFI BINDINGS
// ==========================================

unsafe extern "C"{
    // GPU memory management helpers
    pub fn gpu_alloc(size: usize) -> *mut c_void;
    pub fn gpu_free(device_ptr: *mut c_void);
    pub fn gpu_copy_to_device(device_dest: *mut c_void, host_src: *const c_void, size: usize);
    pub fn gpu_copy_to_host(host_dest: *mut c_void, device_src: *const c_void, size: usize);

    // CUDA kernel launcher wrappers
    pub fn launch_matmul_forward(
        out: *mut f32,
        x: *const f32,
        w: *const f32,
        b: *const f32,
        batch: i32,
        in_dim: i32,
        out_dim: i32,
    );

    pub fn launch_matmul_backward_weight(
        dw: *mut f32,
        x: *const f32,
        dz: *const f32,
        batch: i32,
        in_dim: i32,
        out_dim: i32,
    );

    pub fn launch_matmul_backward_bias(
        db: *mut f32,
        dz: *const f32,
        batch: i32,
        out_dim: i32,
    );

    pub fn launch_matmul_backward_input(
        dx: *mut f32,
        dz: *const f32,
        w: *const f32,
        batch: i32,
        in_dim: i32,
        out_dim: i32,
    );

    pub fn launch_relu_forward(out: *mut f32, inp: *const f32, size: i32);
    pub fn launch_relu_backward(dinp: *mut f32, dout: *const f32, inp: *const f32, size: i32);
    pub fn launch_sgd_step(param: *mut f32, grad: *const f32, lr: f32, size: i32);

 // Phase 2: LayerNorm & AdamW Wrappers
    pub fn launch_layernorm_forward(
        out: *mut f32, cache_mean: *mut f32, cache_var: *mut f32,
        inp: *const f32, gamma: *const f32, beta: *const f32,
        batch: i32, dim: i32, eps: f32,
    );
    pub fn launch_layernorm_backward(
        dinp: *mut f32, dgamma: *mut f32, dbeta: *mut f32,
        dout: *const f32, inp: *const f32, cache_mean: *const f32, cache_var: *const f32,
        gamma: *const f32, batch: i32, dim: i32, eps: f32,
    );
    pub fn launch_adamw_step(
        param: *mut f32, m: *mut f32, v: *mut f32, grad: *const f32,
        lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32,
        step: i32, size: i32,
    );
    
    // Phase 2: Helper to zero out memory arrays on the GPU
    pub fn gpu_memset(device_ptr: *mut c_void, value: i32, size: usize);

    pub fn rnn_forward_kernel(
        x_embed: *const f32,
        h_prev: *const f32,
        w_xh: *const f32,
        w_hh: *const f32,
        b_h: *const f32,
        h_next: *mut f32,
        batch_size: std::os::raw::c_int,
        embed_dim: std::os::raw::c_int,
        hidden_dim: std::os::raw::c_int,
    );

    pub fn rnn_tanh_derivative_kernel(
        dh: *const f32,
        h_t: *const f32,
        dh_raw: *mut f32,
        batch_size: std::os::raw::c_int,
        hidden_dim: std::os::raw::c_int,
    );
}

// ==========================================
// 3. SAFE RUST GPUMEMORY WRAPPER (RAII)
// ==========================================

pub struct GPUMemory {
    pub ptr: *mut f32, // Pointer to memory in GPU VRAM
    pub size: usize,   // Number of float elements allocated
}

impl GPUMemory {
    // Allocate memory on the GPU
    pub fn new(size: usize) -> Self {
        let byte_size = size * std::mem::size_of::<f32>();
        let ptr = unsafe { gpu_alloc(byte_size) as *mut f32 };
        if ptr.is_null() {
            panic!("Failed to allocate {} bytes on GPU VRAM", byte_size);
        }
        Self { ptr, size }
    }

    // Upload data from CPU (Host) to GPU (Device)
    pub fn copy_to_device(&self, host_slice: &[f32]) {
        assert_eq!(host_slice.len(), self.size, "Size mismatch during upload");
        let byte_size = self.size * std::mem::size_of::<f32>();
        unsafe {
            gpu_copy_to_device(
                self.ptr as *mut c_void,
                host_slice.as_ptr() as *const c_void,
                byte_size,
            );
        }
    }

    // Download data from GPU (Device) to CPU (Host)
    pub fn copy_to_host(&self, host_slice: &mut [f32]) {
        assert_eq!(host_slice.len(), self.size, "Size mismatch during download");
        let byte_size = self.size * std::mem::size_of::<f32>();
        unsafe {
            gpu_copy_to_host(
                host_slice.as_mut_ptr() as *mut c_void,
                self.ptr as *const c_void,
                byte_size,
            );
        }
    }

    // Zero out the VRAM (used to wipe gradients before Backprop atomicAdd)
    pub fn zero_memory(&self) {
        let byte_size = self.size * std::mem::size_of::<f32>();
        unsafe {
            gpu_memset(self.ptr as *mut c_void, 0, byte_size);
        }
    }
}

// When a GPUMemory struct falls out of scope, free the VRAM automatically!
impl Drop for GPUMemory {
    fn drop(&mut self) {
        unsafe {
            gpu_free(self.ptr as *mut c_void);
        }
    }
}

// ==========================================
// 4. UNIT TESTS (CPU vs. GPU validation)
// ==========================================
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_cuda_matmul_forward() {
        let batch = 2;
        let in_dim = 3;
        let out_dim = 4;
        // Mock Input data X (batch size of 2, input dimension of 3)
        let x = vec![
            1.0, 2.0, 3.0, 
            4.0, 5.0, 6.0
        ];
        // Mock Weights W (input dimension 3, output dimension 4)
        let w = vec![
            0.1, 0.2, 0.3, 0.4,
            0.5, 0.6, 0.7, 0.8,
            0.9, 1.0, 1.1, 1.2,
        ];
        // Mock Biases B (output dimension 4)
        let b = vec![0.5, -0.5, 1.0, -1.0];
        // 1. Compute on CPU
        let mut cpu_out = vec![0.0; batch * out_dim];
        cpu_matmul_forward(&mut cpu_out, &x, &w, &b, batch, in_dim, out_dim);
        // 2. Compute on GPU
        let gpu_x = GPUMemory::new(x.len());
        let gpu_w = GPUMemory::new(w.len());
        let gpu_b = GPUMemory::new(b.len());
        let gpu_out = GPUMemory::new(cpu_out.len());
        gpu_x.copy_to_device(&x);
        gpu_w.copy_to_device(&w);
        gpu_b.copy_to_device(&b);
        unsafe {
            launch_matmul_forward(
                gpu_out.ptr,
                gpu_x.ptr,
                gpu_w.ptr,
                gpu_b.ptr,
                batch as i32,
                in_dim as i32,
                out_dim as i32,
            );
        }
        let mut gpu_out_data = vec![0.0; cpu_out.len()];
        gpu_out.copy_to_host(&mut gpu_out_data);
        // 3. Assert outputs are identical (within float tolerances)
        for i in 0..cpu_out.len() {
            let diff = (cpu_out[i] - gpu_out_data[i]).abs();
            assert!(
                diff < 1e-5,
                "Forward pass mismatch at index {}: CPU = {}, GPU = {}",
                i,
                cpu_out[i],
                gpu_out_data[i]
            );
        }
        println!("MatMul Forward pass matches perfectly!");
    }
    #[test]
    fn test_cuda_matmul_backward() {
        let batch = 2;
        let in_dim = 3;
        let out_dim = 4;
        // Mock inputs and weights
        let x = vec![
            1.0, 2.0, 3.0, 
            4.0, 5.0, 6.0
        ];
        let w = vec![
            0.1, 0.2, 0.3, 0.4,
            0.5, 0.6, 0.7, 0.8,
            0.9, 1.0, 1.1, 1.2,
        ];
        // Mock incoming gradients from layer above
        let dz = vec![
            0.1, -0.2, 0.3, -0.4, 
            0.5, -0.6, 0.7, -0.8
        ];
        // 1. Compute gradients on CPU
        let mut cpu_dx = vec![0.0; batch * in_dim];
        let mut cpu_dw = vec![0.0; in_dim * out_dim];
        let mut cpu_db = vec![0.0; out_dim];
        cpu_matmul_backward(
            &mut cpu_dx,
            &mut cpu_dw,
            &mut cpu_db,
            &dz,
            &x,
            &w,
            batch,
            in_dim,
            out_dim,
        );
        // 2. Compute gradients on GPU
        let gpu_x = GPUMemory::new(x.len());
        let gpu_w = GPUMemory::new(w.len());
        let gpu_dz = GPUMemory::new(dz.len());
        
        let gpu_dx = GPUMemory::new(cpu_dx.len());
        let gpu_dw = GPUMemory::new(cpu_dw.len());
        let gpu_db = GPUMemory::new(cpu_db.len());
        gpu_x.copy_to_device(&x);
        gpu_w.copy_to_device(&w);
        gpu_dz.copy_to_device(&dz);
        unsafe {
            launch_matmul_backward_input(
                gpu_dx.ptr,
                gpu_dz.ptr,
                gpu_w.ptr,
                batch as i32,
                in_dim as i32,
                out_dim as i32,
            );
            launch_matmul_backward_weight(
                gpu_dw.ptr,
                gpu_x.ptr,
                gpu_dz.ptr,
                batch as i32,
                in_dim as i32,
                out_dim as i32,
            );
            launch_matmul_backward_bias(
                gpu_db.ptr,
                gpu_dz.ptr,
                batch as i32,
                out_dim as i32,
            );
        }
        let mut gpu_dx_data = vec![0.0; cpu_dx.len()];
        let mut gpu_dw_data = vec![0.0; cpu_dw.len()];
        let mut gpu_db_data = vec![0.0; cpu_db.len()];
        gpu_dx.copy_to_host(&mut gpu_dx_data);
        gpu_dw.copy_to_host(&mut gpu_dw_data);
        gpu_db.copy_to_host(&mut gpu_db_data);
        // 3. Assert backpropagation gradients are identical
        for i in 0..cpu_dx.len() {
            assert!((cpu_dx[i] - gpu_dx_data[i]).abs() < 1e-5);
        }
        for i in 0..cpu_dw.len() {
            assert!((cpu_dw[i] - gpu_dw_data[i]).abs() < 1e-5);
        }
        for i in 0..cpu_db.len() {
            assert!((cpu_db[i] - gpu_db_data[i]).abs() < 1e-5);
        }
        println!("MatMul Backward gradients match perfectly!");
    }
}