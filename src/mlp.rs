use crate:: math_train::*;

// ==========================================
// AdamW Optimizer State
// ==========================================
pub struct AdamW {
    pub m_w1: GPUMemory, pub v_w1: GPUMemory,
    pub m_b1: GPUMemory, pub v_b1: GPUMemory,
    pub m_gamma: GPUMemory, pub v_gamma: GPUMemory,
    pub m_beta: GPUMemory, pub v_beta: GPUMemory,
    pub m_w2: GPUMemory, pub v_w2: GPUMemory,
    pub m_b2: GPUMemory, pub v_b2: GPUMemory,
    pub step: i32,
}
impl AdamW {
    pub fn new(input_dim: usize, hidden_dim: usize, output_dim: usize) -> Self {
        let optimizer = Self {
            m_w1: GPUMemory::new(input_dim * hidden_dim), v_w1: GPUMemory::new(input_dim * hidden_dim),
            m_b1: GPUMemory::new(hidden_dim), v_b1: GPUMemory::new(hidden_dim),
            
            // LayerNorm dimensions
            m_gamma: GPUMemory::new(hidden_dim), v_gamma: GPUMemory::new(hidden_dim),
            m_beta: GPUMemory::new(hidden_dim), v_beta: GPUMemory::new(hidden_dim),
            
            m_w2: GPUMemory::new(hidden_dim * output_dim), v_w2: GPUMemory::new(hidden_dim * output_dim),
            m_b2: GPUMemory::new(output_dim), v_b2: GPUMemory::new(output_dim),
            step: 0,
        };
        
        // Wipe all history to 0.0 before we start training!
        optimizer.m_w1.zero_memory(); optimizer.v_w1.zero_memory();
        optimizer.m_b1.zero_memory(); optimizer.v_b1.zero_memory();
        optimizer.m_gamma.zero_memory(); optimizer.v_gamma.zero_memory();
        optimizer.m_beta.zero_memory(); optimizer.v_beta.zero_memory();
        optimizer.m_w2.zero_memory(); optimizer.v_w2.zero_memory();
        optimizer.m_b2.zero_memory(); optimizer.v_b2.zero_memory();
        
        optimizer
    }
    
    // This entirely replaces the old `update_parameters` function!
    pub fn step(&mut self, model: &MLP, lr: f32) {
        self.step += 1;
        let beta1 = 0.9;      // How much momentum to keep
        let beta2 = 0.999;    // How much velocity to keep
        let eps = 1e-8;       // Prevent dividing by zero
        let weight_decay = 0.01; // Friction
        unsafe {
            launch_adamw_step(model.w1.ptr, self.m_w1.ptr, self.v_w1.ptr, model.dw1.ptr, lr, beta1, beta2, eps, weight_decay, self.step, (model.input_dim * model.hidden_dim) as i32);
            launch_adamw_step(model.b1.ptr, self.m_b1.ptr, self.v_b1.ptr, model.db1.ptr, lr, beta1, beta2, eps, weight_decay, self.step, model.hidden_dim as i32);
            
            // LayerNorm variables
            launch_adamw_step(model.gamma.ptr, self.m_gamma.ptr, self.v_gamma.ptr, model.dgamma.ptr, lr, beta1, beta2, eps, weight_decay, self.step, model.hidden_dim as i32);
            launch_adamw_step(model.beta.ptr, self.m_beta.ptr, self.v_beta.ptr, model.dbeta.ptr, lr, beta1, beta2, eps, weight_decay, self.step, model.hidden_dim as i32);
            
            launch_adamw_step(model.w2.ptr, self.m_w2.ptr, self.v_w2.ptr, model.dw2.ptr, lr, beta1, beta2, eps, weight_decay, self.step, (model.hidden_dim * model.output_dim) as i32);
            launch_adamw_step(model.b2.ptr, self.m_b2.ptr, self.v_b2.ptr, model.db2.ptr, lr, beta1, beta2, eps, weight_decay, self.step, model.output_dim as i32);
        }
    }
}

pub struct MLP {

    pub _pool: GPUMemory,

    // Phase 1 Parameters
    pub w1: GPUMemory, pub b1: GPUMemory,
    pub w2: GPUMemory, pub b2: GPUMemory,
    
    // Phase 2: LayerNorm Parameters
    pub gamma: GPUMemory, pub beta: GPUMemory,
    
    // Gradients
    pub dw1: GPUMemory, pub db1: GPUMemory,
    pub dw2: GPUMemory, pub db2: GPUMemory,
    pub dgamma: GPUMemory, pub dbeta: GPUMemory,
    
    // Intermediate activations
    pub z1: GPUMemory,
    pub ln_out: GPUMemory, // Phase 2: Output of LayerNorm
    pub a1: GPUMemory,
    pub z2: GPUMemory,
    
    // Intermediate gradients & caches
    pub dz2: GPUMemory,
    pub da1: GPUMemory,
    pub dln_out: GPUMemory, // Phase 2: Gradient for LayerNorm output
    pub dz1: GPUMemory,
    pub dx: GPUMemory,
    
    // Phase 2: LayerNorm Caches
    pub cache_mean: GPUMemory,
    pub cache_var: GPUMemory,
    
    // Dimensions
    pub batch_size: usize, pub input_dim: usize, pub hidden_dim: usize, pub output_dim: usize,
}
impl MLP {
    pub fn new(batch_size: usize, input_dim: usize, hidden_dim: usize, output_dim: usize) -> Self {
        // Calculate the exact size of all 23 matrices combined
        let s_w1 = input_dim * hidden_dim;   let s_b1 = hidden_dim;
        let s_w2 = hidden_dim * output_dim;  let s_b2 = output_dim;
        let s_gamma = input_dim;             let s_beta = input_dim;
        let s_z1 = batch_size * hidden_dim;  let s_ln_out = batch_size * input_dim;
        let s_a1 = batch_size * hidden_dim;  let s_z2 = batch_size * output_dim;
        let s_dz2 = batch_size * output_dim; let s_da1 = batch_size * hidden_dim;
        let s_dln_out = batch_size * input_dim; let s_dz1 = batch_size * hidden_dim;
        let s_dx = batch_size * input_dim;   let s_cache_mean = batch_size; let s_cache_var = batch_size;
        let total_size = (s_w1 * 2) + (s_b1 * 2) + (s_w2 * 2) + (s_b2 * 2) + (s_gamma * 2) + (s_beta * 2) 
            + s_z1 + s_ln_out + s_a1 + s_z2 + s_dz2 + s_da1 + s_dln_out + s_dz1 + s_dx + s_cache_mean + s_cache_var;
        
        // Allocate exactly ONE block of memory!
        let pool = GPUMemory::new(total_size);
        let mut offset = 0;
        // A neat trick to dole out chunks of memory sequentially
        let mut get = |size: usize| {
            let view = pool.view(offset, size);
            offset += size;
            view
        };
        Self {
            w1: get(s_w1), b1: get(s_b1),
            w2: get(s_w2), b2: get(s_b2),
            gamma: get(s_gamma), beta: get(s_beta),
            
            dw1: get(s_w1), db1: get(s_b1),
            dw2: get(s_w2), db2: get(s_b2),
            dgamma: get(s_gamma), dbeta: get(s_beta),
            
            z1: get(s_z1), ln_out: get(s_ln_out), a1: get(s_a1), z2: get(s_z2),
            dz2: get(s_dz2), da1: get(s_da1), dln_out: get(s_dln_out),
            dz1: get(s_dz1), dx: get(s_dx),
            
            cache_mean: get(s_cache_mean), cache_var: get(s_cache_var),
            
            batch_size, input_dim, hidden_dim, output_dim,
            _pool: pool,
        }
    }
    pub fn initialize_parameters(&self) {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        let std_w1 = (2.0 / self.input_dim as f32).sqrt();
        let mut host_w1 = vec![0.0; self.input_dim * self.hidden_dim];
        for val in host_w1.iter_mut() { *val = rng.gen_range(-1.0..1.0) * std_w1; }
        
        let std_w2 = (2.0 / self.hidden_dim as f32).sqrt();
        let mut host_w2 = vec![0.0; self.hidden_dim * self.output_dim];
        for val in host_w2.iter_mut() { *val = rng.gen_range(-1.0..1.0) * std_w2; }
        
        let host_b1 = vec![0.0; self.hidden_dim];
        let host_b2 = vec![0.0; self.output_dim];
        
        // Phase 2: Gamma starts at 1.0 (no scaling), Beta starts at 0.0 (no shift)
        let host_gamma = vec![1.0; self.input_dim];
        let host_beta = vec![0.0; self.input_dim];
        
        self.w1.copy_to_device(&host_w1);
        self.b1.copy_to_device(&host_b1);
        self.w2.copy_to_device(&host_w2);
        self.b2.copy_to_device(&host_b2);
        self.gamma.copy_to_device(&host_gamma);
        self.beta.copy_to_device(&host_beta);
    }
    // Forward Pass on GPU
    pub fn forward(&self, gpu_input: &GPUMemory) {
        unsafe {
            // Phase 2: LayerNorm (Normalizes input, applies Gamma/Beta, saves to LN_OUT)
            launch_layernorm_forward(self.ln_out.ptr, self.cache_mean.ptr, self.cache_var.ptr, gpu_input.ptr, self.gamma.ptr, self.beta.ptr, self.batch_size as i32, self.input_dim as i32, 1e-5);
            
            // Layer 1: Z1 = LN_OUT * W1 + B1
            launch_matmul_forward(self.z1.ptr, self.ln_out.ptr, self.w1.ptr, self.b1.ptr, self.batch_size as i32, self.input_dim as i32, self.hidden_dim as i32);
            
            // Activation 1: A1 = max(0, Z1)
            launch_relu_forward(self.a1.ptr, self.z1.ptr, (self.batch_size * self.hidden_dim) as i32);
            
            // Layer 2: Z2 = A1 * W2 + B2 (logits)
            launch_matmul_forward(self.z2.ptr, self.a1.ptr, self.w2.ptr, self.b2.ptr, self.batch_size as i32, self.hidden_dim as i32, self.output_dim as i32);
        }
    }
    // Backward Pass on GPU
    pub fn backward(&self, gpu_input: &GPUMemory) {
        unsafe {
            // 1. Backprop through Layer 2
            launch_matmul_backward_weight(self.dw2.ptr, self.a1.ptr, self.dz2.ptr, self.batch_size as i32, self.hidden_dim as i32, self.output_dim as i32);
            launch_matmul_backward_bias(self.db2.ptr, self.dz2.ptr, self.batch_size as i32, self.output_dim as i32);
            launch_matmul_backward_input(self.da1.ptr, self.dz2.ptr, self.w2.ptr, self.batch_size as i32, self.hidden_dim as i32, self.output_dim as i32);
            
            // 2. Backprop through ReLU
            launch_relu_backward(self.dz1.ptr, self.da1.ptr, self.z1.ptr, (self.batch_size * self.hidden_dim) as i32);
            
            // 3. Backprop through Layer 1
            launch_matmul_backward_weight(self.dw1.ptr, self.ln_out.ptr, self.dz1.ptr, self.batch_size as i32, self.input_dim as i32, self.hidden_dim as i32);
            launch_matmul_backward_bias(self.db1.ptr, self.dz1.ptr, self.batch_size as i32, self.hidden_dim as i32);
            launch_matmul_backward_input(self.dln_out.ptr, self.dz1.ptr, self.w1.ptr, self.batch_size as i32, self.input_dim as i32, self.hidden_dim as i32);
            
            // Phase 2: Wipe Gamma/Beta blame scores to 0.0 before atomicAdd!
            self.dgamma.zero_memory();
            self.dbeta.zero_memory();
            
            // Phase 2: Backprop through LayerNorm
            launch_layernorm_backward(self.dx.ptr, self.dgamma.ptr, self.dbeta.ptr, self.dln_out.ptr, gpu_input.ptr, self.cache_mean.ptr, self.cache_var.ptr, self.gamma.ptr, self.batch_size as i32, self.input_dim as i32, 1e-5);
        }
    }
}
   