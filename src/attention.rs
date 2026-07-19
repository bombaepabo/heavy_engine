use crate::math_train::GPUMemory;
use rand::Rng;

pub struct SelfAttention {

    pub _pool: GPUMemory,
    // The famous Q, K, V matrices!
    pub w_q: GPUMemory, // Query
    pub w_k: GPUMemory, // Key
    pub w_v: GPUMemory, // Value
    
    // The final projection after they finish talking
    pub w_o: GPUMemory, 

    pub dw_q: GPUMemory,
    pub dw_k: GPUMemory,
    pub dw_v: GPUMemory,
    pub dw_o: GPUMemory,
    
    pub embed_dim: usize,
}

impl SelfAttention {
    pub fn new(embed_dim: usize) -> Self {
        // In a Transformer, the Q,K,V matrices are all perfect squares
        let size = embed_dim * embed_dim;
        
        let pool = GPUMemory::new(size * 8); 

        Self {
            w_q: pool.view(size * 0, size),
            w_k: pool.view(size * 1, size),
            w_v: pool.view(size * 2, size),
            w_o: pool.view(size * 3, size),

            dw_q: pool.view(size * 4, size),
            dw_k: pool.view(size * 5, size),
            dw_v: pool.view(size * 6, size),
            dw_o: pool.view(size * 7, size),
            _pool: pool,
            embed_dim,
        }
    }

    pub fn initialize_parameters(&self) {
        let mut rng = rand::thread_rng();
        let mut init_matrix = |memory: &GPUMemory, size: usize| {
            let mut host_data = vec![0.0; size];
            for val in host_data.iter_mut() {
                *val = rng.gen_range(-0.1..0.1);
            }
            memory.copy_to_device(&host_data);
        };

        let size = self.embed_dim * self.embed_dim;
        init_matrix(&self.w_q, size);
        init_matrix(&self.w_k, size);
        init_matrix(&self.w_v, size);
        init_matrix(&self.w_o, size);
    }
}