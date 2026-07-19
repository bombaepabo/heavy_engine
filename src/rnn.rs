use crate::math_train::GPUMemory;
use rand::Rng;

pub struct RNN {
    pub embeddings: GPUMemory,
    
    // Weights
    pub w_xh: GPUMemory,
    pub w_hh: GPUMemory,
    pub b_h:  GPUMemory,
    pub w_hy: GPUMemory, 
    pub b_y:  GPUMemory, 
    
    // Gradients (NEW FOR BPTT! This holds the Blame Scores)
    pub dw_xh: GPUMemory,
    pub dw_hh: GPUMemory,
    pub db_h:  GPUMemory,
    pub dw_hy: GPUMemory,
    pub db_y:  GPUMemory,
    
    // The Hidden State History (Stores 25 memory snapshots across time!)
    pub h: GPUMemory,
    
    pub vocab_size: usize,
    pub embed_dim: usize,
    pub hidden_dim: usize,
    pub sequence_length: usize,
}

impl RNN {
    pub fn new(batch_size: usize, vocab_size: usize, embed_dim: usize, hidden_dim: usize, sequence_length: usize) -> Self {
        Self {
            embeddings: GPUMemory::new(vocab_size * embed_dim),
            
            w_xh: GPUMemory::new(embed_dim * hidden_dim),
            w_hh: GPUMemory::new(hidden_dim * hidden_dim), 
            b_h:  GPUMemory::new(hidden_dim),
            w_hy: GPUMemory::new(hidden_dim * vocab_size),
            b_y:  GPUMemory::new(vocab_size),
            
            // Allocate blank GPU memory for the Blame Scores!
            dw_xh: GPUMemory::new(embed_dim * hidden_dim),
            dw_hh: GPUMemory::new(hidden_dim * hidden_dim),
            db_h:  GPUMemory::new(hidden_dim),
            dw_hy: GPUMemory::new(hidden_dim * vocab_size),
            db_y:  GPUMemory::new(vocab_size),
            
            // THE TIME MACHINE: Multiplies memory size by sequence_length!
            h: GPUMemory::new(batch_size * sequence_length * hidden_dim),
            
            vocab_size,
            embed_dim,
            hidden_dim,
            sequence_length,
        }
    }
    
    pub fn initialize_parameters(&self) {
        let mut rng = rand::thread_rng();
        
        // Helper function to fill a GPU memory buffer with random tiny numbers
        let mut init_matrix = |memory: &GPUMemory, size: usize| {
            let mut host_data = vec![0.0; size];
            for val in host_data.iter_mut() {
                *val = rng.gen_range(-0.1..0.1);
            }
            memory.copy_to_device(&host_data);
        };

        init_matrix(&self.embeddings, self.vocab_size * self.embed_dim);
        init_matrix(&self.w_xh, self.embed_dim * self.hidden_dim);
        init_matrix(&self.w_hh, self.hidden_dim * self.hidden_dim);
        init_matrix(&self.w_hy, self.hidden_dim * self.vocab_size);
        
        // Biases start at exactly 0.0
        let zero_bh = vec![0.0; self.hidden_dim];
        self.b_h.copy_to_device(&zero_bh);
        
        let zero_by = vec![0.0; self.vocab_size];
        self.b_y.copy_to_device(&zero_by);
    }
      // The Forward Pass for a single letter in time!
    pub fn forward_step(&self, x_embed: &GPUMemory, h_prev: &GPUMemory, h_next: &GPUMemory, logits: &GPUMemory, batch_size: usize) {
        
        // 1. Calculate the NEW Memory (Combines the new letter and the old memory)
        unsafe {
            crate::math_train::rnn_forward_kernel(
                x_embed.ptr,
                h_prev.ptr,
                self.w_xh.ptr,
                self.w_hh.ptr,
                self.b_h.ptr,
                h_next.ptr,
                batch_size as i32,
                self.embed_dim as i32,
                self.hidden_dim as i32,
            );
        }
        // 2. Guess the next letter from the new memory! 
        unsafe {
            crate::math_train::launch_matmul_forward(
                logits.ptr,
                h_next.ptr,
                self.w_hy.ptr,
                self.b_y.ptr,
                batch_size as i32,
                self.hidden_dim as i32,
                self.vocab_size as i32
            );
        }
    }
}