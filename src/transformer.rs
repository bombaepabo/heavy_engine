use crate::math_train::GPUMemory;
use rand::Rng;
use crate::attention::SelfAttention; // Add this to the top of transformer.rs
use crate::mlp::MLP;

pub struct Transformer {
    // 1. The Dictionaries
    pub token_embeddings: GPUMemory,
    pub position_embeddings: GPUMemory, // NEW! Gives the AI a sense of physical location
    
    // The Attention Mechanism (The "Thinking" part)
    pub attention: SelfAttention, // <--- ADDED LINE!

    pub mlp: MLP,


    // 2. The Final Output Layer (Guessing the next letter)
    pub w_out: GPUMemory, 
    pub b_out: GPUMemory, 
    pub dw_out: GPUMemory,
    pub db_out: GPUMemory,
    
    // Dimensions
    pub vocab_size: usize,
    pub embed_dim: usize,
    pub sequence_length: usize,
}

impl Transformer {
    pub fn new(batch_size: usize, vocab_size: usize, embed_dim: usize, sequence_length: usize) -> Self {
        Self {
            token_embeddings: GPUMemory::new(vocab_size * embed_dim),
            
            // The Position table has a row for every spot in the sequence! (e.g. 0 to 25)
            position_embeddings: GPUMemory::new(sequence_length * embed_dim),
            
            attention: SelfAttention::new(embed_dim),
            
            // The MLP expands the "meaning" 4-times wider so it has room to think!
            mlp: MLP::new(batch_size * sequence_length, embed_dim, embed_dim * 4, embed_dim),

            w_out: GPUMemory::new(embed_dim * vocab_size),
            b_out: GPUMemory::new(vocab_size),
            
            dw_out: GPUMemory::new(embed_dim * vocab_size),
            db_out: GPUMemory::new(vocab_size),

            vocab_size,
            embed_dim,
            sequence_length,
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

        // Fill the tables with random math!
        init_matrix(&self.token_embeddings, self.vocab_size * self.embed_dim);
        init_matrix(&self.position_embeddings, self.sequence_length * self.embed_dim);
        init_matrix(&self.w_out, self.embed_dim * self.vocab_size);
        
        // WE MUST INITIALIZE ATTENTION OR WSL2 PAGE FAULTS ON LAZY ALLOCATION!
        self.attention.initialize_parameters();
        self.mlp.initialize_parameters();
        
        let zero_bout = vec![0.0; self.vocab_size];
        self.b_out.copy_to_device(&zero_bout);
    }
}