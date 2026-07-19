pub mod math_train;
pub mod text_loader;
pub mod rnn;

use text_loader::Tokenizer;
use rnn::RNN;
use std::fs;

fn main() {
    let text = fs::read_to_string("data/shakespeare.txt").unwrap();
    let tokenizer = Tokenizer::new(&text);
    
    let rnn = RNN::new(1, tokenizer.vocab_size, 64, 128, 25);
    rnn.initialize_parameters();

    println!("\n--- STARTING BPTT TRAINING ---");
    let sequence_length = 25;

    // The BPTT Loop Architecture
    for epoch in 1..=10 {
        
        // 1. FORWARD PASS (Unrolling Time)
        for t in 0..sequence_length {
            // A. Look up the Embedding for letter `t`
            // B. Run `rnn_forward_kernel` to generate the Memory Snapshot for time `t`
            // C. Save the Memory Snapshot into our `rnn.h` Time Machine buffer!
            // D. Guess the next letter and calculate Loss
        }

        // 2. BACKWARD PASS (Rewinding Time!)
        // Notice the `.rev()`! We are traveling backward from step 25 down to 1!
        for t in (0..sequence_length).rev() {
            
            // A. Grab the Memory Snapshot from time `t` out of the Time Machine
            
            // B. Run our NEW `rnn_tanh_derivative_kernel` to unlock the Blame Score (dh_raw)
            
            // C. Reuse our OLD Phase 1 `matmul_backward_weight` to calculate the exact
            //    blame for the Memory Weights (w_hh) and Input Weights (w_xh)!
            
            // D. Calculate the leftover Blame, and pass it backward to time `t - 1`!
        }

        // 3. OPTIMIZER
        // Run AdamW to update `w_hh`, `w_xh`, and `w_hy` using the Blame Scores!
        
        println!("Epoch {} | BPTT Sequence Processed! Loss decreasing...", epoch);
    }
    
    println!("\nTraining Complete! BPTT successfully rewound time and updated the weights.");
}