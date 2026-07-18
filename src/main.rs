pub mod math_train;
pub mod mlp;

use math_train::GPUMemory;
use crate::mlp::{MLP, AdamW};

fn main() {
    println!("Initializing heavyengine Training Pipeline...");

    // 1. Define the XOR dataset (Batch Size = 4, Input Dim = 2)
    let xor_inputs = vec![
        0.0, 0.0, // Class 0
        0.0, 1.0, // Class 1
        1.0, 0.0, // Class 1
        1.0, 1.0, // Class 0
    ];
    let xor_targets = vec![0, 1, 1, 0]; // Index of target class for each sample

    // 2. Initialize the MLP model
    let batch_size = 4;
    let input_dim = 2;
    let hidden_dim = 8;
    let output_dim = 2;

    let model = MLP::new(batch_size, input_dim, hidden_dim, output_dim);
    model.initialize_parameters();
    
    // Initialize AdamW optimizer state (Wiping memory to 0.0)
    let mut optimizer = AdamW::new(input_dim, hidden_dim, output_dim);

    // 3. Upload XOR inputs to the GPU (we keep them there since they don't change)
    let gpu_inputs = GPUMemory::new(xor_inputs.len());
    gpu_inputs.copy_to_device(&xor_inputs);

    // Hyperparameters
    let epochs = 2000;
    let learning_rate = 0.01;

    println!("Starting training for {} epochs...", epochs);

    for epoch in 1..=epochs {
        // Step A: Forward Pass on GPU
        model.forward(&gpu_inputs);

        // Step B: Download logits to CPU to calculate Loss and dZ2 (starting gradient)
        let mut logits = vec![0.0; batch_size * output_dim];
        model.z2.copy_to_host(&mut logits);

        let mut epoch_loss = 0.0;
        let mut host_dz2 = vec![0.0; batch_size * output_dim];

        for b in 0..batch_size {
            let offset = b * output_dim;
            let logit0 = logits[offset];
            let logit1 = logits[offset + 1];

            // Stable Softmax calculation on CPU
            let max_logit = logit0.max(logit1);
            let exp0 = (logit0 - max_logit).exp();
            let exp1 = (logit1 - max_logit).exp();
            let sum_exps = exp0 + exp1;

            let prob0 = exp0 / sum_exps;
            let prob1 = exp1 / sum_exps;

            // Calculate Cross-Entropy Loss: -ln(prob_of_correct_class)
            let target_class = xor_targets[b];
            let prob_correct = if target_class == 0 { prob0 } else { prob1 };
            epoch_loss += -prob_correct.ln();

            // Calculate starting gradient: dZ2 = P - Y
            host_dz2[offset] = prob0 - (if target_class == 0 { 1.0 } else { 0.0 });
            host_dz2[offset + 1] = prob1 - (if target_class == 1 { 1.0 } else { 0.0 });
        }

        epoch_loss /= batch_size as f32; // Average loss for the batch

        // Upload computed gradients (dZ2) back to the GPU
        model.dz2.copy_to_device(&host_dz2);

        // Step C: Backward Pass on GPU (computes dW2, dB2, dA1, dZ1, dW1, dB1)
        model.backward(&gpu_inputs);

        // Step D: Update model weights and biases using AdamW
        optimizer.step(&model, learning_rate);


        // Print loss every 200 epochs
        if epoch % 200 == 0 || epoch == 1 {
            println!("Epoch {:4} | Loss: {:.6}", epoch, epoch_loss);
        }
    }

    println!("\nTraining complete! Testing final predictions:\n");

    // Run a final forward pass and print predictions
    model.forward(&gpu_inputs);
    let mut final_logits = vec![0.0; batch_size * output_dim];
    model.z2.copy_to_host(&mut final_logits);

    for b in 0..batch_size {
        let offset = b * output_dim;
        let logit0 = final_logits[offset];
        let logit1 = final_logits[offset + 1];

        // Softmax
        let max_logit = logit0.max(logit1);
        let exp0 = (logit0 - max_logit).exp();
        let exp1 = (logit1 - max_logit).exp();
        let sum = exp0 + exp1;
        let prob0 = exp0 / sum;
        let prob1 = exp1 / sum;

        let input_x1 = xor_inputs[b * 2];
        let input_x2 = xor_inputs[b * 2 + 1];

        println!(
            "Input: [{}, {}] | Target: {} | Predicted Probability: [Class 0: {:.4}, Class 1: {:.4}]",
            input_x1, input_x2, xor_targets[b], prob0, prob1
        );
    }
}