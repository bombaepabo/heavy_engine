pub mod math_train;
pub mod text_loader;
pub mod mlp;
pub mod attention;
pub mod transformer;

use math_train::{
    GPUMemory, launch_matmul_forward, launch_matmul_backward_input,
    launch_matmul_backward_weight, launch_matmul_backward_bias,
    launch_sgd_step, launch_attention_forward, launch_attention_backward,
    launch_cross_entropy, gpu_copy_to_device, gpu_copy_to_host,
    launch_layernorm_forward, launch_layernorm_backward,
    launch_relu_forward, launch_relu_backward,launch_add_inplace
};
use text_loader::Tokenizer;
use transformer::Transformer;
use std::fs;
use std::ffi::c_void;
fn main() {
    let text = fs::read_to_string("data/shakespeare.txt").unwrap();
    let tokenizer = Tokenizer::new(&text);
    
    let gpt = Transformer::new(1, tokenizer.vocab_size, 128, 64);
    gpt.initialize_parameters();

    let seq_len = gpt.sequence_length;
    let embed_dim = gpt.embed_dim;
    let vocab_size = gpt.vocab_size;

    println!("--- STARTING REAL GPU TRANSFORMER TRAINING ---\n");

    // ==========================================
    // THE CUSTOM GPU MEMORY ARENA
    // We allocate ONE block of 20,000 floats to bypass the WSL Driver Limit!
    // ==========================================
    let gpu_pool = GPUMemory::new(100_000);
    
    let gpu_x_ptr      = unsafe { gpu_pool.ptr.add(0) };
    let gpu_q_ptr      = unsafe { gpu_pool.ptr.add(seq_len * embed_dim) };
    let gpu_k_ptr      = unsafe { gpu_pool.ptr.add(seq_len * embed_dim * 2) };
    let gpu_v_ptr      = unsafe { gpu_pool.ptr.add(seq_len * embed_dim * 3) };
    let gpu_attn_ptr   = unsafe { gpu_pool.ptr.add(seq_len * embed_dim * 4) };
    let gpu_logits_ptr = unsafe { gpu_pool.ptr.add(seq_len * embed_dim * 5) };
    
    // We keep stacking the pointers deeper into the pool!
    let offset = (seq_len * embed_dim * 5) + (seq_len * vocab_size);
    let gpu_targets_ptr = unsafe { gpu_pool.ptr.add(offset) };
    let gpu_dlogits_ptr = unsafe { gpu_pool.ptr.add(offset + seq_len) };
    let gpu_loss_ptr    = unsafe { gpu_pool.ptr.add(offset + seq_len + (seq_len * vocab_size)) };
    let gpu_dattn_ptr   = unsafe { gpu_pool.ptr.add(offset + seq_len + (seq_len * vocab_size) * 2) };

    let learning_rate = 0.002f32;

    // Convert sample text chunk to token IDs
    let sample_text = &text[0..seq_len + 1];
    let token_ids: Vec<i32> = sample_text.chars().map(|c| *tokenizer.char_to_id.get(&c).unwrap_or(&0) as i32).collect();
    let input_ids = &token_ids[0..seq_len];
    let target_ids = &token_ids[1..seq_len + 1];

    unsafe { gpu_copy_to_device(gpu_targets_ptr as *mut c_void, target_ids.as_ptr() as *const c_void, seq_len * 4); }

    for epoch in 1..=1500 {
        let zero_loss = vec![0.0f32; 1];
        unsafe { gpu_copy_to_device(gpu_loss_ptr as *mut c_void, zero_loss.as_ptr() as *const c_void, 4); }

        let mut host_tok_emb = vec![0.0f32; vocab_size * embed_dim];
        let mut host_pos_emb = vec![0.0f32; seq_len * embed_dim];
        gpt.token_embeddings.copy_to_host(&mut host_tok_emb);
        gpt.position_embeddings.copy_to_host(&mut host_pos_emb);

        let mut host_x = vec![0.0f32; seq_len * embed_dim];
        for t in 0..seq_len {
            let tok_id = input_ids[t] as usize;
            for d in 0..embed_dim {
                host_x[t * embed_dim + d] = host_tok_emb[tok_id * embed_dim + d] + host_pos_emb[t * embed_dim + d];
            }
        }
        unsafe { gpu_copy_to_device(gpu_x_ptr as *mut c_void, host_x.as_ptr() as *const c_void, host_x.len() * 4); }

        unsafe {
            launch_matmul_forward(gpu_q_ptr, gpu_x_ptr, gpt.attention.w_q.ptr, std::ptr::null(), seq_len as i32, embed_dim as i32, embed_dim as i32);
            launch_matmul_forward(gpu_k_ptr, gpu_x_ptr, gpt.attention.w_k.ptr, std::ptr::null(), seq_len as i32, embed_dim as i32, embed_dim as i32);
            launch_matmul_forward(gpu_v_ptr, gpu_x_ptr, gpt.attention.w_v.ptr, std::ptr::null(), seq_len as i32, embed_dim as i32, embed_dim as i32);

            launch_attention_forward(gpu_q_ptr, gpu_k_ptr, gpu_v_ptr, gpu_attn_ptr, seq_len as i32, embed_dim as i32);
            
            // Residual 1: Add input X to Attention Output
            launch_add_inplace(gpu_attn_ptr, gpu_x_ptr, (seq_len * embed_dim) as i32);

            launch_layernorm_forward(gpt.mlp.ln_out.ptr, gpt.mlp.cache_mean.ptr, gpt.mlp.cache_var.ptr, gpu_attn_ptr, gpt.mlp.gamma.ptr, gpt.mlp.beta.ptr, seq_len as i32, embed_dim as i32, 1e-5);
            launch_matmul_forward(gpt.mlp.z1.ptr, gpt.mlp.ln_out.ptr, gpt.mlp.w1.ptr, gpt.mlp.b1.ptr, seq_len as i32, embed_dim as i32, (embed_dim * 4) as i32);
            launch_relu_forward(gpt.mlp.a1.ptr, gpt.mlp.z1.ptr, (seq_len * embed_dim * 4) as i32);
            launch_matmul_forward(gpt.mlp.z2.ptr, gpt.mlp.a1.ptr, gpt.mlp.w2.ptr, gpt.mlp.b2.ptr, seq_len as i32, (embed_dim * 4) as i32, embed_dim as i32);

            // Residual 2: Add Attention Output (which holds X + Attn) to MLP Output
            launch_add_inplace(gpt.mlp.z2.ptr, gpu_attn_ptr, (seq_len * embed_dim) as i32);

            launch_matmul_forward(gpu_logits_ptr, gpt.mlp.z2.ptr, gpt.w_out.ptr, gpt.b_out.ptr, seq_len as i32, embed_dim as i32, vocab_size as i32);

            launch_cross_entropy(gpu_logits_ptr, gpu_targets_ptr as *const i32, gpu_dlogits_ptr, gpu_loss_ptr, seq_len as i32, vocab_size as i32);

            let zero_q = vec![0.0f32; embed_dim * embed_dim];
            gpt.attention.dw_q.copy_to_device(&zero_q);
            gpt.attention.dw_k.copy_to_device(&zero_q);
            gpt.attention.dw_v.copy_to_device(&zero_q);
            
            let zero_dw1 = vec![0.0f32; embed_dim * embed_dim * 4];
            let zero_db1 = vec![0.0f32; embed_dim * 4];
            let zero_dw2 = vec![0.0f32; embed_dim * 4 * embed_dim];
            let zero_db2 = vec![0.0f32; embed_dim];
            let zero_gamma = vec![0.0f32; embed_dim];
            let zero_beta = vec![0.0f32; embed_dim];
            gpt.mlp.dw1.copy_to_device(&zero_dw1);
            gpt.mlp.db1.copy_to_device(&zero_db1);
            gpt.mlp.dw2.copy_to_device(&zero_dw2);
            gpt.mlp.db2.copy_to_device(&zero_db2);
            gpt.mlp.dgamma.copy_to_device(&zero_gamma);
            gpt.mlp.dbeta.copy_to_device(&zero_beta);

            let zero_dwout = vec![0.0f32; embed_dim * vocab_size];
            let zero_dbout = vec![0.0f32; vocab_size];
            gpt.dw_out.copy_to_device(&zero_dwout);
            gpt.db_out.copy_to_device(&zero_dbout);

            launch_matmul_backward_weight(gpt.dw_out.ptr, gpt.mlp.z2.ptr, gpu_dlogits_ptr, seq_len as i32, embed_dim as i32, vocab_size as i32);
            launch_matmul_backward_bias(gpt.db_out.ptr, gpu_dlogits_ptr, seq_len as i32, vocab_size as i32);
            launch_matmul_backward_input(gpt.mlp.dz2.ptr, gpu_dlogits_ptr, gpt.w_out.ptr, seq_len as i32, embed_dim as i32, vocab_size as i32);
            
            launch_matmul_backward_weight(gpt.mlp.dw2.ptr, gpt.mlp.a1.ptr, gpt.mlp.dz2.ptr, seq_len as i32, (embed_dim * 4) as i32, embed_dim as i32);
            launch_matmul_backward_bias(gpt.mlp.db2.ptr, gpt.mlp.dz2.ptr, seq_len as i32, embed_dim as i32);
            launch_matmul_backward_input(gpt.mlp.da1.ptr, gpt.mlp.dz2.ptr, gpt.mlp.w2.ptr, seq_len as i32, (embed_dim * 4) as i32, embed_dim as i32);

            launch_relu_backward(gpt.mlp.dz1.ptr, gpt.mlp.da1.ptr, gpt.mlp.z1.ptr, (seq_len * embed_dim * 4) as i32);

            launch_matmul_backward_weight(gpt.mlp.dw1.ptr, gpt.mlp.ln_out.ptr, gpt.mlp.dz1.ptr, seq_len as i32, embed_dim as i32, (embed_dim * 4) as i32);
            launch_matmul_backward_bias(gpt.mlp.db1.ptr, gpt.mlp.dz1.ptr, seq_len as i32, (embed_dim * 4) as i32);
            launch_matmul_backward_input(gpt.mlp.dln_out.ptr, gpt.mlp.dz1.ptr, gpt.mlp.w1.ptr, seq_len as i32, embed_dim as i32, (embed_dim * 4) as i32);

            // LayerNorm backward writes its input gradient to dx instead of overwriting dattn directly
            launch_layernorm_backward(gpt.mlp.dx.ptr, gpt.mlp.dgamma.ptr, gpt.mlp.dbeta.ptr, gpt.mlp.dln_out.ptr, gpu_attn_ptr, gpt.mlp.cache_mean.ptr, gpt.mlp.cache_var.ptr, gpt.mlp.gamma.ptr, seq_len as i32, embed_dim as i32, 1e-5);
            
            // Re-initialize dattn to zero, then add the Residual 2 and LayerNorm backward gradients
            let zero_dattn = vec![0.0f32; seq_len * embed_dim];
            gpu_copy_to_device(gpu_dattn_ptr as *mut c_void, zero_dattn.as_ptr() as *const c_void, seq_len * embed_dim * 4);
            launch_add_inplace(gpu_dattn_ptr, gpt.mlp.dz2.ptr, (seq_len * embed_dim) as i32);
            launch_add_inplace(gpu_dattn_ptr, gpt.mlp.dx.ptr, (seq_len * embed_dim) as i32);

            launch_attention_backward(gpu_dattn_ptr, gpu_q_ptr, gpu_k_ptr, gpu_v_ptr, gpt.attention.dw_q.ptr, gpt.attention.dw_k.ptr, gpt.attention.dw_v.ptr, seq_len as i32, embed_dim as i32);

            launch_sgd_step(gpt.attention.w_q.ptr, gpt.attention.dw_q.ptr, learning_rate, (embed_dim * embed_dim) as i32);
            launch_sgd_step(gpt.attention.w_k.ptr, gpt.attention.dw_k.ptr, learning_rate, (embed_dim * embed_dim) as i32);
            launch_sgd_step(gpt.attention.w_v.ptr, gpt.attention.dw_v.ptr, learning_rate, (embed_dim * embed_dim) as i32);
            
            launch_sgd_step(gpt.mlp.w1.ptr, gpt.mlp.dw1.ptr, learning_rate, (embed_dim * embed_dim * 4) as i32);
            launch_sgd_step(gpt.mlp.b1.ptr, gpt.mlp.db1.ptr, learning_rate, (embed_dim * 4) as i32);
            launch_sgd_step(gpt.mlp.w2.ptr, gpt.mlp.dw2.ptr, learning_rate, (embed_dim * 4 * embed_dim) as i32);
            launch_sgd_step(gpt.mlp.b2.ptr, gpt.mlp.db2.ptr, learning_rate, embed_dim as i32);
            launch_sgd_step(gpt.mlp.gamma.ptr, gpt.mlp.dgamma.ptr, learning_rate, embed_dim as i32);
            launch_sgd_step(gpt.mlp.beta.ptr, gpt.mlp.dbeta.ptr, learning_rate, embed_dim as i32);

            launch_sgd_step(gpt.w_out.ptr, gpt.dw_out.ptr, learning_rate, (embed_dim * vocab_size) as i32);
            launch_sgd_step(gpt.b_out.ptr, gpt.db_out.ptr, learning_rate, vocab_size as i32);
        }

        if epoch % 50 == 0 || epoch == 1 {
            let mut loss_host = vec![0.0f32; 1];
            unsafe { gpu_copy_to_host(loss_host.as_mut_ptr() as *mut c_void, gpu_loss_ptr as *const c_void, 4); }
            let live_pred = generate_text(&gpt, &tokenizer, sample_text.chars().next().unwrap(), 60);
            let cleaned = live_pred.replace("\n", "\\n");
            println!("Epoch {:<4} | Loss: {:.4} | Live Output: '{}'", epoch, loss_host[0], cleaned);
        }
    }
    println!("\nReal GPU Training Complete!");
}

pub fn generate_text(gpt: &Transformer, tokenizer: &Tokenizer, seed_char: char, gen_len: usize) -> String {
    let mut current_ids = vec![0i32; gpt.sequence_length];
    if let Some(&id) = tokenizer.char_to_id.get(&seed_char) { current_ids[0] = id as i32; }
    let mut output_str = String::from(seed_char);
    let embed_dim = gpt.embed_dim;
    let seq_len = gpt.sequence_length;
    let vocab_size = gpt.vocab_size;

    let mut host_tok_emb = vec![0.0f32; vocab_size * embed_dim];
    let mut host_pos_emb = vec![0.0f32; seq_len * embed_dim];
    gpt.token_embeddings.copy_to_host(&mut host_tok_emb);
    gpt.position_embeddings.copy_to_host(&mut host_pos_emb);

    // One Arena pool for generation!
    let gpu_pool = GPUMemory::new(100_000);
    let gpu_x_ptr = unsafe { gpu_pool.ptr.add(0) };
    let gpu_q_ptr = unsafe { gpu_pool.ptr.add(seq_len * embed_dim) };
    let gpu_k_ptr = unsafe { gpu_pool.ptr.add(seq_len * embed_dim * 2) };
    let gpu_v_ptr = unsafe { gpu_pool.ptr.add(seq_len * embed_dim * 3) };
    let gpu_attn_ptr = unsafe { gpu_pool.ptr.add(seq_len * embed_dim * 4) };
    let gpu_logits_ptr = unsafe { gpu_pool.ptr.add(seq_len * embed_dim * 5) };

    for pos in 0..gen_len.min(seq_len - 1) {
        let mut host_x = vec![0.0f32; seq_len * embed_dim];
        for t in 0..=pos {
            let tok_id = current_ids[t] as usize;
            for d in 0..embed_dim {
                host_x[t * embed_dim + d] = host_tok_emb[tok_id * embed_dim + d] + host_pos_emb[t * embed_dim + d];
            }
        }
        unsafe {
            gpu_copy_to_device(gpu_x_ptr as *mut c_void, host_x.as_ptr() as *const c_void, host_x.len() * 4);
            launch_matmul_forward(gpu_q_ptr, gpu_x_ptr, gpt.attention.w_q.ptr, std::ptr::null(), seq_len as i32, embed_dim as i32, embed_dim as i32);
            launch_matmul_forward(gpu_k_ptr, gpu_x_ptr, gpt.attention.w_k.ptr, std::ptr::null(), seq_len as i32, embed_dim as i32, embed_dim as i32);
            launch_matmul_forward(gpu_v_ptr, gpu_x_ptr, gpt.attention.w_v.ptr, std::ptr::null(), seq_len as i32, embed_dim as i32, embed_dim as i32);
            launch_attention_forward(gpu_q_ptr, gpu_k_ptr, gpu_v_ptr, gpu_attn_ptr, seq_len as i32, embed_dim as i32);
            
            // Residual 1: Add input X to Attention Output
            launch_add_inplace(gpu_attn_ptr, gpu_x_ptr, (seq_len * embed_dim) as i32);

            launch_layernorm_forward(gpt.mlp.ln_out.ptr, gpt.mlp.cache_mean.ptr, gpt.mlp.cache_var.ptr, gpu_attn_ptr, gpt.mlp.gamma.ptr, gpt.mlp.beta.ptr, seq_len as i32, embed_dim as i32, 1e-5);
            launch_matmul_forward(gpt.mlp.z1.ptr, gpt.mlp.ln_out.ptr, gpt.mlp.w1.ptr, gpt.mlp.b1.ptr, seq_len as i32, embed_dim as i32, (embed_dim * 4) as i32);
            launch_relu_forward(gpt.mlp.a1.ptr, gpt.mlp.z1.ptr, (seq_len * embed_dim * 4) as i32);
            launch_matmul_forward(gpt.mlp.z2.ptr, gpt.mlp.a1.ptr, gpt.mlp.w2.ptr, gpt.mlp.b2.ptr, seq_len as i32, (embed_dim * 4) as i32, embed_dim as i32);

            // Residual 2: Add Attention Output (which holds X + Attn) to MLP Output
            launch_add_inplace(gpt.mlp.z2.ptr, gpu_attn_ptr, (seq_len * embed_dim) as i32);

            launch_matmul_forward(gpu_logits_ptr, gpt.mlp.z2.ptr, gpt.w_out.ptr, gpt.b_out.ptr, seq_len as i32, embed_dim as i32, vocab_size as i32);
        }

        let mut host_logits = vec![0.0f32; seq_len * vocab_size];
        unsafe { gpu_copy_to_host(host_logits.as_mut_ptr() as *mut c_void, gpu_logits_ptr as *const c_void, host_logits.len() * 4); }

        let row_start = pos * vocab_size;
        let mut max_val = -1e9f32;
        let mut best_id = 0usize;
        for v in 0..vocab_size {
            let val = host_logits[row_start + v];
            if val > max_val {
                max_val = val;
                best_id = v;
            }
        }
        if let Some(&c) = tokenizer.id_to_char.get(&best_id) {
            output_str.push(c);
            current_ids[pos + 1] = best_id as i32;
        }
    }
    output_str
}