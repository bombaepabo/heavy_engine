use std::collections::{HashMap, HashSet};

pub struct Tokenizer {
    pub char_to_id: HashMap<char, usize>,
    pub id_to_char: HashMap<usize, char>,
    pub vocab_size: usize,
}

impl Tokenizer {
    pub fn new(text: &str) -> Self {
        // Find every unique character in the text
        let mut unique_chars: Vec<char> = text.chars().collect::<HashSet<_>>().into_iter().collect();
        unique_chars.sort(); // Keep them in alphabetical order so it's consistent

        let mut char_to_id = HashMap::new();
        let mut id_to_char = HashMap::new();

        // Assign an integer ID to every character
        for (id, &c) in unique_chars.iter().enumerate() {
            char_to_id.insert(c, id);
            id_to_char.insert(id, c);
        }

        let vocab_size = unique_chars.len();
        println!("Tokenizer created! Found {} unique characters in Shakespeare.", vocab_size);

        Self {
            char_to_id,
            id_to_char,
            vocab_size,
        }
    }

    // Convert Human Text -> Numbers for the GPU
    pub fn encode(&self, text: &str) -> Vec<usize> {
        text.chars().map(|c| *self.char_to_id.get(&c).unwrap_or(&0)).collect()
    }

    // Convert Numbers from the GPU -> Human Text
    pub fn decode(&self, ids: &[usize]) -> String {
        ids.iter().map(|id| *self.id_to_char.get(id).unwrap_or(&'?')).collect()
    }
}