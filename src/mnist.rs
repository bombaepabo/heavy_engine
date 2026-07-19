use std::fs::File;
use std::io::Read;

// Loads the images and normalizes the pixels to be between 0.0 and 1.0
pub fn load_images(filename: &str) -> Vec<f32> {
    let mut file = File::open(filename).expect("Failed to open MNIST images file");
    
    // The MNIST format has a 16-byte text header we need to skip over
    let mut header = [0u8; 16];
    file.read_exact(&mut header).unwrap();

    let mut data = Vec::new();
    file.read_to_end(&mut data).unwrap();

    // Convert pixels from 0-255 down to 0.0-1.0 so the AI's weights don't explode
    data.into_iter().map(|x| x as f32 / 255.0).collect()
}

// Loads the correct answers (digits 0 through 9)
pub fn load_labels(filename: &str) -> Vec<usize> {
    let mut file = File::open(filename).expect("Failed to open MNIST labels file");
    
    // Labels only have an 8-byte header to skip
    let mut header = [0u8; 8];
    file.read_exact(&mut header).unwrap();

    let mut data = Vec::new();
    file.read_to_end(&mut data).unwrap();

    data.into_iter().map(|x| x as usize).collect()
}