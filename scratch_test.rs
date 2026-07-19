use std::ffi::{c_void, CStr};
use std::os::raw::c_char;

#[link(name = "cudart")]
extern "C" {
    fn cudaMalloc(devPtr: *mut *mut c_void, size: usize) -> i32;
    fn cudaGetErrorString(error: i32) -> *const c_char;
}

fn main() {
    unsafe {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        println!("Testing cudaMalloc...");
        for i in 0..100 {
            let err = cudaMalloc(&mut ptr, 80000);
            if err != 0 {
                let err_str = CStr::from_ptr(cudaGetErrorString(err));
                println!("CUDA ERROR at {}: {}", i, err_str.to_string_lossy());
                return;
            }
        }
        println!("Success! Allocated 100 times.");
    }
}
