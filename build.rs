fn main() {
    cc::Build::new()
        .cuda(true)
        .file("src/cuda/kernel.cu") // Path updated to src/cuda/kernel.cu
        .compile("cuda_kernels");

    println!("cargo:rerun-if-changed=src/cuda/kernel.cu");
    println!("cargo:rustc-link-lib=dylib=cudart");
}