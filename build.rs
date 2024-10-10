/// /usr/local/cuda/lib64 contains other libcuda files like libcublas.so, libcudart.so etc, but not libcuda.so
/// If you don't see libcuda.so in /usr/local/cuda/lib64, but you do see other CUDA-related libraries like libcublas.so
/// and libcudart.so, it's because libcuda.so is not part of the CUDA toolkit itself. Instead, it's part of the NVIDIA
/// driver, and should be installed with the driver rather than with the CUDA toolkit.
/// libcudart.so (CUDA Runtime API): This is the CUDA runtime library (CUDA Runtime API) that provides functions like
/// cudaMemcpy. You should be linking against this library for your application, rather than libcuda.so
fn main() {
    // Link to the CUDA runtime library (libcudart)
    println!("cargo:rustc-link-lib=cudart");

    println!("cargo:rustc-link-search=native=cuda_kernels"); // Path to where libvec_add.so is
                                                             // Specify the directory containing `libadd.so` (not the file itself)
    println!("cargo:rustc-link-lib=dylib=add"); // Link to libvec_add.so (without the 'lib' prefix)
    println!("cargo:rustc-link-arg=-Wl,-rpath=./cuda_kernels"); // Add rpath to embed the library search path in the binary

    // If necessary, specify the path to the CUDA library (if installed in a non-standard location)
    // println!("cargo:rustc-link-search=/usr/local/cuda/lib64");
}
