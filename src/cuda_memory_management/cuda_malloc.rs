use libc::size_t;
use std::ffi::c_void;

use crate::{cuda_bindings::*, cuda_errors::cudaError_t};

pub fn cuda_malloc(dev_ptr: *mut *mut u8, size: usize) -> Result<(), cudaError_t> {
    let result = unsafe { cudaMalloc(dev_ptr as *mut *mut c_void, size as size_t) };

    match result {
        cudaError_t::cudaSuccess => Ok(()),
        cudaError_t::cudaErrorMemoryAllocation => Err(result),
        _ => Err(result),
    }
}

#[cfg(test)]
mod tests {
    use crate::cuda_memory_management::cuda_free::cuda_free;

    use super::*;

    #[test]
    fn test_cuda_malloc() {
        let mut src = std::ptr::null_mut(); // This is almost never what you want
                                            // https://stackoverflow.com/questions/47878236/how-do-i-make-the-equivalent-of-a-c-double-pointer-in-rust
        let _ = cuda_malloc(&mut src, 10 * std::mem::size_of::<f64>());

        assert!(!src.is_null()); // assert src is not null means the memory allocation is successful

        // Free cuda memory
        let _ = cuda_free(src);
    }
}
