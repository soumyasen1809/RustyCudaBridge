use std::ffi::c_void;

use super::cuda_bindings::{cudaError_t, cudaFree};

pub fn cuda_free(dev_ptr: *mut u8) -> Result<(), cudaError_t> {
    let result = unsafe { cudaFree(dev_ptr as *mut c_void) };

    match result {
        cudaError_t::cudaSuccess => Ok(()),
        cudaError_t::cudaErrorInvalidDevicePointer => Err(result),
        cudaError_t::cudaErrorInitializationError => Err(result),
        _ => Err(result),
    }
}

#[cfg(test)]
mod tests {
    use crate::cuda_functions::cuda_malloc::cuda_malloc;

    use super::*;

    #[test]
    fn test_cuda_malloc() {
        let mut src = std::ptr::null_mut(); // This is almost never what you want
                                            // https://stackoverflow.com/questions/47878236/how-do-i-make-the-equivalent-of-a-c-double-pointer-in-rust
        let _ = cuda_malloc(&mut src, 10 * std::mem::size_of::<i32>());
        assert!(!src.is_null()); // assert src is not null means the memory allocation is successful

        let result = cuda_free(src);
        // assert!(src.is_null()); // WRONG assert: cudaFree does not modify the pointer: It deallocates the memory that src points to on the device, but it doesn't nullify or change src itself.
        assert!(result.is_ok());
    }
}
