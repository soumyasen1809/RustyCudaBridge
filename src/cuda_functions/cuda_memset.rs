use std::ffi::c_void;

use libc::{c_int, size_t};

use super::cuda_bindings::{cudaError_t, cudaMemset};

pub fn cuda_memset(dev_ptr: *mut u8, value: i32, count: usize) -> Result<(), cudaError_t> {
    let result = unsafe { cudaMemset(dev_ptr as *mut c_void, value as c_int, count as size_t) };

    match result {
        cudaError_t::cudaSuccess => Ok(()),
        cudaError_t::cudaErrorInvalidValue => Err(result),
        cudaError_t::cudaErrorInvalidDevicePointer => Err(result),
        _ => Err(result),
    }
}

#[cfg(test)]
mod tests {
    use crate::cuda_functions::{
        cuda_free::cuda_free, cuda_malloc::cuda_malloc, cuda_memcpy::cuda_memcpy,
    };

    use super::*;

    #[test]
    fn test_cuda_memset() {
        let mut src = std::ptr::null_mut(); // This is almost never what you want
                                            // https://stackoverflow.com/questions/47878236/how-do-i-make-the-equivalent-of-a-c-double-pointer-in-rust
        let _ = cuda_malloc(&mut src, 10 * std::mem::size_of::<i32>());

        assert!(!src.is_null()); // assert src is not null means the memory allocation is successful

        let _ = cuda_memset(src, 20, 5);

        //  Copy the result back to the host
        let mut h_src = vec![0; 10];
        let _ = cuda_memcpy(
            h_src.as_mut_ptr(),
            src,
            h_src.len(),
            crate::cuda_functions::cuda_bindings::cudaMemcpyKind::cudaMemcpyDeviceToHost,
        );
        assert_eq!(h_src[0], 20);
        assert_eq!(h_src[4], 20);
        assert_eq!(h_src[5], 0);
        assert_eq!(h_src[9], 0);

        // Free cuda memory
        let _ = cuda_free(src);
    }
}
