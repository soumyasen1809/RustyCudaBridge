use std::ffi::c_void;

use crate::{cuda_bindings::*, cuda_errors::cudaError_t};

pub fn cuda_host_alloc(
    p_host: *mut *mut u8,
    size: usize,
    flags: cudaHostAllocFlag,
) -> Result<(), cudaError_t> {
    let result = unsafe { cudaHostAlloc(p_host as *mut *mut c_void, size, flags) };

    match result {
        cudaError_t::cudaSuccess => Ok(()),
        cudaError_t::cudaErrorMemoryAllocation => Err(result),
        _ => Err(result),
    }
}

#[cfg(test)]
mod tests {
    use crate::cuda_functions::cuda_free::cuda_free;

    use super::*;

    #[test]
    fn test_cuda_host_alloc() {
        let mut src = std::ptr::null_mut();

        let result = cuda_host_alloc(
            &mut src,
            10 * std::mem::size_of::<i32>(),
            cudaHostAllocFlag::cudaHostAllocDefault,
        );
        assert!(result.is_ok());

        // Free cuda memory
        let _ = cuda_free(src);
    }
}
