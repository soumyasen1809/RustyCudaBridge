use std::ffi::c_void;

use super::cuda_bindings::{cudaError_t, cudaFreeHost};

pub fn cuda_free_host(ptr: *mut u8) -> Result<(), cudaError_t> {
    let result = unsafe { cudaFreeHost(ptr as *mut c_void) };

    match result {
        cudaError_t::cudaSuccess => Ok(()),
        cudaError_t::cudaErrorInitializationError => Err(result),
        _ => Err(result),
    }
}

#[cfg(test)]
mod tests {
    use crate::cuda_functions::{
        cuda_bindings::cudaHostAllocFlag, cuda_host_alloc::cuda_host_alloc,
    };

    use super::*;

    #[test]
    fn test_cuda_malloc() {
        let src_vec = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let mut src = std::ptr::null_mut();
        // The src pointer is initialized to std::ptr::null_mut(),
        // which means it doesn't point to any valid memory location

        let result = cuda_host_alloc(
            &mut src,
            10 * std::mem::size_of::<i32>(),
            cudaHostAllocFlag::cudaHostAllocDefault,
        );
        assert!(result.is_ok());

        //Data can be copied only when you have allocated memory (done using cuda_host_alloc)
        unsafe {
            std::ptr::copy(src_vec.as_ptr(), src as *mut i32, src_vec.len());
        }

        // Free host allocated memory
        let result = cuda_free_host(src);
        assert!(result.is_ok());
    }
}
