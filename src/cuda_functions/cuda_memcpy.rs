use libc::size_t;
use std::ffi::c_void;

use crate::cuda_functions::cuda_bindings::{cudaError_t, cudaMemcpy, cudaMemcpyKind};

pub fn cuda_memcpy(
    dst: *mut u8,
    src: *const u8,
    count: usize,
    kind: cudaMemcpyKind,
) -> Result<(), cudaError_t> {
    let cpy_result = unsafe {
        cudaMemcpy(
            dst as *mut c_void,
            src as *const c_void,
            count as size_t,
            kind,
        )
    };

    match cpy_result {
        cudaError_t::cudaSuccess => Ok(()),
        cudaError_t::cudaErrorInvalidValue => Err(cpy_result),
        _ => Err(cpy_result),
    }
}

#[cfg(test)]
mod tests {
    use crate::cuda_functions::cuda_malloc::cuda_malloc;

    use super::*;

    #[test]
    fn test_cuda_memcpy_fail() {
        let src = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let mut dst = vec![0; src.len()];
        let test_result = cuda_memcpy(
            dst.as_mut_ptr(),
            src.as_ptr(),
            src.len(),
            cudaMemcpyKind::cudaMemcpyHostToDevice,
        )
        .unwrap_err(); // Error because dst needs to be on the device

        assert!(test_result == cudaError_t::cudaErrorInvalidValue);
    }

    #[test]
    fn test_cuda_memcpy() {
        let src = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let mut dst = std::ptr::null_mut();
        let _ = cuda_malloc(&mut dst, src.len() * size_of::<i32>());
        let test_result = cuda_memcpy(
            dst,
            src.as_ptr(),
            src.len(),
            cudaMemcpyKind::cudaMemcpyHostToDevice,
        );

        assert!(test_result.is_ok());
    }
}
