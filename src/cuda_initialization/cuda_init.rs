use std::ffi::c_uint;

use crate::cuda_bindings::{cuInit, cudaError_t};

pub fn cuda_init(flags: u8) -> Result<(), cudaError_t> {
    let result = unsafe { cuInit(flags as c_uint) };

    match result {
        cudaError_t::cudaSuccess => Ok(()),
        _ => Err(result),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_init() {
        let flag = 0;
        let result = cuda_init(flag);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cuda_init_fail() {
        let flag = 1;
        let result = cuda_init(flag);
        assert!(result.is_err());
    }
}
