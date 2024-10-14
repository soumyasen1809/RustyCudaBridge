use std::ffi::c_uint;

use crate::cuda_bindings::{cuCtxCreate, cuCtxDestroy, cudaError_t, CUcontext, CUdevice};

pub fn cuda_ctx_create(pctx: *mut CUcontext, flags: u32, dev: CUdevice) -> Result<(), cudaError_t> {
    let result = unsafe { cuCtxCreate(pctx, flags as c_uint, dev) };

    match result {
        cudaError_t::cudaSuccess => Ok(()),
        _ => Err(result),
    }
}

pub fn cuda_ctx_destroy(pctx: *mut CUcontext) -> Result<(), cudaError_t> {
    let result = unsafe { cuCtxDestroy(pctx) };
    println!("{:?}", result);

    match result {
        cudaError_t::cudaSuccess => Ok(()),
        _ => Err(result),
    }
}

#[cfg(test)]
mod tests {
    use crate::{cuda_bindings::CUctx_st, cuda_initialization::cuda_init::cuda_init};

    use super::*;

    #[test]
    fn test_cuda_ctx_create() {
        let mut pctx: *mut CUctx_st = std::ptr::null_mut();
        let flags: u32 = 0;
        let device: CUdevice = 0;

        cuda_init(0).expect("Failed to initialize");

        let result = cuda_ctx_create(&mut pctx as *mut CUcontext, flags, device);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cuda_ctx_destroy() {
        let mut pctx: *mut CUctx_st = std::ptr::null_mut();

        cuda_init(0).expect("Failed to initialize");

        let result = cuda_ctx_destroy(&mut pctx as *mut CUcontext).err().unwrap();
        assert_eq!(result, cudaError_t::cudaErrorContextIsDestroyed);
    }
}
