use std::ffi::c_uint;

use crate::{
    cuda_bindings::{
        cuCtxCreate, cuCtxDestroy, cuCtxGetApiVersion, cuCtxGetCurrent, cuCtxPopCurrent,
        cuCtxPushCurrent, cuCtxSynchronize, CUcontext, CUdevice,
    },
    cuda_errors::cudaError_t,
};

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

pub fn cuda_ctx_get_api_version(ctx: CUcontext, version: *mut u32) -> Result<(), cudaError_t> {
    let result = unsafe { cuCtxGetApiVersion(ctx, version as *mut c_uint) };
    println!("{:?}", result);

    match result {
        cudaError_t::cudaSuccess => Ok(()),
        _ => Err(result),
    }
}

pub fn cuda_ctx_get_current(pctx: *mut CUcontext) -> Result<(), cudaError_t> {
    let result = unsafe { cuCtxGetCurrent(pctx) };

    match result {
        cudaError_t::cudaSuccess => Ok(()),
        _ => Err(result),
    }
}

pub fn cuda_ctx_pop_current(pctx: *mut CUcontext) -> Result<(), cudaError_t> {
    let result = unsafe { cuCtxPopCurrent(pctx) };

    match result {
        cudaError_t::cudaSuccess => Ok(()),
        _ => Err(result),
    }
}

pub fn cuda_ctx_push_current(ctx: CUcontext) -> Result<(), cudaError_t> {
    let result = unsafe { cuCtxPushCurrent(ctx) };
    println!("result: {:?}", result);

    match result {
        cudaError_t::cudaSuccess => Ok(()),
        _ => Err(result),
    }
}

pub fn cuda_ctx_synchronize() -> Result<(), cudaError_t> {
    let result = unsafe { cuCtxSynchronize() };
    println!("result: {:?}", result);

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

    #[test]
    fn test_cuda_ctx_get_api_version() {
        let mut pctx: *mut CUctx_st = std::ptr::null_mut();
        let flags: u32 = 0;
        let device: CUdevice = 0;

        cuda_init(0).expect("Failed to initialize");

        cuda_ctx_create(&mut pctx as *mut CUcontext, flags, device).unwrap();

        let mut version: u32 = 0;

        let result = cuda_ctx_get_api_version(pctx as CUcontext, &mut version as *mut u32);
        assert!(result.is_ok());
        assert_eq!(version, 3010);
    }

    #[test]
    fn test_cuda_ctx_get_current() {
        let mut pctx: *mut CUctx_st = std::ptr::null_mut();
        let flags: u32 = 0;
        let device: CUdevice = 0;

        cuda_init(0).expect("Failed to initialize");

        cuda_ctx_create(&mut pctx as *mut CUcontext, flags, device).unwrap();

        let mut pctx_new: *mut CUctx_st = std::ptr::null_mut();
        let result = cuda_ctx_get_current(&mut pctx_new as *mut CUcontext);
        assert!(result.is_ok());
        assert_eq!(pctx_new, pctx);
    }

    #[test]
    fn test_cuda_ctx_pop_current() {
        let mut pctx: *mut CUctx_st = std::ptr::null_mut();
        let flags: u32 = 0;
        let device: CUdevice = 0;

        cuda_init(0).expect("Failed to initialize");

        cuda_ctx_create(&mut pctx as *mut CUcontext, flags, device).unwrap();

        let result = cuda_ctx_pop_current(&mut pctx as *mut CUcontext);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cuda_ctx_push_current() {
        let mut ctx: *mut CUctx_st = std::ptr::null_mut();
        let flags: u32 = 0;
        let device: CUdevice = 0;

        cuda_init(0).expect("Failed to initialize");

        cuda_ctx_create(&mut ctx as *mut CUcontext, flags, device).unwrap();
        // Ensure that the context is valid (not null)
        assert!(!ctx.is_null(), "CUDA context is null after creation");

        cuda_ctx_pop_current(&mut ctx as *mut CUcontext).unwrap(); // pop the created context to ctx

        let result = cuda_ctx_push_current(ctx as CUcontext); // push the ctx obtained to CUcontext list
        assert!(result.is_ok());
    }

    #[test]
    fn test_cuda_ctx_sync() {
        let mut pctx: *mut CUctx_st = std::ptr::null_mut();
        let flags: u32 = 0;
        let device: CUdevice = 0;

        cuda_init(0).expect("Failed to initialize");

        cuda_ctx_create(&mut pctx as *mut CUcontext, flags, device).unwrap();

        let result = cuda_ctx_synchronize();
        assert!(result.is_ok());
    }
}
