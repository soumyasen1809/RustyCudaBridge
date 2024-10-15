use libc::{c_int, c_uint};

use crate::{
    cuda_bindings::{
        cuDevicePrimaryCtxGetState, cuDevicePrimaryCtxRelease, cuDevicePrimaryCtxReset,
        cuDevicePrimaryCtxSetFlags, CUdevice,
    },
    cuda_errors::cudaError_t,
};

pub fn cuda_device_primary_ctx_get_state(
    dev: CUdevice,
    flags: &mut u32,
    active: &mut i32,
) -> Result<(), cudaError_t> {
    let result =
        unsafe { cuDevicePrimaryCtxGetState(dev, flags as *mut c_uint, active as *mut c_int) };

    match result {
        cudaError_t::cudaSuccess => Ok(()),
        _ => Err(result),
    }
}

pub fn cuda_device_primary_ctx_release(dev: CUdevice) -> Result<(), cudaError_t> {
    let result = unsafe { cuDevicePrimaryCtxRelease(dev) };

    match result {
        cudaError_t::cudaSuccess => Ok(()),
        _ => Err(result),
    }
}

pub fn cuda_device_primary_ctx_reset(dev: CUdevice) -> Result<(), cudaError_t> {
    let result = unsafe { cuDevicePrimaryCtxReset(dev) };

    match result {
        cudaError_t::cudaSuccess => Ok(()),
        _ => Err(result),
    }
}

pub fn cuda_device_primary_ctx_set_flags(dev: CUdevice, flags: u32) -> Result<(), cudaError_t> {
    let result = unsafe { cuDevicePrimaryCtxSetFlags(dev, flags as c_uint) };

    match result {
        cudaError_t::cudaSuccess => Ok(()),
        _ => Err(result),
    }
}

#[cfg(test)]
mod tests {
    use crate::cuda_initialization::cuda_init::cuda_init;

    use super::*;

    #[test]
    fn test_cuda_primary_ctx_get_state() {
        let mut flags: u32 = 1;
        let mut active: i32 = 1;
        let device: CUdevice = 0;

        cuda_init(0).expect("Failed to initialize"); // actual flag passed as 0

        cuda_device_primary_ctx_get_state(device, &mut flags, &mut active)
            .expect("Issue in primary context state");
        assert_eq!(flags, 0); // flag changes to 0
                              // assert_eq!(active, 0);
    }

    #[test]
    fn test_cuda_primary_ctx_release() {
        let device: CUdevice = 0;

        cuda_init(0).expect("Failed to initialize"); // actual flag passed as 0

        let result = cuda_device_primary_ctx_release(device);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cuda_primary_ctx_reset() {
        let device: CUdevice = 0;

        cuda_init(0).expect("Failed to initialize"); // actual flag passed as 0

        let result = cuda_device_primary_ctx_reset(device);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cuda_primary_ctx_set_flags() {
        let flags: u32 = 1;
        let device: CUdevice = 0;

        cuda_init(0).expect("Failed to initialize"); // actual flag passed as 0

        let result = cuda_device_primary_ctx_set_flags(device, flags);
        assert!(result.is_ok());
        assert_eq!(flags, 1); // flag changes to 1
    }
}
