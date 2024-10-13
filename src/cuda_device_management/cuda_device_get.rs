use std::ffi::c_int;

use crate::cuda_bindings::{cuDeviceGet, cudaError_t, CUdevice};

pub fn cuda_device_get(device: *mut CUdevice, ordinal: i32) -> Result<(), cudaError_t> {
    let result = unsafe { cuDeviceGet(device, ordinal as c_int) };

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
    fn test_cuda_get_device() {
        let ordinal = 0;
        let mut device: CUdevice = 0;

        cuda_init(0).expect("Failed to initialize");

        cuda_device_get(&mut device, ordinal).expect("Issue in getting device");
        assert_eq!(device, 0);
    }
}
