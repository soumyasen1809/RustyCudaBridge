use std::ffi::{c_char, c_int};

use crate::cuda_bindings::{cuDeviceGet, cuDeviceGetCount, cuDeviceGetName, cudaError_t, CUdevice};

pub fn cuda_device_get(device: *mut CUdevice, ordinal: i32) -> Result<(), cudaError_t> {
    let result = unsafe { cuDeviceGet(device, ordinal as c_int) };

    match result {
        cudaError_t::cudaSuccess => Ok(()),
        _ => Err(result),
    }
}

pub fn cuda_device_get_count(count: *mut i32) -> Result<(), cudaError_t> {
    let result = unsafe { cuDeviceGetCount(count as *mut c_int) };

    match result {
        cudaError_t::cudaSuccess => Ok(()),
        _ => Err(result),
    }
}

pub fn cuda_device_get_name(
    name: *mut char,
    max_string_len: i32,
    dev: CUdevice,
) -> Result<(), cudaError_t> {
    let result = unsafe { cuDeviceGetName(name as *mut c_char, max_string_len, dev) };

    match result {
        cudaError_t::cudaSuccess => Ok(()),
        _ => Err(result),
    }
}

#[cfg(test)]
mod tests {
    use core::str;

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

    #[test]
    fn test_cuda_get_device_count() {
        let mut count: i32 = 0;

        cuda_init(0).expect("Failed to initialize");

        cuda_device_get_count(&mut count as *mut i32).expect("Issue in getting device");
        assert_eq!(count, 1);
    }

    #[test]
    fn test_cuda_get_device_name() {
        let max_len: i32 = 128;
        let device: CUdevice = 0;
        let mut name = [0u8; 128];

        cuda_init(0).expect("Failed to initialize");

        cuda_device_get_name(&mut name[0] as *mut u8 as *mut char, max_len, device)
            .expect("Issue in getting device");
        // Alternative: // cuda_device_get_name(name.as_mut_ptr() as *mut char, max_len, device).expect("Issue in getting device");

        let device_name_expected = "NVIDIA GeForce GTX 1650";
        let name_obtained = str::from_utf8(&name).unwrap();
        assert!(name_obtained.contains(device_name_expected));
    }

    #[test]
    fn test_cuda_get_device_name_with_string_type() {
        // Alternative test to test_cuda_get_device_name()
        let max_len: i32 = 128;
        let device: CUdevice = 0;
        let mut name = String::with_capacity(max_len as usize);
        // IMPORTANT STEP: length of the string (name) is not updated
        // when you directly use the internal pointer to pass data to the CUDA function.
        // In Rust, String::as_mut_ptr() returns a raw pointer to the string's internal buffer.
        // However, this does not automatically increase the string's length when the
        // CUDA function writes data into that buffer.
        unsafe { name.as_mut_vec().set_len(max_len as usize - 1) };

        cuda_init(0).expect("Failed to initialize");

        cuda_device_get_name(name.as_mut_ptr() as *mut char, max_len, device)
            .expect("Issue in getting device");

        let device_name_expected = "NVIDIA GeForce GTX 1650";
        assert!(name.contains(device_name_expected));
    }
}
