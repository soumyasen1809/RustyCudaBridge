use std::ffi::{c_char, c_int};

use libc::size_t;

use crate::{
    cuda_bindings::{
        cuDeviceGet, cuDeviceGetAttribute, cuDeviceGetCount, cuDeviceGetDefaultMemPool,
        cuDeviceGetMemPool, cuDeviceGetName, cuDeviceSetMemPool, cuDeviceTotalMem, CUdevice,
        CUmemoryPool,
    },
    cuda_device_attributes::CUdevice_attribute,
    cuda_errors::cudaError_t,
};

pub fn cuda_device_get(device: *mut CUdevice, ordinal: i32) -> Result<(), cudaError_t> {
    let result = unsafe { cuDeviceGet(device, ordinal as c_int) };

    match result {
        cudaError_t::cudaSuccess => Ok(()),
        _ => Err(result),
    }
}

pub fn cuda_device_get_attribute(
    pi: *mut i32,
    attrib: CUdevice_attribute,
    dev: CUdevice,
) -> Result<(), cudaError_t> {
    let result = unsafe { cuDeviceGetAttribute(pi as *mut c_int, attrib, dev) };

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

pub fn cuda_device_get_default_mem_pool(
    pool_out: *mut CUmemoryPool,
    dev: CUdevice,
) -> Result<(), cudaError_t> {
    let result = unsafe { cuDeviceGetDefaultMemPool(pool_out, dev) };

    match result {
        cudaError_t::cudaSuccess => Ok(()),
        _ => Err(result),
    }
}

pub fn cuda_device_get_mem_pool(
    pool_out: *mut CUmemoryPool,
    dev: CUdevice,
) -> Result<(), cudaError_t> {
    let result = unsafe { cuDeviceGetMemPool(pool_out, dev) };

    match result {
        cudaError_t::cudaSuccess => Ok(()),
        _ => Err(result),
    }
}

pub fn cuda_device_set_mem_pool(dev: CUdevice, pool_out: CUmemoryPool) -> Result<(), cudaError_t> {
    let result = unsafe { cuDeviceSetMemPool(dev, pool_out) };

    match result {
        cudaError_t::cudaSuccess => Ok(()),
        _ => Err(result),
    }
}

pub fn cuda_device_total_mem(bytes: &mut usize, dev: CUdevice) -> Result<(), cudaError_t> {
    let result = unsafe { cuDeviceTotalMem(bytes as *mut size_t, dev) };

    match result {
        cudaError_t::cudaSuccess => Ok(()),
        _ => Err(result),
    }
}

#[cfg(test)]
mod tests {
    use core::str;

    use crate::{cuda_bindings::CUmemPoolHandle_st, cuda_initialization::cuda_init::cuda_init};

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
    fn test_cuda_get_attribute() {
        let mut attribute_value = 0;
        let atribute_to_query = CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X;
        let device: CUdevice = 0;

        cuda_init(0).expect("Failed to initialize");

        cuda_device_get_attribute(&mut attribute_value as *mut i32, atribute_to_query, device)
            .expect("Issue in getting attribute");
        assert_eq!(attribute_value, 1024);
    }

    #[test]
    fn test_cuda_get_attribute_unified_addressing() {
        let mut attribute_value = 0;
        let atribute_to_query = CUdevice_attribute::CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING;
        let device: CUdevice = 0;

        cuda_init(0).expect("Failed to initialize");

        cuda_device_get_attribute(&mut attribute_value as *mut i32, atribute_to_query, device)
            .expect("Issue in getting attribute");
        assert_eq!(attribute_value, 1);
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

    #[test]
    fn test_cuda_get_device_default_mem_pool() {
        let mut pool_out: *mut CUmemPoolHandle_st = std::ptr::null_mut();
        let device: CUdevice = 0;

        cuda_init(0).expect("Failed to initialize");

        let result = cuda_device_get_default_mem_pool(&mut pool_out as *mut CUmemoryPool, device);
        assert!(result.is_ok());
        assert_ne!(pool_out, std::ptr::null_mut() as *mut CUmemPoolHandle_st); // Ensure the returned memory pool handle is not null
    }

    #[test]
    fn test_cuda_get_device_mem_pool() {
        let mut pool_out: *mut CUmemPoolHandle_st = std::ptr::null_mut();
        let device: CUdevice = 0;

        cuda_init(0).expect("Failed to initialize");

        let result = cuda_device_get_mem_pool(&mut pool_out as *mut CUmemoryPool, device);
        assert!(result.is_ok());
        assert_ne!(pool_out, std::ptr::null_mut() as *mut CUmemPoolHandle_st); // Ensure the returned memory pool handle is not null
    }

    #[test]
    fn test_cuda_set_device_mem_pool() {
        let mut pool_out: *mut CUmemPoolHandle_st = std::ptr::null_mut();
        let device: CUdevice = 0;

        cuda_init(0).expect("Failed to initialize");

        // Retrieve the memory pool first to get a valid pool handle for the device
        let get_pool_result = cuda_device_get_mem_pool(&mut pool_out as *mut CUmemoryPool, device);
        assert!(get_pool_result.is_ok());
        assert_ne!(pool_out, std::ptr::null_mut() as *mut CUmemPoolHandle_st); // Ensure the returned memory pool handle is not null

        // Set the device memory pool next
        let set_pool_result = cuda_device_set_mem_pool(device, pool_out as CUmemoryPool);
        assert!(set_pool_result.is_ok());
        assert_ne!(pool_out, std::ptr::null_mut() as *mut CUmemPoolHandle_st); // Ensure the returned memory pool handle is not null
    }

    #[test]
    fn test_cuda_get_total_mem() {
        let mut mem_bytes: usize = 0;
        let device: CUdevice = 0;

        cuda_init(0).expect("Failed to initialize");

        cuda_device_total_mem(&mut mem_bytes, device).expect("Issue in getting total memory");
        assert_eq!(mem_bytes, 4294639616);
    }
}
