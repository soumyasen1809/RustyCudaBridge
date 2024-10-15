use libc::size_t;

use crate::{
    cuda_bindings::{cuMemAdvise, CUdevice, CUdeviceptr},
    cuda_errors::cudaError_t,
    cuda_memory_enums::CUmem_advise,
};

pub fn cuda_mem_advise(
    dev_ptr: CUdeviceptr,
    count: usize,
    advice: CUmem_advise,
    device: CUdevice,
) -> Result<(), cudaError_t> {
    let result = unsafe { cuMemAdvise(dev_ptr, count as size_t, advice, device) };
    println!("{:?}", result);

    match result {
        cudaError_t::cudaSuccess => Ok(()),
        _ => Err(result),
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        cuda_device_management::cuda_device_get::cuda_device_get,
        cuda_initialization::cuda_init::cuda_init,
        cuda_memory_management::cuda_malloc::cuda_malloc,
    };

    use super::*;

    #[test]
    fn test_cuda_ctx_create() {
        let mut devptr: CUdeviceptr = 0;
        let advise = CUmem_advise::CU_MEM_ADVISE_SET_READ_MOSTLY;
        let count: usize = std::mem::size_of::<i32>();
        let mut device: CUdevice = 0;

        cuda_init(0).expect("Failed to initialize");
        cuda_device_get(&mut device, 0).expect("Issue in getting device");
        cuda_malloc(&mut devptr as *mut CUdeviceptr as *mut *mut u8, count).unwrap(); // needed else error uninitialized

        let result = cuda_mem_advise(devptr as CUdeviceptr, count, advise, device);
        assert!(result.is_err()); // WRONG!
    }
}
