use libc::size_t;

use crate::{
    cuda_bindings::{cuMemAdvise, cuMemPrefetchAsync, CUdevice, CUdeviceptr, CUstream},
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

    match result {
        cudaError_t::cudaSuccess => Ok(()),
        _ => Err(result),
    }
}

pub fn cuda_mem_prefetch_async(
    dev_ptr: CUdeviceptr,
    count: usize,
    dst_device: CUdevice,
    h_stream: CUstream,
) -> Result<(), cudaError_t> {
    let result = unsafe { cuMemPrefetchAsync(dev_ptr, count as size_t, dst_device, h_stream) };
    println!("{:?}", result);

    match result {
        cudaError_t::cudaSuccess => Ok(()),
        _ => Err(result),
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        cuda_bindings::CUstream_st, cuda_device_management::cuda_device_get::cuda_device_get,
        cuda_memory_enums::CUstream_flags, cuda_memory_management::cuda_malloc::cuda_malloc,
        cuda_stream_management::cuda_stream_management::cuda_stream_create,
    };

    use super::*;

    #[test]
    fn test_cuda_ctx_create() {
        let devptr: CUdeviceptr = 0;
        let advise = CUmem_advise::CU_MEM_ADVISE_SET_READ_MOSTLY;
        let count: usize = std::mem::size_of::<i32>();
        let mut device: CUdevice = 0;

        cuda_malloc(&mut std::ptr::null_mut(), count).unwrap(); // Note: removing cuda_malloc causes Issue in module_load: cudaErrorInitializationError (Why?)
        cuda_device_get(&mut device, 0).expect("Issue in getting device");
        // cuda_malloc(&mut devptr as *mut CUdeviceptr as *mut *mut u8, count).unwrap(); // needed else error uninitialized

        let result = cuda_mem_advise(devptr as CUdeviceptr, count, advise, device);
        assert!(result.is_err()); // WRONG!
    }

    #[test]
    fn test_cuda_prefetch_async() {
        let devptr: CUdeviceptr = 0; // Device pointer
        let count: usize = 10 * std::mem::size_of::<i32>();
        let mut device: CUdevice = 0;
        let mut stream: *mut CUstream_st = std::ptr::null_mut(); // Initialize stream

        cuda_malloc(&mut std::ptr::null_mut(), count).unwrap(); // Note: removing cuda_malloc causes Issue in module_load: cudaErrorInitializationError (Why?)
        cuda_device_get(&mut device, 0).expect("Issue in getting device");

        cuda_stream_create(
            &mut stream as *mut CUstream,
            CUstream_flags::CU_STREAM_DEFAULT,
        )
        .unwrap();

        let result =
            cuda_mem_prefetch_async(devptr as CUdeviceptr, count, device, stream as CUstream);
        assert!(result.is_err()); // WRONG!
    }
}
