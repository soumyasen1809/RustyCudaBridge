use crate::{
    cuda_bindings::{cuStreamCreate, CUstream},
    cuda_errors::cudaError_t,
    cuda_memory_enums::CUstream_flags,
};

pub fn cuda_stream_create(
    ph_stream: *mut CUstream,
    flags: CUstream_flags,
) -> Result<(), cudaError_t> {
    let result = unsafe { cuStreamCreate(ph_stream, flags) };

    match result {
        cudaError_t::cudaSuccess => Ok(()),
        _ => Err(result),
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        cuda_bindings::CUstream_st, cuda_memory_enums::CUstream_flags,
        cuda_memory_management::cuda_malloc::cuda_malloc,
    };

    use super::*;

    #[test]
    fn test_cuda_ctx_create() {
        let mut ph_stream: *mut CUstream_st = std::ptr::null_mut();
        cuda_malloc(
            &mut std::ptr::null_mut(),
            1 as usize * std::mem::size_of::<i32>(),
        )
        .unwrap(); // Note: removing cuda_malloc causes Issue in module_load: cudaErrorInitializationError (Why?)

        let result = cuda_stream_create(
            &mut ph_stream as *mut CUstream,
            CUstream_flags::CU_STREAM_DEFAULT,
        );
        assert!(result.is_ok());
    }
}
