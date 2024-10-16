use crate::{
    cuda_bindings::{
        cuStreamBeginCapture, cuStreamCopyAttributes, cuStreamCreate, cuStreamCreateWithPriority,
        cuStreamDestroy, CUstream,
    },
    cuda_errors::cudaError_t,
    cuda_memory_enums::{CUstreamCaptureMode, CUstream_flags},
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

pub fn cuda_stream_begin_capture(
    h_stream: CUstream,
    mode: CUstreamCaptureMode,
) -> Result<(), cudaError_t> {
    let result = unsafe { cuStreamBeginCapture(h_stream, mode) };

    match result {
        cudaError_t::cudaSuccess => Ok(()),
        _ => Err(result),
    }
}

pub fn cuda_stream_copy_attributes(dst: CUstream, src: CUstream) -> Result<(), cudaError_t> {
    let result = unsafe { cuStreamCopyAttributes(dst, src) };
    println!("result: {:?}", result);

    match result {
        cudaError_t::cudaSuccess => Ok(()),
        _ => Err(result),
    }
}

pub fn cuda_stream_create_with_prio(
    ph_stream: *mut CUstream,
    flags: CUstream_flags,
    prio: i32,
) -> Result<(), cudaError_t> {
    let result = unsafe { cuStreamCreateWithPriority(ph_stream, flags, prio) };

    match result {
        cudaError_t::cudaSuccess => Ok(()),
        _ => Err(result),
    }
}

pub fn cuda_stream_destroy(h_stream: CUstream) -> Result<(), cudaError_t> {
    let result = unsafe { cuStreamDestroy(h_stream) };

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
    fn test_cuda_stream_create() {
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

    #[test]
    fn test_cuda_begin_stream_capture() {
        let mut ph_stream: *mut CUstream_st = std::ptr::null_mut();
        cuda_malloc(
            &mut std::ptr::null_mut(),
            1 as usize * std::mem::size_of::<i32>(),
        )
        .unwrap(); // Note: removing cuda_malloc causes Issue in module_load: cudaErrorInitializationError (Why?)

        cuda_stream_create(
            &mut ph_stream as *mut CUstream,
            CUstream_flags::CU_STREAM_DEFAULT,
        )
        .unwrap();

        let result = cuda_stream_begin_capture(
            ph_stream,
            CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_GLOBAL,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_cuda_stream_copy_attribute() {
        let dst_stream: *mut CUstream_st = std::ptr::null_mut();
        let mut src_stream: *mut CUstream_st = std::ptr::null_mut();

        cuda_malloc(
            &mut std::ptr::null_mut(),
            1 as usize * std::mem::size_of::<i32>(),
        )
        .unwrap(); // Note: removing cuda_malloc causes Issue in module_load: cudaErrorInitializationError (Why?)

        cuda_stream_create(
            &mut src_stream as *mut CUstream,
            CUstream_flags::CU_STREAM_DEFAULT,
        )
        .unwrap();

        let result = cuda_stream_copy_attributes(dst_stream, src_stream);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cuda_stream_create_with_prio() {
        let mut ph_stream: *mut CUstream_st = std::ptr::null_mut();
        cuda_malloc(
            &mut std::ptr::null_mut(),
            1 as usize * std::mem::size_of::<i32>(),
        )
        .unwrap(); // Note: removing cuda_malloc causes Issue in module_load: cudaErrorInitializationError (Why?)

        let result = cuda_stream_create_with_prio(
            &mut ph_stream as *mut CUstream,
            CUstream_flags::CU_STREAM_DEFAULT,
            1,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_cuda_stream_destroy() {
        let mut h_stream: *mut CUstream_st = std::ptr::null_mut();

        cuda_malloc(
            &mut std::ptr::null_mut(),
            1 as usize * std::mem::size_of::<i32>(),
        )
        .unwrap(); // Note: removing cuda_malloc causes Issue in module_load: cudaErrorInitializationError (Why?)

        cuda_stream_create(
            &mut h_stream as *mut CUstream,
            CUstream_flags::CU_STREAM_DEFAULT,
        )
        .unwrap();

        let result = cuda_stream_destroy(h_stream);
        assert!(result.is_ok());
    }
}
