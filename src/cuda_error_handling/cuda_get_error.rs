use std::ffi::c_char;

use crate::{
    cuda_bindings::{cuGetErrorName, cuGetErrorString},
    cuda_errors::cudaError_t,
};

pub fn cuda_get_error_name(
    err: cudaError_t,
    string_rep: &mut *const c_char,
) -> Result<(), cudaError_t> {
    let result = unsafe { cuGetErrorName(err, string_rep as *mut *const c_char) };

    match result {
        cudaError_t::cudaSuccess => Ok(()),
        _ => Err(result),
    }
}

pub fn cuda_get_error_string(
    err: cudaError_t,
    string_rep: &mut *const c_char,
) -> Result<(), cudaError_t> {
    let result = unsafe { cuGetErrorString(err, string_rep as *mut *const c_char) };

    match result {
        cudaError_t::cudaSuccess => Ok(()),
        _ => Err(result),
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::CStr;

    use crate::cuda_functions::cuda_malloc::cuda_malloc;

    use super::*;

    #[test]
    fn test_cuda_get_error_name() {
        cuda_malloc(
            &mut std::ptr::null_mut(),
            1 as usize * std::mem::size_of::<i32>(),
        )
        .unwrap(); // Note: removing cuda_malloc causes Issue in module_load: cudaErrorInitializationError (Why?)
        let error_code = cudaError_t::cudaErrorCapturedEvent;
        let mut string_rep: *const c_char = std::ptr::null();

        cuda_get_error_name(error_code, &mut string_rep).unwrap();

        // Convert string_rep to Rust string for comparison
        let error_name = unsafe { CStr::from_ptr(string_rep) }.to_str().unwrap();
        assert_eq!(error_name, "CUDA_ERROR_CAPTURED_EVENT");
    }

    #[test]
    fn test_cuda_get_error_string() {
        cuda_malloc(
            &mut std::ptr::null_mut(),
            1 as usize * std::mem::size_of::<i32>(),
        )
        .unwrap(); // Note: removing cuda_malloc causes Issue in module_load: cudaErrorInitializationError (Why?)
        let error_code = cudaError_t::cudaErrorCapturedEvent;
        let mut string_rep: *const c_char = std::ptr::null();

        cuda_get_error_string(error_code, &mut string_rep).unwrap();

        // Convert string_rep to Rust string for comparison
        let error_name = unsafe { CStr::from_ptr(string_rep) }.to_str().unwrap();
        assert_eq!(
            error_name,
            "operation not permitted on an event last recorded in a capturing stream"
        );
    }
}
