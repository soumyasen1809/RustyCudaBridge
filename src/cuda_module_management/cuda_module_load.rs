use std::ffi::CString;

use crate::{cuda_bindings::*, cuda_errors::cudaError_t};

pub fn cuda_module_load(ptx_path: &str, hmod: &mut *mut CUmod_st) -> Result<(), cudaError_t> {
    let bytes = CString::new(ptx_path).expect("Failed to convert path to CString");

    let result = unsafe { cuModuleLoad(hmod as *mut CUmodule, bytes.as_ptr()) };
    println!("module loaded: {:?}", hmod); // hmod should not be 0x0

    match result {
        cudaError_t::cudaSuccess => Ok(()),
        _ => Err(result),
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use crate::{
        cuda_memory_management::cuda_malloc::cuda_malloc,
        cuda_module_management::cuda_module_load::cuda_module_load,
    };

    #[test]
    fn test_cuda_get_function() {
        cuda_malloc(
            &mut std::ptr::null_mut(),
            1 as usize * std::mem::size_of::<i32>(),
        )
        .unwrap(); // Note: removing cuda_malloc causes Issue in module_load: cudaErrorInitializationError (Why?)

        let ptx_path: &str = "all_cuda_kernels/add.ptx"; // of PTX file
        if !Path::new(ptx_path).exists() {
            panic!("PTX file not found at ptx_path: {}", ptx_path);
        }

        let mut hmod = std::ptr::null_mut();
        cuda_module_load(ptx_path, &mut hmod).expect("Issue in module_load");
        assert_ne!(hmod, std::ptr::null_mut()); // hmod can not be 0x0 if module loaded properly
    }
}
