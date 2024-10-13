use crate::cuda_functions::cuda_bindings::{cuModuleUnload, cudaError_t, CUmod_st};

pub fn cuda_module_unload(hmod: *mut CUmod_st) -> Result<(), cudaError_t> {
    let result = unsafe { cuModuleUnload(hmod) };
    // cuda_module_unload() does not automatically set the module handle (hmod) to null_mut() after unloading.
    // The cuda_module_unload() function in the CUDA Driver API only unloads the module from the context,
    // but it does not modify or reset the handle (hmod) you passed in.
    // The handle itself remains unchanged, and its value still points to the address of the previously loaded module,
    // even though it is no longer valid after unloading.

    match result {
        cudaError_t::cudaSuccess => Ok(()),
        _ => Err(result),
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use crate::cuda_functions::{
        cuda_malloc::cuda_malloc, cuda_module_load::cuda_module_load,
        cuda_module_unload::cuda_module_unload,
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

        let result = cuda_module_unload(hmod);
        assert!(result.is_ok());
    }
}
