use libc::c_char;

use crate::{cuda_bindings::*, cuda_errors::cudaError_t};

pub fn cuda_module_get_function(
    hfunc: *mut CUfunction,
    hmod: CUmodule,
    name: *const c_char,
) -> Result<(), cudaError_t> {
    let result = unsafe { cuModuleGetFunction(hfunc, hmod, name as *const c_char) };

    match result {
        cudaError_t::cudaSuccess => Ok(()),
        _ => Err(result),
    }
}

#[cfg(test)]
mod tests {
    use std::{ffi::CString, path::Path};

    use crate::{
        cuda_functions::cuda_malloc::cuda_malloc,
        cuda_module_management::cuda_module_load::cuda_module_load,
    };

    use super::*;

    #[test]
    fn test_cuda_get_function() {
        cuda_malloc(
            &mut std::ptr::null_mut(),
            1 as usize * std::mem::size_of::<i32>(),
        )
        .unwrap(); // Note: removing cuda_malloc causes Issue in module_load: cudaErrorInitializationError (Why?)

        let mut f: CUfunction = std::ptr::null_mut();
        let kernel_name = "vec_add"; // from PTX file
        let ptx_path: &str = "all_cuda_kernels/add.ptx"; // of PTX file
        if !Path::new(ptx_path).exists() {
            panic!("PTX file not found at ptx_path: {}", ptx_path);
        }

        let mut hmod = std::ptr::null_mut();
        cuda_module_load(ptx_path, &mut hmod).expect("Issue in module_load");
        assert_ne!(hmod, std::ptr::null_mut()); // hmod can not be 0x0 if module loaded properly

        let name_string = CString::new(kernel_name).expect("Issue in name_string"); // name of kernel from PTX: vec_add
        let name = name_string.as_ptr();
        cuda_module_get_function(&mut f as *mut CUfunction, hmod, name)
            .expect("Issue in get_function");
        assert_ne!(f, std::ptr::null_mut()); // f can not be 0x0 if module loaded properly
    }
}
