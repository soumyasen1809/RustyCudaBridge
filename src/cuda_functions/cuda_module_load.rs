use std::ffi::CString;

use crate::cuda_functions::cuda_bindings::{cuModuleLoad, cudaError_t, CUmod_st, CUmodule};

pub fn cuda_module_load(ptx_path: &str, hmod: &mut *mut CUmod_st) -> Result<(), cudaError_t> {
    let bytes = CString::new(ptx_path).expect("Failed to convert path to CString");

    let result = unsafe { cuModuleLoad(hmod as *mut CUmodule, bytes.as_ptr()) };

    println!("module: {:?}", hmod); // hmod should not be 0x0

    match result {
        cudaError_t::cudaSuccess => Ok(()),
        _ => Err(result),
    }
}
