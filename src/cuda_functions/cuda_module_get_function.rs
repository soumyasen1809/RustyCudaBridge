use libc::c_char;

use super::cuda_bindings::{cuModuleGetFunction, cudaError_t, CUfunction, CUmodule};

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
