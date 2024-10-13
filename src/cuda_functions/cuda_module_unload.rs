use crate::cuda_functions::cuda_bindings::{cuModuleUnload, cudaError_t, CUmod_st, CUmodule};

pub fn cuda_module_unload(hmod: &mut *mut CUmod_st) -> Result<(), cudaError_t> {
    let result = unsafe { cuModuleUnload(hmod as *mut CUmodule) };

    match result {
        cudaError_t::cudaSuccess => Ok(()),
        _ => Err(result),
    }
}
