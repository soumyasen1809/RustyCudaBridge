use std::ffi::{c_uint, c_void};

use super::cuda_bindings::{cuLaunchKernel, cudaError_t, CUfunction, CUstream};

pub fn cuda_launch_kernel(
    f: CUfunction,
    grid_dim_x: i32,
    grid_dim_y: i32,
    grid_dim_z: i32,
    block_dim_x: i32,
    block_dim_y: i32,
    block_dim_z: i32,
    shared_mem_bytes: i32,
    h_stream: CUstream,
    kernel_params: *mut *mut i32,
    extra: *mut *mut i32,
) -> Result<(), cudaError_t> {
    let result = unsafe {
        cuLaunchKernel(
            f,
            grid_dim_x as c_uint,
            grid_dim_y as c_uint,
            grid_dim_z as c_uint,
            block_dim_x as c_uint,
            block_dim_y as c_uint,
            block_dim_z as c_uint,
            shared_mem_bytes as c_uint,
            h_stream,
            kernel_params as *mut *mut c_void,
            extra as *mut *mut c_void,
        )
    };

    match result {
        cudaError_t::cudaSuccess => Ok(()),
        _ => Err(result),
    }
}
