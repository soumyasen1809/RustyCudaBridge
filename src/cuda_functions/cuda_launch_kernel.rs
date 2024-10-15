use std::ffi::{c_uint, c_void};

use crate::{cuda_bindings::*, cuda_errors::cudaError_t};

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

#[cfg(test)]
mod tests {
    use std::{ffi::CString, path::Path};

    use crate::{
        cuda_functions::{
            cuda_free::cuda_free, cuda_malloc::cuda_malloc, cuda_memcpy::cuda_memcpy,
        },
        cuda_module_management::{
            cuda_module_get_function::cuda_module_get_function, cuda_module_load::cuda_module_load,
        },
    };

    use super::*;

    #[test]
    fn test_cuda_launch_kernel() {
        let a = vec![1, 2, 3, 4, 5];
        let b = vec![6, 7, 8, 9, 10];
        let n = a.len() as i32;

        let c = vec![0; n as usize];

        // Allocate memory on the device
        let mut dev_a: *mut u8 = std::ptr::null_mut(); // Note that the type needs to specified as *mut u8
        let mut dev_b: *mut u8 = std::ptr::null_mut(); // else, it becomes *mut *mut u8
        let mut dev_c: *mut u8 = std::ptr::null_mut();

        cuda_malloc(&mut dev_a, n as usize * std::mem::size_of::<i32>()).unwrap(); // we need to pass *mut *mut u8 to cuda_malloc
        cuda_malloc(&mut dev_b, n as usize * std::mem::size_of::<i32>()).unwrap(); // hence, we do &mut dev_a
        cuda_malloc(&mut dev_c, n as usize * std::mem::size_of::<i32>()).unwrap(); // since dev_a is specified as *mut u8

        // Copy vec from host to device
        cuda_memcpy(
            dev_a,
            a.as_ptr() as *const u8,
            n as usize * std::mem::size_of::<i32>(), // IMP: The size needs to be multiplied by std::mem::size_of::<i32>()
            cudaMemcpyKind::cudaMemcpyHostToDevice,
        )
        .unwrap();
        cuda_memcpy(
            dev_b,
            b.as_ptr() as *const u8,
            n as usize * std::mem::size_of::<i32>(),
            cudaMemcpyKind::cudaMemcpyHostToDevice,
        )
        .unwrap();
        cuda_memcpy(
            dev_c,
            c.as_ptr() as *const u8,
            n as usize * std::mem::size_of::<i32>(),
            cudaMemcpyKind::cudaMemcpyHostToDevice,
        )
        .unwrap();

        // cuLaunchKernel
        let mut f: CUfunction = std::ptr::null_mut();
        let h_stream: CUstream = std::ptr::null_mut();
        let kernel_name = "vec_add"; // from PTX file
        let ptx_path: &str = "all_cuda_kernels/add.ptx"; // of PTX file
        if !Path::new(ptx_path).exists() {
            panic!("PTX file not found at ptx_path: {}", ptx_path);
        }

        let mut hmod = std::ptr::null_mut();
        cuda_module_load(ptx_path, &mut hmod).expect("Issue in module_load");

        let name_string = CString::new(kernel_name).expect("Issue in name_string"); // name of kernel from PTX: vec_add
        let name = name_string.as_ptr();
        cuda_module_get_function(&mut f as *mut CUfunction, hmod, name)
            .expect("Issue in get_function");

        let param_array: &[*mut i32] = &[
            &mut dev_a as *mut _ as *mut i32,
            &mut dev_b as *mut _ as *mut i32,
            &mut dev_c as *mut _ as *mut i32,
            &mut (n as i32) as *mut _ as *mut i32, // `n` is passed by value
        ];

        let grid_dim_x = 16;
        let grid_dim_y = 1;
        let grid_dim_z = 1;
        let block_dim_x = 256;
        let block_dim_y = 1;
        let block_dim_z = 1;
        let shared_mem_bytes = 0;
        let kernel_params = param_array.as_ptr() as *mut _;
        let extra: *mut *mut i32 = std::ptr::null_mut();
        cuda_launch_kernel(
            f,
            grid_dim_x,
            grid_dim_y,
            grid_dim_z,
            block_dim_x,
            block_dim_y,
            block_dim_z,
            shared_mem_bytes,
            h_stream,
            kernel_params,
            extra as *mut _,
        )
        .expect("Issue in launch_kernel");

        // Copy vec from device to host
        cuda_memcpy(
            c.as_ptr() as *mut u8,
            dev_c,
            n as usize * std::mem::size_of::<i32>(),
            cudaMemcpyKind::cudaMemcpyDeviceToHost,
        )
        .unwrap();

        // Free cuda memory
        cuda_free(dev_a).unwrap();
        cuda_free(dev_b).unwrap();
        cuda_free(dev_c).unwrap();

        let expected = vec![7, 9, 11, 13, 15];
        assert_eq!(c, expected);
    }
}
