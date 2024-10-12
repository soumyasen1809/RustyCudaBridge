// The #[link] attribute links your Rust code with the libadd.so shared library.

use std::{
    ffi::{c_uint, c_void, CString},
    path::Path,
};

use libc::c_char;

use crate::cuda_functions::{
    cuda_bindings::{
        cuLaunchKernel, cuModuleGetFunction, cuModuleLoad, cudaError_t, CUfunction, CUmodule,
        CUstream,
    },
    cuda_free::cuda_free,
    cuda_malloc::cuda_malloc,
    cuda_memcpy::cuda_memcpy,
};

// #[link(name = "add", kind = "dylib")]
// extern "C" {
//     fn launch_vec_add(a: *const i32, b: *const i32, c: *mut i32, n: i32);
// }

pub fn launch_kernel(
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

pub fn get_function(
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

pub fn cuda_vec_add(a: &Vec<i32>, b: &Vec<i32>, n: i32) -> Result<Vec<i32>, cudaError_t> {
    assert!(a.len() == b.len());
    assert!(a.len() == n as usize);
    let c = vec![0; n as usize];

    // Allocate memory on the device
    let mut dev_a: *mut u8 = std::ptr::null_mut(); // Note that the type needs to specified as *mut u8
    let mut dev_b: *mut u8 = std::ptr::null_mut(); // else, it becomes *mut *mut u8
    let mut dev_c: *mut u8 = std::ptr::null_mut();

    cuda_malloc(&mut dev_a, n as usize * std::mem::size_of::<i32>())?; // we need to pass *mut *mut u8 to cuda_malloc
    cuda_malloc(&mut dev_b, n as usize * std::mem::size_of::<i32>())?; // hence, we do &mut dev_a
    cuda_malloc(&mut dev_c, n as usize * std::mem::size_of::<i32>())?; // since dev_a is specified as *mut u8

    // Copy vec from host to device
    cuda_memcpy(
        dev_a,
        a.as_ptr() as *const u8,
        n as usize * std::mem::size_of::<i32>(), // IMP: The size needs to be multiplied by std::mem::size_of::<i32>()
        crate::cuda_functions::cuda_bindings::cudaMemcpyKind::cudaMemcpyHostToDevice,
    )?;
    cuda_memcpy(
        dev_b,
        b.as_ptr() as *const u8,
        n as usize * std::mem::size_of::<i32>(),
        crate::cuda_functions::cuda_bindings::cudaMemcpyKind::cudaMemcpyHostToDevice,
    )?;
    cuda_memcpy(
        dev_c,
        c.as_ptr() as *const u8,
        n as usize * std::mem::size_of::<i32>(),
        crate::cuda_functions::cuda_bindings::cudaMemcpyKind::cudaMemcpyHostToDevice,
    )?;

    // cuLaunchKernel
    let mut f: CUfunction = std::ptr::null_mut();
    let h_stream: CUstream = std::ptr::null_mut();

    // param_array: ChatGPT help
    // https://chatgpt.com/share/670afffd-cf64-8008-b96b-18b867d50200
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

    let name_string = CString::new("vec_add").expect("Issue in name_string"); // name of kernel from PTX: vec_add
    let name = name_string.as_ptr();
    let mut hmod = std::ptr::null_mut();

    let path: &str = "all_cuda_kernels/add.ptx";
    if !Path::new(path).exists() {
        panic!("PTX file not found at path: {}", path);
    }
    let bytes = CString::new(path).expect("Failed to convert path to CString");
    let _ = unsafe {
        let result = cuModuleLoad(&mut hmod as *mut CUmodule, bytes.as_ptr());
        match result {
            cudaError_t::cudaSuccess => Ok(()),
            _ => Err(result),
        }
    };
    println!("module: {:?}", hmod); // hmod should not be 0x00

    get_function(&mut f as *mut CUfunction, hmod, name).expect("Issue in get_function");

    launch_kernel(
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
        crate::cuda_functions::cuda_bindings::cudaMemcpyKind::cudaMemcpyDeviceToHost,
    )?;

    // Free cuda memory
    cuda_free(dev_a)?;
    cuda_free(dev_b)?;
    cuda_free(dev_c)?;

    Ok(c)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec_add() {
        let a = vec![1, 2, 3, 4, 5];
        let b = vec![6, 7, 8, 9, 10];
        let n = a.len() as i32;

        let c = cuda_vec_add(&a, &b, n).unwrap();

        let expected = vec![7, 9, 11, 13, 15];
        assert_eq!(c, expected);
    }
}
