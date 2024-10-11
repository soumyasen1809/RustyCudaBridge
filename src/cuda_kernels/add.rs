// The #[link] attribute links your Rust code with the libadd.so shared library.

use crate::cuda_functions::{
    cuda_bindings::cudaError_t, cuda_free::cuda_free, cuda_malloc::cuda_malloc,
    cuda_memcpy::cuda_memcpy,
};
#[link(name = "add", kind = "dylib")]
extern "C" {
    fn launch_vec_add(a: *const i32, b: *const i32, c: *mut i32, n: i32);
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

    // Call the kernel
    unsafe {
        launch_vec_add(
            dev_a as *const i32,
            dev_b as *const i32,
            dev_c as *mut i32,
            n,
        );
    }

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

    println!("{:?}", c);

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
