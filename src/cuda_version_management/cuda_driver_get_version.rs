use libc::c_int;

use crate::{cuda_bindings::cuDriverGetVersion, cuda_errors::cudaError_t};

pub fn cuda_driver_get_version(driver_version: *mut i32) -> Result<(), cudaError_t> {
    let result = unsafe { cuDriverGetVersion(&mut *driver_version as *mut c_int) };

    match result {
        cudaError_t::cudaSuccess => Ok(()),
        _ => Err(result),
    }
}

pub fn get_cuda_version_string(driver_version: i32) -> String {
    let minor_version = (driver_version % 100) / 10;
    let major_version = driver_version / 1000;

    let version_string = major_version.to_string() + &"." + &minor_version.to_string();
    version_string
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_driver_get_version() {
        let mut driver_version = 0;
        cuda_driver_get_version(&mut driver_version).expect("Issue in getting driver version");
        assert_eq!(driver_version, 12030);

        let cuda_version_string = get_cuda_version_string(driver_version);
        assert_eq!(cuda_version_string, String::from("12.3"));
    }
}
