#![allow(non_camel_case_types)]
use libc::size_t;
use std::ffi::c_void;

#[link(name = "cuda")]
extern "C" {
    // https://developer.download.nvidia.com/compute/DevZone/docs/html/C/doc/html/group__CUDART__MEMORY_g48efa06b81cc031b2aa6fdc2e9930741.html#g48efa06b81cc031b2aa6fdc2e9930741
    pub fn cudaMemcpy(
        dst: *mut c_void,
        src: *const c_void,
        count: size_t,
        kind: cudaMemcpyKind,
    ) -> cudaError_t;
}
#[link(name = "cuda")]
extern "C" {
    // https://developer.download.nvidia.com/compute/DevZone/docs/html/C/doc/html/group__CUDART__MEMORY_gc63ffd93e344b939d6399199d8b12fef.html
    pub fn cudaMalloc(devPtr: *mut *mut c_void, size: size_t) -> cudaError_t;
}

#[link(name = "cuda")]
extern "C" {
    // https://developer.download.nvidia.com/compute/DevZone/docs/html/C/doc/html/group__CUDART__MEMORY_gb17fef862d4d1fefb9dba35bd62a187e.html#gb17fef862d4d1fefb9dba35bd62a187e
    pub fn cudaFree(devPtr: *mut c_void) -> cudaError_t;
}

#[repr(C)]
#[derive(Debug, PartialEq)]
pub enum cudaError_t {
    cudaSuccess = 0,
    cudaErrorInvalidValue = 1,
    cudaErrorMemoryAllocation = 2,
}

#[repr(C)]
pub enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4,
}
