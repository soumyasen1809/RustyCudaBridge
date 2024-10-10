#![allow(non_camel_case_types)]
use libc::{c_int, size_t};
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

#[link(name = "cuda")]
extern "C" {
    // https://developer.download.nvidia.com/compute/DevZone/docs/html/C/doc/html/group__CUDART__MEMORY_gedaeb2708ad3f74d5b417ee1874ec84a.html#gedaeb2708ad3f74d5b417ee1874ec84a
    pub fn cudaFreeHost(ptr: *mut c_void) -> cudaError_t;
}

#[link(name = "cuda")]
extern "C" {
    // https://developer.download.nvidia.com/compute/DevZone/docs/html/C/doc/html/group__CUDART__MEMORY_g15a3871f15f8c38f5b7190946845758c.html#g15a3871f15f8c38f5b7190946845758c
    pub fn cudaHostAlloc(
        pHost: *mut *mut c_void,
        size: size_t,
        flags: cudaHostAllocFlag,
    ) -> cudaError_t;
}

#[link(name = "cuda")]
extern "C" {
    // https://developer.download.nvidia.com/compute/DevZone/docs/html/C/doc/html/group__CUDART__MEMORY_ge07c97b96efd09abaeb3ca3b5f8da4ee.html#ge07c97b96efd09abaeb3ca3b5f8da4ee
    pub fn cudaMemset(devPtr: *mut c_void, value: c_int, count: size_t) -> cudaError_t;
}

#[repr(C)]
#[derive(Debug, PartialEq)]
pub enum cudaError_t {
    // https://developer.download.nvidia.com/compute/DevZone/docs/html/C/doc/html/group__CUDART__TYPES_g3f51e3575c2178246db0a94a430e0038.html#g3f51e3575c2178246db0a94a430e0038
    cudaSuccess = 0,
    cudaErrorInvalidValue = 1,
    cudaErrorMemoryAllocation = 2,
    cudaErrorInvalidDevicePointer = 3,
    cudaErrorInitializationError = 4,
}

#[repr(C)]
pub enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4,
}

#[repr(C)]
pub enum cudaHostAllocFlag {
    cudaHostAllocDefault = 0,
    cudaHostAllocPortable = 1,
    cudaHostAllocMapped = 2,
    cudaHostAllocWriteCombined = 3,
}
