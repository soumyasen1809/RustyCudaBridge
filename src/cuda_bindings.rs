#![allow(non_camel_case_types)]
use libc::{c_char, size_t};
use std::ffi::{c_int, c_uint, c_void};

use crate::{cuda_device_attributes::CUdevice_attribute, cuda_errors::cudaError_t};

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

#[link(name = "cuda")]
extern "C" {
    // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html#group__CUDA__EXEC_1gb8f3dc3031b40da29d5f9a7139e52e15
    pub fn cuLaunchKernel(
        f: CUfunction,
        gridDimX: c_uint,
        gridDimY: c_uint,
        gridDimZ: c_uint,
        blockDimX: c_uint,
        blockDimY: c_uint,
        blockDimZ: c_uint,
        sharedMemBytes: c_uint,
        hStream: CUstream,
        kernelParams: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> cudaError_t;
}

#[link(name = "cuda")]
extern "C" {
    // https://developer.download.nvidia.cn/compute/DevZone/docs/html/C/doc/html/group__CUDA__MODULE_ga52be009b0d4045811b30c965e1cb2cf.html
    pub fn cuModuleGetFunction(
        hfunc: *mut CUfunction,
        hmod: CUmodule,
        name: *const c_char,
    ) -> cudaError_t;
}

#[link(name = "cuda")]
extern "C" {
    // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html#group__CUDA__MODULE_1g366093bd269dafd0af21f1c7d18115d3
    pub fn cuModuleLoad(module: *mut CUmodule, name: *const c_char) -> cudaError_t;
}

#[link(name = "cuda")]
extern "C" {
    // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html#group__CUDA__MODULE_1g8ea3d716524369de3763104ced4ea57b
    pub fn cuModuleUnload(module: CUmodule) -> cudaError_t;
}

#[link(name = "cuda")]
extern "C" {
    // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__ERROR.html#group__CUDA__ERROR_1g2c4ac087113652bb3d1f95bf2513c468
    pub fn cuGetErrorName(error: cudaError_t, pStr: *mut *const c_char) -> cudaError_t;
}

#[link(name = "cuda")]
extern "C" {
    // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__ERROR.html#group__CUDA__ERROR_1g72758fcaf05b5c7fac5c25ead9445ada
    pub fn cuGetErrorString(error: cudaError_t, pStr: *mut *const c_char) -> cudaError_t;
}

#[link(name = "cuda")]
extern "C" {
    // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__INITIALIZE.html#group__CUDA__INITIALIZE_1g0a2f1517e1bd8502c7194c3a8c134bc3
    pub fn cuInit(Flags: c_uint) -> cudaError_t;
}

#[link(name = "cuda")]
extern "C" {
    // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VERSION.html#group__CUDA__VERSION_1g8b7a10395392e049006e61bcdc8ebe71
    pub fn cuDriverGetVersion(driverVersion: *mut c_int) -> cudaError_t;
}

pub type CUdevice = i32;

#[link(name = "cuda")]
extern "C" {
    // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g8bdd1cc7201304b01357b8034f6587cb
    pub fn cuDeviceGet(device: *mut CUdevice, ordinal: c_int) -> cudaError_t;
}

#[link(name = "cuda")]
extern "C" {
    // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g9c3e1414f0ad901d3278a4d6645fc266
    pub fn cuDeviceGetAttribute(
        pi: *mut c_int,
        attrib: CUdevice_attribute,
        dev: CUdevice,
    ) -> cudaError_t;
}

#[link(name = "cuda")]
extern "C" {
    // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g8bdd1cc7201304b01357b8034f6587cb
    pub fn cuDeviceGetCount(count: *mut c_int) -> cudaError_t;
}

#[link(name = "cuda")]
extern "C" {
    // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE_1gef75aa30df95446a845f2a7b9fffbb7f
    pub fn cuDeviceGetName(name: *mut c_char, len: c_int, dev: CUdevice) -> cudaError_t;
}

#[link(name = "cuda")]
extern "C" {
    // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE_1gc8bca3c97a78816303b8aa5773b741f2
    pub fn cuDeviceGetDefaultMemPool(pool_out: *mut CUmemoryPool, dev: CUdevice) -> cudaError_t;
}

#[link(name = "cuda")]
extern "C" {
    // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE_1gdf186e9559d53a5eb18e572d48c1121b
    pub fn cuDeviceGetMemPool(pool_out: *mut CUmemoryPool, dev: CUdevice) -> cudaError_t;
}

#[link(name = "cuda")]
extern "C" {
    // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g4f2f276b84d9c2eaefdc76d6274db4a0
    pub fn cuDeviceSetMemPool(dev: CUdevice, pool_out: CUmemoryPool) -> cudaError_t;
}

#[link(name = "cuda")]
extern "C" {
    // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE_1gc6a0d6551335a3780f9f3c967a0fde5d
    pub fn cuDeviceTotalMem(bytes: *mut size_t, dev: CUdevice) -> cudaError_t;
}

#[link(name = "cuda")]
extern "C" {
    // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PRIMARY__CTX.html#group__CUDA__PRIMARY__CTX_1g65f3e018721b6d90aa05cfb56250f469
    pub fn cuDevicePrimaryCtxGetState(
        dev: CUdevice,
        flags: *mut c_uint,
        active: *mut c_int,
    ) -> cudaError_t;
}

#[link(name = "cuda")]
extern "C" {
    // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PRIMARY__CTX.html#group__CUDA__PRIMARY__CTX_1gf2a8bc16f8df0c88031f6a1ba3d6e8ad
    pub fn cuDevicePrimaryCtxRelease(dev: CUdevice) -> cudaError_t;
}

#[link(name = "cuda")]
extern "C" {
    // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PRIMARY__CTX.html#group__CUDA__PRIMARY__CTX_1g5d38802e8600340283958a117466ce12
    pub fn cuDevicePrimaryCtxReset(dev: CUdevice) -> cudaError_t;
}

#[link(name = "cuda")]
extern "C" {
    // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PRIMARY__CTX.html#group__CUDA__PRIMARY__CTX_1gd779a84f17acdad0d9143d9fe719cfdf
    pub fn cuDevicePrimaryCtxSetFlags(dev: CUdevice, flags: c_uint) -> cudaError_t;
}

#[link(name = "cuda")]
extern "C" {
    // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g65dc0012348bc84810e2103a40d8e2cf
    pub fn cuCtxCreate(pctx: *mut CUcontext, flags: c_uint, dev: CUdevice) -> cudaError_t;
}

#[link(name = "cuda")]
extern "C" {
    // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g27a365aebb0eb548166309f58a1e8b8e
    pub fn cuCtxDestroy(pctx: *mut CUcontext) -> cudaError_t;
}

#[link(name = "cuda")]
extern "C" {
    // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g088a90490dafca5893ef6fbebc8de8fb
    pub fn cuCtxGetApiVersion(ctx: CUcontext, version: *mut c_uint) -> cudaError_t;
}

#[link(name = "cuda")]
extern "C" {
    // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g8f13165846b73750693640fb3e8380d0
    pub fn cuCtxGetCurrent(pctx: *mut CUcontext) -> cudaError_t;
}

#[link(name = "cuda")]
extern "C" {
    // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g2fac188026a062d92e91a8687d0a7902
    pub fn cuCtxPopCurrent(pctx: *mut CUcontext) -> cudaError_t;
}

#[link(name = "cuda")]
extern "C" {
    // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1gb02d4c850eb16f861fe5a29682cc90ba
    pub fn cuCtxPushCurrent(ctx: CUcontext) -> cudaError_t;
}

#[link(name = "cuda")]
extern "C" {
    // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g7a54725f28d34b8c6299f0c6ca579616
    pub fn cuCtxSynchronize() -> cudaError_t;
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

#[repr(C)]
pub struct CUstream_st(c_void);
// https://docs.rs/cuda-sys/latest/cuda_sys/cuda/struct.CUstream_st.html
pub type CUstream = *mut CUstream_st;

#[repr(C)]
pub struct CUfunc_st(c_void);
//https://docs.rs/cuda-driver-sys/latest/cuda_driver_sys/struct.CUfunc_st.html
pub type CUfunction = *mut CUfunc_st;

#[repr(C)]
pub struct CUmod_st(c_void); // Using c_void abstracts away the underlying implementation details
                             //                                // of CUfunction. You don't need to know the exact structure or contents
                             //                                // of a CUfunction object in Rust.
pub type CUmodule = *mut CUmod_st;

#[repr(C)]
pub struct CUmemPoolHandle_st(c_void);
pub type CUmemoryPool = *mut CUmemPoolHandle_st;

#[repr(C)]
pub struct CUctx_st(c_void);
pub type CUcontext = *mut CUctx_st;
