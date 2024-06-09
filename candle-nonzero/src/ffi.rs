use std::ffi::c_void;

extern "C" {
    pub fn count_nonzero_bf16(d_in: *const c_void, N: u32) -> u32;
    pub fn count_nonzero_f16(d_in: *const c_void, N: u32) -> u32;
    pub fn count_nonzero_f32(d_in: *const c_void, N: u32) -> u32;
    pub fn count_nonzero_f64(d_in: *const c_void, N: u32) -> u32;
    pub fn count_nonzero_u8(d_in: *const c_void, N: u32) -> u32;
    pub fn count_nonzero_u32(d_in: *const c_void, N: u32) -> u32;
    pub fn count_nonzero_i64(d_in: *const c_void, N: u32) -> u32;
    pub fn nonzero_bf16(
        d_in: *const c_void,
        N: u32,
        num_nonzero: u32,
        dims: *const c_void,
        num_dims: u32,
        d_out: *mut c_void,
    );
    pub fn nonzero_f16(
        d_in: *const c_void,
        N: u32,
        num_nonzero: u32,
        dims: *const c_void,
        num_dims: u32,
        d_out: *mut c_void,
    );
    pub fn nonzero_f32(
        d_in: *const c_void,
        N: u32,
        num_nonzero: u32,
        dims: *const c_void,
        num_dims: u32,
        d_out: *mut c_void,
    );
    pub fn nonzero_f64(
        d_in: *const c_void,
        N: u32,
        num_nonzero: u32,
        dims: *const c_void,
        num_dims: u32,
        d_out: *mut c_void,
    );
    pub fn nonzero_u8(
        d_in: *const c_void,
        N: u32,
        num_nonzero: u32,
        dims: *const c_void,
        num_dims: u32,
        d_out: *mut c_void,
    );
    pub fn nonzero_u32(
        d_in: *const c_void,
        N: u32,
        num_nonzero: u32,
        dims: *const c_void,
        num_dims: u32,
        d_out: *mut c_void,
    );
    pub fn nonzero_i64(
        d_in: *const c_void,
        N: u32,
        num_nonzero: u32,
        dims: *const c_void,
        num_dims: u32,
        d_out: *mut c_void,
    );
}
