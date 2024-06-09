mod ffi;

use std::{ffi::c_void, ops::Deref};

use candle_core::{
    backend::BackendStorage,
    cuda::{
        cudarc::driver::{CudaSlice, DevicePtr, DeviceRepr, LaunchAsync, LaunchConfig},
        WrapErr,
    },
    CpuStorage, CustomOp1, Error, Layout, Result, Shape, Tensor, WithDType,
};
use half::{bf16, f16};

struct NonZero {}
impl NonZero {
    fn nonzero<T: WithDType>(&self, vs: &[T], layout: &Layout) -> Vec<u32> {
        println!("layout.dims(): {:?}", layout.dims());
        let n = layout.dims().len();
        let mut result = Vec::new();
        let mut indices = vec![0u32; n];
        for (i, v) in vs.iter().enumerate() {
            if !v.is_zero() {
                //result.push(i as u32);
                let mut idx = i;
                for (dim_index, dim) in layout.dims().iter().enumerate().rev() {
                    let d = idx % dim;
                    indices[dim_index] = d as u32;
                    idx /= dim;
                }
                result.extend_from_slice(&indices);
            }
        }
        result
    }
}

impl CustomOp1 for NonZero {
    fn name(&self) -> &'static str {
        "nonzero"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        let result = match storage {
            candle_core::CpuStorage::U8(vs) => self.nonzero(vs, layout),
            candle_core::CpuStorage::U32(vs) => self.nonzero(vs, layout),
            candle_core::CpuStorage::I64(vs) => self.nonzero(vs, layout),
            candle_core::CpuStorage::BF16(vs) => self.nonzero(vs, layout),
            candle_core::CpuStorage::F16(vs) => self.nonzero(vs, layout),
            candle_core::CpuStorage::F32(vs) => self.nonzero(vs, layout),
            candle_core::CpuStorage::F64(vs) => self.nonzero(vs, layout),
        };
        let index_len = layout.dims().len();
        let result_len = result.len() / index_len;
        let result = CpuStorage::U32(result);
        let shape = Shape::from_dims(&[result_len, index_len]);
        Ok((result, shape))
    }

    fn cuda_fwd(
        &self,
        storage: &candle_core::CudaStorage,
        layout: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        let dev = storage.device().clone();
        let d_in = match storage.dtype() {
            candle_core::DType::U8 => *storage.as_cuda_slice::<u8>()?.device_ptr(),
            candle_core::DType::U32 => *storage.as_cuda_slice::<u32>()?.device_ptr(),
            candle_core::DType::I64 => *storage.as_cuda_slice::<i64>()?.device_ptr(),
            candle_core::DType::BF16 => *storage.as_cuda_slice::<bf16>()?.device_ptr(),
            candle_core::DType::F16 => *storage.as_cuda_slice::<f16>()?.device_ptr(),
            candle_core::DType::F32 => *storage.as_cuda_slice::<f32>()?.device_ptr(),
            candle_core::DType::F64 => *storage.as_cuda_slice::<f64>()?.device_ptr(),
        } as *const c_void;
        let n = layout.shape().elem_count();
        let num_nonzeros = unsafe { ffi::count_nonzero_f32(d_in, n as u32) };
        let d_out = unsafe {
            dev.alloc::<u32>(num_nonzeros as usize * layout.dims().len())
                .map_err(|e| {
                    Error::Msg("Failed to allocate memory for nonzero result".to_string())
                })?
        };
        let dims = layout.dims().as_ptr() as *const c_void;
    }
}

trait NonZeroOp {
    fn nonzero(&self) -> Result<Tensor>;
}

impl NonZeroOp for Tensor {
    fn nonzero(&self) -> Result<Tensor> {
        if !self.is_contiguous() {
            return Err(candle_core::Error::RequiresContiguous { op: "nonzero" });
        }
        self.apply_op1_no_bwd(&NonZero {})
    }
}

#[test]
fn test_nonzero_cuda() {
    use std::ffi::c_void;
    use std::ops::Deref;

    use crate::ffi::count_nonzero_f32;
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;
    use candle_core::Tensor;
    let device = candle_core::Device::new_cuda(0).unwrap();
    let a = Tensor::from_vec(
        vec![1f32, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0],
        &[4, 2],
        &device,
    )
    .unwrap();
    let (storage, _) = a.storage_and_layout();
    let d_in = match storage.deref() {
        candle_core::Storage::Cuda(storage) => {
            *storage.as_cuda_slice::<f32>().unwrap().device_ptr()
        }
        _ => unreachable!("Expected CudaStorage"),
    } as *const c_void;
    let n = 8;
    let count = unsafe { count_nonzero_f32(d_in, n) };
    println!("count: {}", count);
}
