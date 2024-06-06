use candle_core::{
    backend::BackendStorage,
    cuda::{
        cudarc::driver::{CudaSlice, DeviceRepr, LaunchAsync, LaunchConfig},
        WrapErr,
    },
    CpuStorage, CudaDevice, CustomOp1, Layout, Result, Shape, Tensor, WithDType,
};
struct NonZero {}
impl NonZero {
    fn nonzero<T: WithDType>(&self, vs: &[T], layout: &Layout) -> Vec<u32> {
        println!("layout.dims(): {:?}", layout.dims());
        let n = layout.dims().len();
        let mut result = Vec::new();
        for (i, v) in vs.iter().enumerate() {
            if !v.is_zero() {
                //result.push(i as u32);
                let mut idx = i;
                let mut indices = vec![0u32; n];
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
        fn fwd<T: WithDType + DeviceRepr>(
            slice: &CudaSlice<T>,
            layout: &Layout,
            dev: CudaDevice,
            func_name: &str,
        ) -> Result<(candle_core::CudaStorage, Shape)> {
            let slice = match layout.contiguous_offsets() {
                None => candle_core::bail!("input has to be contiguous"),
                Some((o1, o2)) => slice.slice(o1..o2),
            };
            let elem_count = layout.shape().elem_count();
            let dst = unsafe { dev.alloc::<u32>(elem_count) }.w()?;
            let func = dev.get_or_load_func(func_name, llava_kernels::CUDA)?;
            let params = (&slice, &dst);
            let cfg = LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe { func.launch(cfg, params) }.w()?;
            let dst = candle_core::CudaStorage::wrap_cuda_slice(dst, dev);
            Ok((dst, layout.shape().clone()))
        }
        let dev = storage.device().clone();
        let func_name = match storage.dtype() {
            candle_core::DType::U8 => "cuda_nonzero_u8",
            candle_core::DType::U32 => "cuda_nonzero_u32",
            candle_core::DType::I64 => "cuda_nonzero_i64",
            candle_core::DType::BF16 => "cuda_nonzero_bf16",
            candle_core::DType::F16 => "cuda_nonzero_f16",
            candle_core::DType::F32 => "cuda_nonzero_f32",
            candle_core::DType::F64 => "cuda_nonzero_f64",
        };
        match storage.dtype() {
            candle_core::DType::U8 => {
                let slice = storage.as_cuda_slice::<u8>()?;
                fwd(slice, layout, dev, func_name)
            }
            candle_core::DType::U32 => {
                let slice = storage.as_cuda_slice::<u32>()?;
                fwd(slice, layout, dev, func_name)
            }
            candle_core::DType::I64 => {
                let slice = storage.as_cuda_slice::<i64>()?;
                fwd(slice, layout, dev, func_name)
            }
            candle_core::DType::BF16 => {
                todo!()
            }
            candle_core::DType::F16 => {
                todo!()
            }
            candle_core::DType::F32 => {
                let slice = storage.as_cuda_slice::<f32>()?;
                fwd(slice, layout, dev, func_name)
            }
            candle_core::DType::F64 => {
                let slice = storage.as_cuda_slice::<f64>()?;
                fwd(slice, layout, dev, func_name)
            }
        }
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

#[cfg(test)]
mod tests {
    use crate::ops::NonZeroOp;
    use candle_core::Tensor;
    #[test]
    fn test_nonzero() {
        let device = candle_core::Device::Cpu;
        let a = Tensor::from_vec(
            vec![0u32, 1, 0, 2, 0, 3, 1, 0, 0, 2, 4, 5],
            &[2, 2, 3],
            &device,
        )
        .unwrap();
        let b = a.nonzero().unwrap();
        println!("a: {}\n nonzero: {}", a, b);
    }

    #[test]
    fn test_nonzero_cuda() {
        let device = candle_core::Device::new_cuda(0).unwrap();
        let a = Tensor::from_vec(
            vec![0u32, 1, 0, 2, 0, 3, 1, 0, 0, 2, 4, 5],
            &[2, 2, 3],
            &device,
        )
        .unwrap();
        let _ = a.nonzero().unwrap();
        let b = a.to_dtype(candle_core::DType::F32).unwrap();
        let _ = b.nonzero().unwrap();
    }
}
