use candle_nonzero::NonZeroOp;
use candle_core::Tensor;
use criterion::{black_box, criterion_group, criterion_main, Criterion};


fn criterion_benchmark(c: &mut Criterion) {
    let dev = candle_core::Device::new_cuda(0).unwrap();
    let cpu = candle_core::Device::Cpu;
    let n = 100000;
    let mut x = vec![0f32;n];
    for i in 0..n {
        if i % 6 == 0 {
            x[i] = 1.0;
        }
    }
    let cuda_tensor = Tensor::from_vec(x.clone(),&[n/8,2,2,2],&dev).unwrap();
    let cpu_tensor = Tensor::from_vec(x,&[n/8,2,2,2],&cpu).unwrap();
    c.bench_function("cuda nonzero", |b|b.iter(||{
        let x = cuda_tensor.nonzero().unwrap();
        black_box(x)
    }));
    c.bench_function("cpu nonzero", |b|b.iter(||{
        let x = cpu_tensor.nonzero().unwrap();
        black_box(x)
    }));
    c.bench_function("gpu-cpu-gpu nonzero", |b|b.iter(||{
        let x = cuda_tensor.to_device(&cpu).unwrap();
        let x = x.nonzero().unwrap();
        let x = x.to_device(&dev).unwrap();
        black_box(x) 
    }));
}


criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);