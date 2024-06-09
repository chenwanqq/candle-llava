#include <cstdio>
#include <cub/cub.cuh>
#include <iostream>
#include <stdint.h>

template <typename T> struct NonZeroOp {
  __host__ __device__ __forceinline__ bool operator()(const T &a) const {
    return (a != T(0));
  }
};

// count the number of non-zero elements in an array, to better allocate memory
template <typename T>
void count_nonzero(const T *d_in, const size_t N, size_t *h_out) {
  cub::TransformInputIterator<bool, NonZeroOp<T>, const T *> itr(
      d_in, NonZeroOp<T>());
  size_t temp_storage_bytes = 0;
  size_t *d_num_nonzero;
  cudaMalloc((void **)&d_num_nonzero, sizeof(size_t));
  cub::DeviceReduce::Sum(nullptr, temp_storage_bytes, itr, d_num_nonzero, N);
  void **d_temp_storage;
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, itr,
                         d_num_nonzero, N);
  cudaMemcpy(h_out, d_num_nonzero, sizeof(size_t), cudaMemcpyDeviceToHost);
  cudaFree(d_num_nonzero);
  cudaFree(d_temp_storage);
}

#define COUNT_NONZERO_OP(TYPENAME, RUST_NAME)                                  \
  extern "C" size_t count_nonzero_##RUST_NAME(const TYPENAME *d_in,          \
                                                size_t N) {                  \
    size_t result;                                                           \
    count_nonzero(d_in, N, &result);                                           \
    return result;                                                             \
  }

//#if __CUDA_ARCH__ >= 800
COUNT_NONZERO_OP(__nv_bfloat16, bf16)
//#endif

//#if __CUDA_ARCH__ >= 530
COUNT_NONZERO_OP(__half, f16)
//#endif

COUNT_NONZERO_OP(float, f32)
COUNT_NONZERO_OP(double, f64)
COUNT_NONZERO_OP(uint8_t, u8)
COUNT_NONZERO_OP(uint32_t, u32)
COUNT_NONZERO_OP(int64_t, i64)

__global__ void transform_indices(const size_t *temp_indices,
                                  const size_t num_nonzero,
                                  const size_t *dims, const size_t num_dims,
                                  size_t *d_out) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_nonzero) {
    int temp_index = temp_indices[idx];
    for (int i = num_dims - 1; i >= 0; i--) {
      d_out[idx * num_dims + i] = temp_index % dims[i];
      temp_index /= dims[i];
    }
  }
}

// get the indices of non-zero elements in an array
template <typename T>
void nonzero(const T *d_in, const size_t N, const size_t num_nonzero,
             const size_t *dims, const size_t num_dims, size_t *d_out) {
  cub::TransformInputIterator<bool, NonZeroOp<T>, const T *> itr(
      d_in, NonZeroOp<T>());
  cub::CountingInputIterator<size_t> counting_itr(0);
  size_t *out_temp;
  size_t *num_selected_out;
  cudaMalloc((void **)&out_temp, num_nonzero * sizeof(size_t));
  cudaMalloc((void **)&num_selected_out, sizeof(size_t));
  size_t temp_storage_bytes = 0;
  cub::DeviceSelect::Flagged(nullptr, temp_storage_bytes, counting_itr, itr,
                             out_temp, num_selected_out, N);
  void **d_temp_storage;
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, counting_itr,
                             itr, out_temp, num_selected_out, (int)N);
  const int nthreads = 256;
  const int nblocks = (num_nonzero + nthreads - 1) / nthreads;
  transform_indices<<<nblocks, nthreads>>>(out_temp, num_nonzero, dims,
                                           num_dims, d_out);
  cudaDeviceSynchronize();
  cudaFree(out_temp);
  cudaFree(d_temp_storage);
  cudaFree(num_selected_out);
}

#define NONZERO_OP(TYPENAME, RUST_NAME)                                        \
  extern "C" void nonzero_##RUST_NAME(                                         \
      const TYPENAME *d_in, size_t N, size_t num_nonzero,                 \
      const size_t *dims, size_t num_dims, size_t *d_out) {              \
    nonzero(d_in, N, num_nonzero, dims, num_dims, d_out);                     \
  }

//#if __CUDA_ARCH__ >= 800
NONZERO_OP(__nv_bfloat16, bf16)
//#endif

//#if __CUDA_ARCH__ >= 530
NONZERO_OP(__half, f16)
//#endif

NONZERO_OP(float, f32)
NONZERO_OP(double, f64)
NONZERO_OP(uint8_t, u8)
NONZERO_OP(uint32_t, u32)
NONZERO_OP(int64_t, i64)