#include <cub/cub.cuh>
#include <stdint.h>

template <typename T> struct NonZeroOp {
  __host__ __device__ __forceinline__ bool operator()(const T &a) const {
    return (a != T(0));
  }
};

// count the number of non-zero elements in an array, to better allocate memory
template <typename T>
void count_nonzero(const T *d_in, const uint32_t N, uint32_t *h_out) {
  cub::TransformInputIterator<bool, NonZeroOp<T>, const T *> itr(
      d_in, NonZeroOp<T>());
  size_t temp_storage_bytes = 0;
  uint32_t *d_num_nonzeros;
  cudaMalloc((void **)&d_num_nonzeros, sizeof(uint32_t));
  cub::DeviceReduce::Sum(nullptr, temp_storage_bytes, itr, d_num_nonzeros, N);
  void **d_temp_storage;
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, itr,
                         d_num_nonzeros, N);
  cudaMemcpy(h_out, d_num_nonzeros, sizeof(uint32_t), cudaMemcpyDeviceToHost);
  cudaFree(d_num_nonzeros);
  cudaFree(d_temp_storage);
}

#define COUNT_NONZERO_OP(TYPENAME, RUST_NAME)                                  \
  extern "C" uint32_t count_nonzero_##RUST_NAME(const TYPENAME *d_in,          \
                                                uint32_t N) {                  \
    uint32_t result;                                                           \
    count_nonzero(d_in, N, &result);                                           \
    return result;                                                             \
  }

#if __CUDA_ARCH__ >= 800
COUNT_NONZERO_OP(__nv_bfloat16, bf16)
#endif

#if __CUDA_ARCH__ >= 530
COUNT_NONZERO_OP(__half, f16)
#endif

COUNT_NONZERO_OP(float, f32)
COUNT_NONZERO_OP(double, f64)
COUNT_NONZERO_OP(uint8_t, u8)
COUNT_NONZERO_OP(uint32_t, u32)
COUNT_NONZERO_OP(int64_t, i64)

__global__ void transform_indices(const uint32_t *temp_indices,
                                  const uint32_t num_nonzeros,
                                  const uint32_t *dims, const uint32_t num_dims,
                                  uint32_t *d_out) {
  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_nonzeros) {
    auto temp_index = temp_indices[idx];
    for (int i = num_dims - 1; i >= 0; i--) {
      d_out[idx * num_dims + i] = temp_index % dims[i];
      temp_index /= dims[i];
    }
  }
}

// get the indices of non-zero elements in an array
template <typename T>
void nonzero(const T *d_in, const uint32_t N, const uint32_t num_nonzeros,
             const uint32_t *dims, const uint32_t num_dims, uint32_t *d_out) {
  cub::TransformInputIterator<bool, NonZeroOp<T>, const T *> itr(
      d_in, NonZeroOp<T>());
  cub::CountingInputIterator<uint32_t> counting_itr(0);
  uint32_t *out_temp;
  uint32_t *num_selected_out;
  cudaMalloc((void **)&out_temp, num_nonzeros * sizeof(uint32_t));
  cudaMalloc((void **)&num_selected_out, sizeof(uint32_t));
  size_t temp_storage_bytes = 0;
  cub::DeviceSelect::Flagged(nullptr, temp_storage_bytes, counting_itr, itr,
                             out_temp, num_selected_out, N);
  void **d_temp_storage;
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, counting_itr,
                             itr, out_temp, num_selected_out, (int)N);
  const int nthreads = 256;
  const int nblocks = (num_nonzeros + nthreads - 1) / nthreads;
  // TODO: transform indices to d_out
  transform_indices<<<nblocks, nthreads>>>(out_temp, num_nonzeros, dims,
                                           num_dims, d_out);
  cudaFree(out_temp);
  cudaFree(d_temp_storage);
  cudaFree(num_selected_out);
}

#define NONZERO_OP(TYPENAME, RUST_NAME)                                        \
  extern "C" void nonzero_##RUST_NAME(                                         \
      const TYPENAME *d_in, uint32_t N, uint32_t num_nonzeros,                 \
      const uint32_t *dims, uint32_t num_dims, uint32_t *d_out) {              \
    nonzero(d_in, N, num_nonzeros, dims, num_dims, d_out);                     \
  }

#if __CUDA_ARCH__ >= 800
NONZERO_OP(__nv_bfloat16, bf16)
#endif

#if __CUDA_ARCH__ >= 530
NONZERO_OP(__half, f16)
#endif

NONZERO_OP(float, f32)
NONZERO_OP(double, f64)
NONZERO_OP(uint8_t, u8)
NONZERO_OP(uint32_t, u32)
NONZERO_OP(int64_t, i64)