#include<stdint.h>
#include "cuda_utils.cuh"

template<typename T>
__device__ void cuda_nonzero(const T * x, uint32_t * dst){
    printf("Hello World from GPU!\n");
}
/*
extern "C" __global__ void cuda_nonzero_u32(const uint32_t * x, uint32_t * dst ) { 
    cuda_nonzero(x, dst); 
} 
*/

#define NONZERO_OP(TYPENAME, RUST_NAME) \
extern "C" __global__ void cuda_nonzero_##RUST_NAME( \
    const TYPENAME * x, uint32_t * dst \
) { \
    cuda_nonzero(x, dst); \
} \

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
