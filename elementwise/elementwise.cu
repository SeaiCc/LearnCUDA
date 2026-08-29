#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <vector>

#define WARP_SIZE 32
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])

// grid -> block -> thread
// gridDim
// blockDim: num of thread in one block

// FP32
// ElementWise ADD grid(N/256)
// block(256) a: Nx1 , b: Nx1 , c: Nx1, c = elementwise_add(a, b)
__global__ void elementwise_add_f32_kernel(float *a, float *b, float *c, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < N) 
    c[idx] = a[idx] + b[idx];
}


// FP32 x 4
// grid(N/256), block(256/4)
// a: Nx1 b:Nx1 c:Nx1, c = elementwise_add(a, b)
__global__ void elementwise_add_f32x4_kernel(float *a, float *b, float *c, int N) {
  int idx = 4 * (blockDim.x * blockIdx.x + threadIdx.x);
  if((idx + 3) < N) {
    float4 reg_a = FLOAT4(a[idx]);
    float4 reg_b = FLOAT4(b[idx]);
    float4 reg_c;
    reg_c.x = reg_a.x + reg_b.x;
    reg_c.y = reg_a.y + reg_b.y;
    reg_c.z = reg_a.z + reg_b.z;
    reg_c.w = reg_a.w + reg_b.w;
    FLOAT4(c[idx]) = reg_c;
  } else if(idx < N) {
    for (int i = 0; (idx + i) < N; i++) {
      c[idx + i] = a[idx + i] + b[idx + i];
    }
  }
}


#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)                                  \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if(((T).options().dtype() != (th_type))) {                                   \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("value must be " #th_type);                       \
  }

#define TORCH_BINDING_ELEM_ADD(packed_type, th_type, element_type, n_elements) \
  void elementwise_add_##packed_type(torch::Tensor a, torch::Tensor b,         \
                                     torch::Tensor c) {                        \
    CHECK_TORCH_TENSOR_DTYPE(a, (th_type))                                     \
    CHECK_TORCH_TENSOR_DTYPE(b, (th_type))                                     \
    CHECK_TORCH_TENSOR_DTYPE(c, (th_type))                                     \
    const int ndim = a.dim();                                                 \
    if (ndim != 2) {                                                          \
      int N = 1;                                                              \
      for (int i = 0; i < ndim; ++i) {                                        \
        N *= a.size(i);                                                       \
      }                                                                       \
      dim3 block(256 / (n_elements));                                         \
      dim3 grid((N+256-1)/256);                                               \
      elementwise_add_##packed_type##_kernel<<<grid, block>>>(                \
        reinterpret_cast<element_type *>(a.data_ptr()),                       \
        reinterpret_cast<element_type *>(b.data_ptr()),                       \
        reinterpret_cast<element_type *>(c.data_ptr()), N);                   \
    } else {                                                                  \
      const int S = a.size(0);                                                \
      const int K = a.size(1);                                                \
      const int N = S * K;                                                    \
      if ((K / (n_elements)) <= 1024) {                                       \
        dim3 block(K / (n_elements));                                         \
        dim3 grid(S);                                                         \
        elementwise_add_##packed_type##_kernel<<<grid, block>>>(              \
            reinterpret_cast<element_type *>(a.data_ptr()),                   \
            reinterpret_cast<element_type *>(b.data_ptr()),                   \
            reinterpret_cast<element_type *>(c.data_ptr()), N);               \
      } else {                                                                \
        int N = 1;                                                            \
        for (int i = 0; i < ndim; i++) {                                      \
          N *= a.size(i);                                                     \
        }                                                                     \
        dim3 block(256 / (n_elements));                                       \
        dim3 grid((N + 256 - 1) / 256);                                       \
        elementwise_add_##packed_type##_kernel<<<grid, block>>>(              \
            reinterpret_cast<element_type *>(a.data_ptr()),                   \
            reinterpret_cast<element_type *>(b.data_ptr()),                   \
            reinterpret_cast<element_type *>(c.data_ptr()), N);               \
      }                                                                       \
    }                                                                         \
  }

TORCH_BINDING_ELEM_ADD(f32, torch::kFloat32, float, 1)
TORCH_BINDING_ELEM_ADD(f32x4, torch::kFloat32, float, 4)



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f32)
  TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f32x4)
}
