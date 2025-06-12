#ifndef KERNELS_PTX_H

#include <cassert>
#include <cstdio>

#include "llmc/cuda_common.h"
#include <cuda.h>

inline void cudaErrorCheck() {
  cudaCheck(cudaGetLastError());
  cudaCheck(cudaDeviceSynchronize());
}

#define cuCheck(err)                                                           \
  do {                                                                         \
    CUresult res = (err);                                                      \
    if (res != CUDA_SUCCESS) {                                                 \
      const char *err_str;                                                     \
      cuGetErrorString(res, &err_str);                                         \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              err_str);                                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

void encoder_forward(float *out, const int *inp, const float *wte,
                     const float *wpe, int B, int T, int C,
                     cudaStream_t stream = 0) {
  static CUmodule module = nullptr;
  static CUfunction kernel = nullptr;

  assert(C % 4 == 0);
  if (module == nullptr) {
    const char *ptx_path = "ptx/encoder_kernel.ptx";
    cuCheck(cuModuleLoad(&module, ptx_path));
    cuCheck(cuModuleGetFunction(&kernel, module, "encoder_fwd_kernel"));
  }

  void *args[] = {&out, &inp, &wte, &wpe, &B, &T, &C};

  cuCheck(cuLaunchKernel(kernel, T, B,
                         1, // Grid dimensions (x,y,z)
                         std::min(1024, C / 4), 1,
                         1,         // Block dimensions (x,y,z)
                         0,         // Shared memory size in bytes
                         stream,    // Stream
                         args,      // Kernel arguments
                         nullptr)); // Extra options

  cudaErrorCheck();
}

void layernorm_forward(float *out, const float *inp, const float *weight,
                       const float *bias, int B, int T, int C,
                       cudaStream_t stream = 0) {
  static CUmodule module = nullptr;
  static CUfunction kernel = nullptr;

  if (module == nullptr) {
    const char *ptx_path = "ptx/layernorm_kernel.ptx";
    cuCheck(cuModuleLoad(&module, ptx_path));
    cuCheck(cuModuleGetFunction(&kernel, module, "layernorm_fwd_kernel"));
  }
  auto block_size = 32;
  assert(block_size % 32 == 0);
  assert(block_size <= 1024);
  int N = B * T;

  void *args[] = {&out, &inp, &weight, &bias, &N, &C};

  cuCheck(cuLaunchKernel(kernel, N, 1,
                         1,                // Grid dimensions (x,y,z)
                         block_size, 1, 1, // Block dimensions (x,y,z)
                         0,                // Shared memory size in bytes
                         stream,           // Stream
                         args,             // Kernel arguments
                         nullptr));        // Extra options

  cudaErrorCheck();
}

void matmul_forward(float *out, const float *inp, const float *weight,
                    const float *bias, int B, int T, int C, int OC,
                    cudaStream_t stream = 0) {
  static CUmodule module = nullptr;
  static CUfunction kernel = nullptr;

  if (module == nullptr) {
    const char *ptx_path = "ptx/matmul_kernel.ptx";
    cuCheck(cuModuleLoad(&module, ptx_path));
    cuCheck(cuModuleGetFunction(&kernel, module, "matmul_fwd_kernel"));
  }
  auto sqrt_block_size = 16;

  dim3 gridDim(CEIL_DIV(B * T, 8 * sqrt_block_size),
               CEIL_DIV(OC, 8 * sqrt_block_size));
  dim3 blockDim(sqrt_block_size, sqrt_block_size);

  void *args[] = {&out, &inp, &weight, &bias, &C, &OC};

  cuCheck(cuLaunchKernel(kernel, gridDim.x, gridDim.y,
                         1,                         // Grid dimensions (x,y,z)
                         blockDim.x, blockDim.y, 1, // Block dimensions (x,y,z)
                         0,         // Shared memory size in bytes
                         stream,    // Stream
                         args,      // Kernel arguments
                         nullptr)); // Extra options

  cudaErrorCheck();
}

void attention_forward(float *out, float *preatt, float *att, const float *inp,
                       int B, int T, int C, int NH, cudaStream_t stream = 0) {
  static CUmodule module = nullptr;
  static CUfunction kernel = nullptr;

  if (module == nullptr) {
    const char *ptx_path = "ptx/attention_kernel.ptx";
    cuCheck(cuModuleLoad(&module, ptx_path));
    cuCheck(cuModuleGetFunction(&kernel, module, "attention_fwd_kernel"));
  }

  void *args[] = {&out, &preatt, &att, &inp, &B, &T, &C, &NH};

  cuCheck(cuLaunchKernel(kernel, NH, B,
                         1,         // Grid dimensions (x,y,z)
                         T, 1, 1,   // Block dimensions (x,y,z)
                         0,         // Shared memory size in bytes
                         stream,    // Stream
                         args,      // Kernel arguments
                         nullptr)); // Extra options

  cudaErrorCheck();
}

static const int THREADS_PER_BLOCK = 1024;

void gelu_forward(float *out, const float *inp, int N,
                  cudaStream_t stream = 0) {
  static CUmodule module = nullptr;
  static CUfunction kernel = nullptr;

  if (module == nullptr) {
    const char *ptx_path = "ptx/gelu_kernel.ptx";
    cuCheck(cuModuleLoad(&module, ptx_path));
    cuCheck(cuModuleGetFunction(&kernel, module, "gelu_fwd_kernel"));
  }

  int blocks_per_grid = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  void *args[] = {&out, &inp, &N};

  cuCheck(cuLaunchKernel(kernel, blocks_per_grid, 1,
                         1,                       // Grid dimensions (x,y,z)
                         THREADS_PER_BLOCK, 1, 1, // Block dimensions (x,y,z)
                         0,                       // Shared memory size in bytes
                         stream,                  // Stream
                         args,                    // Kernel arguments
                         nullptr));               // Extra options

  cudaErrorCheck();
}

void residual_forward(float *out, const float *inp1, const float *inp2, int N,
                      cudaStream_t stream = 0) {
  static CUmodule module = nullptr;
  static CUfunction kernel = nullptr;

  if (module == nullptr) {
    const char *ptx_path = "ptx/residual_kernel.ptx";
    cuCheck(cuModuleLoad(&module, ptx_path));
    cuCheck(cuModuleGetFunction(&kernel, module, "residual_fwd_kernel"));
  }

  int blocks_per_grid = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  void *args[] = {&out, &inp1, &inp2, &N};

  cuCheck(cuLaunchKernel(kernel, blocks_per_grid, 1,
                         1,                       // Grid dimensions (x,y,z)
                         THREADS_PER_BLOCK, 1, 1, // Block dimensions (x,y,z)
                         0,                       // Shared memory size in bytes
                         stream,                  // Stream
                         args,                    // Kernel arguments
                         nullptr));               // Extra options

  cudaErrorCheck();
}

void softmax_forward(float *probs, const float *logits, int B, int T, int V,
                     int Vp, cudaStream_t stream = 0) {
  static CUmodule module = nullptr;
  static CUfunction kernel = nullptr;

  if (module == nullptr) {
    const char *ptx_path = "ptx/softmax_kernel.ptx";
    cuCheck(cuModuleLoad(&module, ptx_path));
    cuCheck(cuModuleGetFunction(&kernel, module, "softmax_fwd_kernel"));
  }

  void *args[] = {&probs, &logits, &B, &T, &V, &Vp};

  cuCheck(cuLaunchKernel(kernel, B * T, 1,
                         1, // Grid dimensions (x,y,z)
                         std::min(1024, Vp), 1,
                         1,         // Block dimensions (x,y,z)
                         0,         // Shared memory size in bytes
                         stream,    // Stream
                         args,      // Kernel arguments
                         nullptr)); // Extra options

  cudaErrorCheck();
}

#endif // KERNELS_PTX_H
