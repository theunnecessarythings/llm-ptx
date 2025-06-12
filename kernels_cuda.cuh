#ifndef LLMC_KERNELS_CUDA_H

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <unistd.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include "llmc/cuda_common.h"
#include "llmc/utils.h"

void cudaErrorCheck() {
  cudaCheck(cudaGetLastError());
  cudaCheck(cudaDeviceSynchronize());
}

// warp‐level reductions
__inline__ __device__ float warpReduceMax(float val) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2)
    val = max(val, __shfl_down_sync(0xffffffff, val, offset));
  return val;
}

__inline__ __device__ float warpReduceSum(float val) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2)
    val += __shfl_down_sync(0xffffffff, val, offset);
  return val;
}

// block‐wide reductions (assumes blockDim.x ≤ 1024)
__inline__ __device__ float blockReduceMax(float val) {
  static __shared__ float shared[32];
  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  val = warpReduceMax(val); // Each warp finds its max

  if (lane == 0) {
    shared[wid] = val; // Warp leaders write their max to shared memory
  }
  __syncthreads();

  // The first warp reduces the partial results from the warp leaders
  val = (wid == 0) ? shared[lane] : -CUDART_INF_F;
  val = warpReduceMax(val);

  // --- THE CORRECT BROADCAST ---
  // Only thread 0, which has the final answer, writes it to shared memory
  if (threadIdx.x == 0) {
    shared[0] = val;
  }
  // All threads wait for that write to complete
  __syncthreads();
  // All threads now read the same, correct final value
  return shared[0];
}

__inline__ __device__ float blockReduceSum(float val) {
  static __shared__ float shared[32];
  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  val = warpReduceSum(val); // Each warp finds its sum

  if (lane == 0) {
    shared[wid] = val; // Warp leaders write their sum to shared memory
  }
  __syncthreads();

  // The first warp reduces the partial results from the warp leaders
  val = (wid == 0) ? shared[lane] : 0.0f;
  val = warpReduceSum(val);

  // --- THE CORRECT BROADCAST ---
  // Only thread 0, which has the final answer, writes it to shared memory
  if (threadIdx.x == 0) {
    shared[0] = val;
  }
  // All threads wait for that write to complete
  __syncthreads();
  // All threads now read the same, correct final value
  return shared[0];
}

__global__ void encoder_fwd_kernel(float *out, const int *inp, const float *wte,
                                   const float *wpe, int B, int T, int C) {
  int b = blockIdx.y;
  int t = blockIdx.x;
  int c = threadIdx.x;
  if (b < B && t < T && c < C) {
    float *out_bt = out + b * T * C + t * C;
    int ix = inp[b * T + t];
    const float *wte_ix = wte + ix * C;
    const float *wpe_t = wpe + t * C;
    out_bt[c] = wte_ix[c] + wpe_t[c];
  }
}

__global__ void encoder_fwd_kernel_vec(float *out, const int *inp,
                                       const float *wte, const float *wpe,
                                       int B, int T, int C) {
  int b = blockIdx.y;
  int t = blockIdx.x;
  // Each thread now handles 4 float elements, so our grid of threads is smaller
  int c_vec = threadIdx.x;

  // The starting index for the 4 floats this thread will handle
  int c_start = c_vec * 4;

  // Boundary check
  if (b < B && t < T && c_start < C) {
    // Get the token index for this position
    int ix = inp[b * T + t];

    // Get base pointers to the correct rows in the embedding tables
    const float *wte_row = wte + ix * C;
    const float *wpe_row = wpe + t * C;
    float *out_row = out + b * T * C + t * C;

    // Cast pointers to float4 pointers to load 128 bits at once
    const float4 *wte_ptr = reinterpret_cast<const float4 *>(wte_row + c_start);
    const float4 *wpe_ptr = reinterpret_cast<const float4 *>(wpe_row + c_start);
    float4 *out_ptr = reinterpret_cast<float4 *>(out_row + c_start);

    // Load the vectorized data
    float4 wte_val = *wte_ptr;
    float4 wpe_val = *wpe_ptr;

    // Perform the additions component-wise
    float4 out_val;
    out_val.x = wte_val.x + wpe_val.x;
    out_val.y = wte_val.y + wpe_val.y;
    out_val.z = wte_val.z + wpe_val.z;
    out_val.w = wte_val.w + wpe_val.w;

    // Write the vectorized result back to global memory
    *out_ptr = out_val;
  }
}

void encoder_forward(float *out, int *inp, float *wte, float *wpe, int B, int T,
                     int C) {
  // C must be divisible by 4 for this to work
  assert(C % 4 == 0);

  dim3 grid(T, B);
  // We launch C/4 threads, as each thread now handles 4 elements
  dim3 block(std::min(1024, C / 4));

  encoder_fwd_kernel_vec<<<grid, block>>>(out, (int *)inp, wte, wpe, B, T, C);
  cudaErrorCheck();
}

__global__ void layernorm_fwd_kernel(float *out, const float *inp,
                                     const float *weight, const float *bias,
                                     int C, float eps) {
  extern __shared__ float shared_buffer[];

  int bt = blockIdx.x;
  const float *x = inp + bt * C;
  float *y = out + bt * C;

  int tid = threadIdx.x;
  int block_size = blockDim.x;

  // --- Parallel Mean Calculation ---
  float sum = 0.0f;
  for (int i = tid; i < C; i += block_size) {
    sum += x[i];
  }
  shared_buffer[tid] = sum;
  __syncthreads();

  // Reduction in shared memory
  for (int stride = block_size / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared_buffer[tid] += shared_buffer[tid + stride];
    }
    __syncthreads();
  }
  float mean = shared_buffer[0] / C;
  __syncthreads(); // Ensure mean is visible to all

  // --- Parallel Variance Calculation ---
  sum = 0.0f;
  for (int i = tid; i < C; i += block_size) {
    float diff = x[i] - mean;
    sum += diff * diff;
  }
  shared_buffer[tid] = sum;
  __syncthreads();

  // Reduction in shared memory
  for (int stride = block_size / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared_buffer[tid] += shared_buffer[tid + stride];
    }
    __syncthreads();
  }
  float var = shared_buffer[0] / C;
  float rstd = rsqrtf(var + eps);

  // --- Final Application ---
  for (int i = tid; i < C; i += block_size) {
    float n = (x[i] - mean) * rstd;
    y[i] = n * weight[i] + bias[i];
  }
}

void layernorm_forward(float *out, const float *inp, const float *weight,
                       const float *bias, int B, int T, int C,
                       cudaStream_t stream = 0) {
  const float eps = 1e-5f;
  int N = B * T;

  int blocks = N;
  // Threads must be a power of 2 for the reduction to work. 256 is good.
  int threads = 1024;

  // Shared memory for the reduction (one float per thread)
  size_t shared_mem_size = threads * sizeof(float);

  layernorm_fwd_kernel<<<blocks, threads, shared_mem_size, stream>>>(
      out, inp, weight, bias, C, eps);

  cudaErrorCheck();
}

__global__ void add_bias_kernel(float *out, const float *bias, int M, int N) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < M && col < N) {
    out[row * N + col] += bias[col];
  }
}

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemmVectorize(int M, int N, int K, float alpha, float *A,
                               float *B, float beta, float *C) {
  const uint cRow = blockIdx.y; // Block row index
  const uint cCol = blockIdx.x; // Block col index

  // Each thread is responsible for a TM x TN sub-tile of the output.
  // threadRow/threadCol is the thread's index within the 2D grid of threads in
  // a block.
  const int threadCol = threadIdx.x % (BN / TN);
  const int threadRow = threadIdx.x / (BN / TN);

  // Shared memory for tiles of A and B
  __shared__ float As[BM * BK];
  __shared__ float Bs[BN * BK];

  // Move global memory pointers to the start of the respective tiles
  A += cRow * BM * K;
  B += cCol * BN * K;
  C += cRow * BM * N + cCol * BN;

  // Threads collaboratively load tiles from global to shared memory.
  // Each thread loads 4 elements (1x float4) at a time.
  const uint innerRowA = threadIdx.x / (BK / 4);
  const uint innerColA = threadIdx.x % (BK / 4);
  const uint innerRowB = threadIdx.x / (BK / 4);
  const uint innerColB = threadIdx.x % (BK / 4);

  float threadResults[TM * TN] = {0.0f};

  // Loop over the K dimension in tiles of size BK
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // Load a tile of A and a tile of B into shared memory.
    // Both A and B are transposed on-the-fly during the load.
    float4 tmpA =
        reinterpret_cast<float4 *>(&A[innerRowA * K + innerColA * 4])[0];
    As[(innerColA * 4 + 0) * BM + innerRowA] = tmpA.x;
    As[(innerColA * 4 + 1) * BM + innerRowA] = tmpA.y;
    As[(innerColA * 4 + 2) * BM + innerRowA] = tmpA.z;
    As[(innerColA * 4 + 3) * BM + innerRowA] = tmpA.w;

    float4 tmpB =
        reinterpret_cast<float4 *>(&B[innerRowB * K + innerColB * 4])[0];
    Bs[(innerColB * 4 + 0) * BN + innerRowB] = tmpB.x;
    Bs[(innerColB * 4 + 1) * BN + innerRowB] = tmpB.y;
    Bs[(innerColB * 4 + 2) * BN + innerRowB] = tmpB.z;
    Bs[(innerColB * 4 + 3) * BN + innerRowB] = tmpB.w;

    __syncthreads();

    // Advance global memory pointers to the next tile in the K dimension
    A += BK;
    B += BK;

// Main calculation loop.
// This computes the dot product of tiles from As and Bs.
#pragma unroll
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      float regM[TM];
      float regN[TN];

      // Load a slice of As and Bs into registers
      for (uint i = 0; i < TM; ++i) {
        regM[i] = As[dotIdx * BM + threadRow * TM + i];
      }
      for (uint i = 0; i < TN; ++i) {
        regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
      }

      // Compute outer product and accumulate in thread-local results
      for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
          threadResults[resIdxM * TN + resIdxN] +=
              regM[resIdxM] * regN[resIdxN];
        }
      }
    }
    __syncthreads();
  }

  // Write results from registers back to global memory
  for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
    for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
      // Calculate the row and column in the output C tile for this thread
      uint c_row = threadRow * TM + resIdxM;
      uint c_col = threadCol * TN + resIdxN;

      C[c_row * N + c_col] = threadResults[resIdxM * TN + resIdxN];
    }
  }
}

__device__ float4 ld_vec(const float *address) {
  return *reinterpret_cast<const float4 *>(address);
}

__device__ void st_vec(float *address, float4 val) {
  *reinterpret_cast<float4 *>(address) = val;
}

__global__ void __launch_bounds__(16 * 16)
    matmul_fwd_kernel(float *out, const float *inp, const float *weight,
                      const float *bias, int C, int OC) {
  // out is (B,T,OC). OC is short for "output channels", e.g. OC = 4 * C
  // inp is (B,T,C), weight is (OC, C), bias is (OC)
  // each thread handles 8x8 elements; each block 128 by 128 elements.
  int oc = 8 * (blockIdx.y * blockDim.y + threadIdx.y);

  // buffers to cache chunks of the input matrices
  __shared__ float lhs_s[128][32];
  __shared__ float rhs_s[128][32];

  // adjust our pointers for the current block
  inp += 128 * blockIdx.x * C;
  weight += 128 * blockIdx.y * C;
  out += 128 * blockIdx.x * OC + 128 * blockIdx.y;

  float vals[8][8] = {};
  if (bias != NULL) {
    for (int i = 0; i < 8; i++) {
      for (int j = 0; j < 8; j += 4) {
        float4 b = ld_vec(bias + oc + j);
        vals[i][j + 0] = b.x;
        vals[i][j + 1] = b.y;
        vals[i][j + 2] = b.z;
        vals[i][j + 3] = b.w;
      }
    }
  }

  int si_start = 4 * (16 * threadIdx.y + threadIdx.x);
  for (int so = 0; so < C; so += 32) {
    __syncthreads();
    int xmod8 = threadIdx.x % 8;
    int xby8 = threadIdx.x / 8;
    int xo = 4 * xmod8;
    for (int y = 2 * threadIdx.y + xby8; y < 128; y += 32) {
      st_vec(&lhs_s[y][xo], ld_vec(inp + y * C + so + xo));
      st_vec(&rhs_s[y][xo], ld_vec(weight + y * C + so + xo));
    }
    __syncthreads();

    for (int si = si_start; si < si_start + 32; si += 4) {
      float4 rhs[8];
      for (int u = 0; u < 8; ++u) {
        rhs[u] = ld_vec(&rhs_s[u + 8 * threadIdx.y][si % 32]);
      }

      for (int ii = 0; ii < 8; ++ii) {
        float4 lhs = ld_vec(&lhs_s[ii + 8 * threadIdx.x][si % 32]);
        for (int ji = 0; ji < 8; ++ji) {
          vals[ii][ji] += lhs.x * rhs[ji].x;
          vals[ii][ji] += lhs.y * rhs[ji].y;
          vals[ii][ji] += lhs.z * rhs[ji].z;
          vals[ii][ji] += lhs.w * rhs[ji].w;
        }
      }
    }
  }

  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 8; j += 4) {
      float4 result;
      result.x = vals[i][j + 0];
      result.y = vals[i][j + 1];
      result.z = vals[i][j + 2];
      result.w = vals[i][j + 3];
      st_vec(out + (8 * threadIdx.x + i) * OC + 8 * threadIdx.y + j, result);
    }
  }
}

// handwritten, relatively efficient non-tensorcore matmul kernel
void matmul_forward(float *out, const float *inp, const float *weight,
                    const float *bias, int B, int T, int C, int OC) {
  // out is (B,T,OC). OC is short for "output channels", e.g. OC = 4 * C
  // inp is (B,T,C), weight is (OC, C), bias is (OC)
  auto sqrt_block_size = 16;

  dim3 gridDim(CEIL_DIV(B * T, 8 * sqrt_block_size),
               CEIL_DIV(OC, 8 * sqrt_block_size));
  dim3 blockDim(sqrt_block_size, sqrt_block_size);
  matmul_fwd_kernel<<<gridDim, blockDim>>>(out, inp, weight, bias, C, OC);
  cudaCheck(cudaGetLastError());
}
__global__ void attention_fwd_kernel(float *out, float *preatt, float *att,
                                     const float *inp, int B, int T, int C,
                                     int NH) {
  // Each thread block is for one head and one batch item: grid(NH, B)
  // Each thread is for one query token: block(T)
  int h = blockIdx.x;
  int b = blockIdx.y;
  int t = threadIdx.x;

  if (b >= B || h >= NH || t >= T)
    return;

  int C3 = C * 3;
  int hs = C / NH; // head size
  float scale = 1.0f / sqrtf((float)hs);

  // Pointer to the input for this batch item
  const float *inp_b = inp + b * T * C3;
  // Pointer to the query vector for this thread (b, t, h)
  const float *query_t = inp_b + t * C3 + h * hs;

  // Pointers to the output attention scores for this thread's row
  float *preatt_bth = preatt + (b * NH * T * T) + (h * T * T) + (t * T);
  float *att_bth = att + (b * NH * T * T) + (h * T * T) + (t * T);

  // --- Pass 1: Calculate Q.K^T and find maxval (causal attention) ---
  // Each thread finds its OWN maxval, no block reduction needed.
  float maxval = -10000.0f;
  for (int t2 = 0; t2 <= t; t2++) {
    // Pointer to the key vector for position t2
    const float *key_t2 = inp_b + t2 * C3 + h * hs + C; // +C offset for key

    // Dot product
    float val = 0.0f;
    for (int i = 0; i < hs; i++) {
      val += query_t[i] * key_t2[i];
    }
    val *= scale;
    if (val > maxval) {
      maxval = val;
    }
    preatt_bth[t2] = val;
  }

  // --- Pass 2: Calculate exponentials and sum for the softmax denominator
  // Each thread calculates its OWN sum, no block reduction.
  float expsum = 0.0f;
  for (int t2 = 0; t2 <= t; t2++) {
    // Subtract maxval for numerical stability
    float expv = expf(preatt_bth[t2] - maxval);
    expsum += expv;
    att_bth[t2] = expv; // Store the numerator temporarily
  }
  float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

  // --- Pass 3: Normalize to get final softmax scores ---
  for (int t2 = 0; t2 <= t; t2++) {
    att_bth[t2] *= expsum_inv;
  }
  // Explicitly zero out future tokens
  for (int t2 = t + 1; t2 < T; t2++) {
    att_bth[t2] = 0.0f;
  }

  // --- Pass 4: Accumulate weighted values into the output ---
  float *out_bth = out + (b * T * C) + (t * C) + (h * hs);
  for (int i = 0; i < hs; i++) {
    out_bth[i] = 0.0f;
  }
  for (int t2 = 0; t2 <= t; t2++) {
    // Pointer to the value vector for position t2
    const float *value_t2 =
        inp_b + t2 * C3 + h * hs + C * 2; // +2C offset for value
    float att_score = att_bth[t2];
    for (int i = 0; i < hs; i++) {
      out_bth[i] += att_score * value_t2[i];
    }
  }
}

// The calling wrapper function for the kernel
void attention_forward(float *out, float *preatt, float *att, const float *inp,
                       int B, int T, int C, int NH, cudaStream_t stream = 0) {
  dim3 grid(NH, B);
  dim3 block(T);
  // No shared memory is needed for this corrected, simpler version
  attention_fwd_kernel<<<grid, block, 0, stream>>>(out, preatt, att, inp, B, T,
                                                   C, NH);
  cudaErrorCheck();
}
#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)

__global__ void gelu_fwd_kernel(float *__restrict__ out,
                                const float *__restrict__ inp, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N)
    return;
  float x = inp[i];
  float cube = 0.044715f * x * x * x;
  out[i] = 0.5f * x * (1.0f + tanhf(GELU_SCALING_FACTOR * (x + cube)));
}

__global__ void residual_fwd_kernel(float *__restrict__ out,
                                    const float *__restrict__ inp1,
                                    const float *__restrict__ inp2, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N)
    return;
  out[i] = inp1[i] + inp2[i];
}

static const int THREADS_PER_BLOCK = 1024;

void gelu_forward(float *out, const float *inp, int N,
                  cudaStream_t stream = 0) {
  int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  gelu_fwd_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(out, inp, N);
  cudaErrorCheck();
}

void residual_forward(float *out, const float *inp1, const float *inp2, int N,
                      cudaStream_t stream = 0) {
  int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  residual_fwd_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(out, inp1, inp2,
                                                                N);
  cudaErrorCheck();
}
__global__ void
softmax_fwd_kernel(float *__restrict__ probs,        // [B*T][Vp]
                   const float *__restrict__ logits, // [B*T][Vp]
                   int B, int T, int V, int Vp) {
  int bt = blockIdx.x; // in [0..B*T)
  int N = B * T;
  if (bt >= N)
    return;

  const float *logits_bt = logits + bt * Vp;
  float *probs_bt = probs + bt * Vp;

  int tid = threadIdx.x;
  int threads = blockDim.x;

  // 1) find max over real vocab [0..V)
  float local_max = -CUDART_INF_F;
  for (int i = tid; i < V; i += threads) {
    local_max = fmaxf(local_max, logits_bt[i]);
  }
  float maxval = blockReduceMax(local_max);
  __syncthreads();

  // 2) compute exp(logit - maxval) and partial sum
  float local_sum = 0.0f;
  for (int i = tid; i < V; i += threads) {
    float e = expf(logits_bt[i] - maxval);
    probs_bt[i] = e;
    local_sum += e;
  }
  float sum = blockReduceSum(local_sum);
  __syncthreads();

  float inv_sum = (sum > 0.0f ? 1.0f / sum : 0.0f);

  // 3) normalize the real-vocab probabilities
  for (int i = tid; i < V; i += threads) {
    probs_bt[i] *= inv_sum;
  }
  // 4) zero out the padded entries [V..Vp)
  for (int i = V + tid; i < Vp; i += threads) {
    probs_bt[i] = 0.0f;
  }
}

void softmax_forward(float *probs, const float *logits, int B, int T, int V,
                     int Vp, cudaStream_t stream = 0) {
  int N = B * T;
  int threads = std::min(1024, Vp);
  int blocks = N;
  softmax_fwd_kernel<<<blocks, threads, 0, stream>>>(probs, logits, B, T, V,
                                                     Vp);
  cudaErrorCheck();
}

#endif // LLMC_KERNELS_CUDA_H
